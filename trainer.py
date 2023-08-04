# standard imports
import os
import time
import shutil
import logging

# third-party imports
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Optimizer
from tensorboard.plugins import projector
from colorama import Fore, init
from tqdm import tqdm

# module imports
from model import Transformer
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy

# logging house-keeping
init(autoreset=True)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """Utility class to train a language model."""

    def __init__(
        self,
        model: Transformer,
        optimizer: Optimizer,
        criterion: LabelSmoothedCrossEntropy,
        train_loader: SequenceLoader,
        val_loader: SequenceLoader,
        log_dir: str = "logs",
    ):
        """
        Initializes the Trainer with TensorFlow objects.

        :param model: the Transformer model
        :param optimizer: keras Optimizer
        :param criterion: Smooth Label Cross-Entropy loss
        :param train_loader: the sequence loader in train configuration
        :param val_loader: the sequence loader in validation configuration
        :param log_dir: directory location for TensorBoard logging
        """
        logging.info(f"{Fore.GREEN}Initializing Trainer")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir

        if os.path.exists(log_dir):
            logging.info(f"{Fore.YELLOW}Flushing Logs")
            shutil.rmtree(log_dir)

        logging.info(f"{Fore.CYAN}Creating Summary Writer")
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def log_embeddings(self):
        """Logs the encoder and decoder embeddings to TensorBoard."""

        logging.info(f"{Fore.MAGENTA}Logging embeddings to projector")

        # Get embeddings
        encoder_embeddings = self.model.encoder.embedding.get_weights()[0]
        decoder_embeddings = self.model.decoder.embedding.get_weights()[0]

        weights_encoder = tf.Variable(encoder_embeddings, name="encoder_embeddings")
        weights_decoder = tf.Variable(decoder_embeddings, name="decoder_embeddings")

        # Get training checkpoint
        checkpoint = tf.train.Checkpoint(
            encoder_embedding=weights_encoder, decoder_embedding=weights_decoder
        )
        checkpoint.save(os.path.join(self.log_dir, "embedding.ckpt"))

        # Get the vocabulary from the data loader
        vocab = self.train_loader.get_vocabulary()

        # Write vocabulary to file
        with open(
            os.path.join(self.log_dir, "metadata.tsv"), "w", encoding="utf-8"
        ) as f:
            for word in vocab:
                f.write("{}\n".format(word))

        # Configure and write to projector
        config = projector.ProjectorConfig()

        embedding_encoder = config.embeddings.add()
        embedding_encoder.tensor_name = "encoder_embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding_encoder.metadata_path = "metadata.tsv"

        embedding_decoder = config.embeddings.add()
        embedding_decoder.tensor_name = "decoder_embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding_decoder.metadata_path = "metadata.tsv"

        projector.visualize_embeddings(self.log_dir, config)
        logging.info(f"{Fore.WHITE} Embeddings can be visualized on projector")

    def train(
        self,
        start_epoch: int,
        epochs: int,
        batches_per_step: int,
        print_frequency: int,
        save_every: int,
        path_to_checkpoint: str,
    ) -> None:
        """
        Trains the model for the number of specified epochs.

        :param path_to_checkpoint: path to the directory containing the checkpoints
        :param save_every: save every this many number of steps
        :param start_epoch: starting epoch
        :param epochs: total number of training epochs
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """

        if path_to_checkpoint:
            self.load_checkpoint(checkpoint_dir=path_to_checkpoint)

        for epoch in range(start_epoch, epochs):
            logging.info(f"{Fore.GREEN}Started training at epoch {epoch + 1}")
            self.train_loader.create_batches()
            logging.info(f"{Fore.BLUE}Created training batches")
            self.train_one_epoch(
                epoch, batches_per_step, print_frequency, epochs, save_every
            )
            logging.info(f"{Fore.MAGENTA}Finished training epoch {epoch + 1}")
            self.val_loader.create_batches()
            logging.info(f"{Fore.BLUE}Created validation batches")
            self.validate_one_epoch()

    def save_checkpoint(self, idx: int, epoch: int, prefix: str = "checkpoints"):
        """
        Saves the model weights.

        :param epoch: the current epoch
        :param idx: index for saving
        :param prefix: path prefix
        """
        logging.info(f"{Fore.GREEN}Saving model at step {idx} of epoch {epoch}")
        self.model.save_weights(
            f"{prefix}/transformer_checkpoint_{idx}_{epoch}/checkpoint",
            save_format="tf",
        )
        logging.info(f"{Fore.CYAN}Successfully saved weights")

    def load_checkpoint(self, checkpoint_dir: str):
        """
        Loads the model from the latest checkpoint.

        :param checkpoint_dir: path to the directory containing the checkpoints
        """
        logging.info(f"{Fore.GREEN}Loading model from {checkpoint_dir}")

        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        logging.info(f"{Fore.CYAN}Latest checkpoint at {checkpoint_path}")

        # Dummy data to build the model
        dummy_encoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_decoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_encoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)
        dummy_decoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)

        # Calling the model on the dummy data to build it
        self.model(
            encoder_sequences=dummy_encoder_sequences,
            decoder_sequences=dummy_decoder_sequences,
            encoder_sequence_lengths=dummy_encoder_sequence_lengths,
            decoder_sequence_lengths=dummy_decoder_sequence_lengths,
            training=False,
        )

        self.model.load_weights(checkpoint_path)
        logging.info(f"{Fore.YELLOW}Successfully loaded weights")

    def train_one_epoch(
        self,
        epoch: int,
        batches_per_step: int,
        print_frequency: int,
        epochs: int,
        save_every: int,
    ) -> None:
        """
        Trains the model for one epoch.

        :param save_every: save every this many number of steps
        :param epochs: total number of training epochs
        :param epoch: the current training epoch
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """
        step = 1
        data_time = Mean()
        step_time = Mean()
        losses = Mean()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (
            source_seqs,
            target_seqs,
            source_seq_lengths,
            target_seq_lengths,
        ) in enumerate(self.train_loader):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model(
                    encoder_sequences=source_seqs,
                    decoder_sequences=target_seqs,
                    encoder_sequence_lengths=source_seq_lengths,
                    decoder_sequence_lengths=target_seq_lengths,
                    training=True,
                )
                # Compute loss
                loss = self.criterion(
                    y_true=target_seqs[:, 1:],  # skip <BOS> tag for targets
                    y_pred=predictions[:, :-1, :],
                )  # skip <EOS> tag for predictions

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if (i + 1) % batches_per_step == 0:
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
                step += 1

            data_time.update_state(time.time() - start_data_time)
            losses.update_state(loss)

            if step % print_frequency == 0:
                step_time.update_state(time.time() - start_step_time)
                logging.info(
                    f"{Fore.GREEN}Epoch {epoch + 1}/{epochs} "
                    f"{Fore.BLUE}Batch {i + 1}/{self.train_loader.n_batches}-----"
                    f"{Fore.WHITE}Step {step}-----"
                    f"{Fore.MAGENTA}Data Time {data_time.result():.3f}-----"
                    f"{Fore.CYAN}Step Time {step_time.result():.3f}-----"
                    f"{Fore.GREEN}Loss {loss:.4f}-----"
                    f"{Fore.YELLOW}Average Loss {losses.result():.4f}"
                )
                with self.summary_writer.as_default():
                    tf.summary.scalar("Loss", losses.result(), step=step)
                    tf.summary.scalar(
                        "Learning Rate", self.optimizer.learning_rate(step), step=step
                    )

            if (i + 1) % save_every == 0:
                self.save_checkpoint(i + 1, epoch + 1)
                self.log_embeddings()

            start_data_time = time.time()
            start_step_time = time.time()
            losses.reset_states()
            data_time.reset_states()
            step_time.reset_states()

    def validate_one_epoch(self) -> None:
        """Validates the model over the validation loader."""
        losses = Mean()

        for i, (
            source_seqs,
            target_seqs,
            source_seq_lengths,
            target_seq_lengths,
        ) in enumerate(tqdm(self.val_loader, total=self.val_loader.n_batches)):
            # Forward pass
            predictions = self.model(
                encoder_sequences=source_seqs,
                decoder_sequences=target_seqs,
                encoder_sequence_lengths=source_seq_lengths,
                decoder_sequence_lengths=target_seq_lengths,
                training=False,
            )
            # Compute loss
            loss = self.criterion(
                y_true=target_seqs[:, 1:],  # skip <BOS> tag for targets
                y_pred=predictions[:, :-1, :],
            )  # skip <EOS> tag for predictions
            losses.update_state(loss)

        logging.info(f"{Fore.YELLOW}Average Validation Loss {losses.result():.4f}")
        losses.reset_states()
