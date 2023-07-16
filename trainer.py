# standard imports
import os
import math
import time
import shutil
import logging

# third-party imports
from colorama import Fore, init  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.metrics import Mean  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore

# module imports
from model import Transformer, MultiHeadAttention
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy

# logging house-keeping
init(autoreset=True)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """ Utility class to train a language model."""

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
        """
        logging.info(f'{Fore.GREEN}Initializing Trainer')

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

        if os.path.exists(log_dir):
            logging.info(f'{Fore.YELLOW}Flushing Logs')
            shutil.rmtree(log_dir)

        logging.info(f'{Fore.CYAN}Creating Summary Writer')
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def schedule_learning_rate(step: int, d_model: int, warmup_steps: int) -> float:
        """
        The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repo.

        :param step: training step number
        :param d_model: size of vectors throughout the transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        :return: updated learning rate
        """
        lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
        return lr

    def train(
        self,
        start_epoch: int,
        epochs: int,
        d_model: int,
        warmup_steps: int,
        batches_per_step: int,
        print_frequency: int,
        save_every: int,
    ) -> None:
        """
        Trains the model for the number of specified epochs.

        :param save_every: save every this many number of steps
        :param start_epoch: starting epoch
        :param epochs: total number of training epochs
        :param d_model: size of the vectors throughout the Transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """
        for epoch in range(start_epoch, epochs):
            logging.info(f'{Fore.GREEN}Started training at epoch {epoch}')
            self.train_loader.create_batches()
            logging.info(f'{Fore.YELLOW}Created training batches')
            self.train_one_epoch(epoch, d_model, warmup_steps, batches_per_step, print_frequency, epochs, save_every)
            logging.info(f'{Fore.CYAN}Finished training epoch {epoch}')
            self.val_loader.create_batches()
            logging.info(f'{Fore.YELLOW}Created validation batches')
            self.validate_one_epoch()
            logging.info(f'{Fore.CYAN}Finished validating epoch {epoch}')

    def save_checkpoint(self, idx: int, prefix: str = 'checkpoints'):
        """
        Saves the model weights.

        :param idx: index for saving
        :param prefix: path prefix
        """
        logging.info(f'{Fore.GREEN}Saving model at step {idx}')
        self.model.save_weights(f"{prefix}/transformer_checkpoint_{idx}/checkpoint", save_format='tf')
        logging.info(f'{Fore.CYAN}Successfully saved weights')

    def load_checkpoint(self, checkpoint_dir: str):
        """
        Loads the model from the latest checkpoint.

        :param checkpoint_dir: path to the directory containing the checkpoints
        """
        logging.info(f'{Fore.GREEN}Loading model from {checkpoint_dir}')

        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        logging.info(f'{Fore.CYAN}Latest checkpoint at {checkpoint_path}')

        # Dummy data to build the model
        dummy_encoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_decoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_encoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)
        dummy_decoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)

        # Calling the model on the dummy data to build it
        self.model(encoder_sequences=dummy_encoder_sequences,
                   decoder_sequences=dummy_decoder_sequences,
                   encoder_sequence_lengths=dummy_encoder_sequence_lengths,
                   decoder_sequence_lengths=dummy_decoder_sequence_lengths,
                   training=False)

        self.model.load_weights(checkpoint_path)
        logging.info(f'{Fore.YELLOW}Successfully loaded weights')

    def train_one_epoch(
        self,
        epoch: int,
        d_model: int,
        warmup_steps: int,
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
        :param d_model: size of the vectors throughout the Transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """
        step = 1
        data_time = Mean()
        step_time = Mean()
        losses = Mean()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (source_seqs, target_seqs, source_seq_lengths, target_seq_lengths) in enumerate(self.train_loader):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model(encoder_sequences=source_seqs,
                                         decoder_sequences=target_seqs,
                                         encoder_sequence_lengths=source_seq_lengths,
                                         decoder_sequence_lengths=target_seq_lengths,
                                         training=True)
                # Compute loss
                loss = self.criterion(y_true=target_seqs[:, 1:],  # skip <BOS> tag for targets
                                      y_pred=predictions[:, :-1, :])  # skip <EOS> tag for predictions

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if (i + 1) % batches_per_step == 0:
                self.optimizer.learning_rate = self.schedule_learning_rate(step, d_model, warmup_steps)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                step += 1

            data_time.update_state(time.time() - start_data_time)
            losses.update_state(loss)

            if step % print_frequency == 0:
                step_time.update_state(time.time() - start_step_time)
                logging.info(f'{Fore.GREEN}Epoch {epoch + 1}/{epochs} '
                             f'{Fore.BLUE}Batch {i + 1}/{self.train_loader.n_batches}-----'
                             f'{Fore.WHITE}Step {step}-----'
                             f'{Fore.MAGENTA}Data Time {data_time.result():.3f}-----'
                             f'{Fore.CYAN}Step Time {step_time.result():.3f}-----'
                             f'{Fore.GREEN}Loss {loss:.4f}-----'
                             f'{Fore.YELLOW}Average Loss {losses.result():.4f}')
                with self.summary_writer.as_default():  # log metrics for TensorBoard
                    tf.summary.scalar('Loss', losses.result(), step=step)
                    tf.summary.scalar('Learning Rate', self.optimizer.learning_rate, step=step)

            if (i + 1) % save_every == 0:
                self.save_checkpoint(i + 1)

            start_data_time = time.time()
            start_step_time = time.time()
            losses.reset_states()
            data_time.reset_states()
            step_time.reset_states()

    def validate_one_epoch(self) -> None:
        """Validates the model over the validation loader."""
        losses = Mean()

        for i, (source_seqs, target_seqs, source_seq_lengths, target_seq_lengths) in enumerate(self.val_loader):
            # Forward pass
            predictions = self.model(encoder_sequences=source_seqs,
                                     decoder_sequences=target_seqs,
                                     encoder_sequence_lengths=source_seq_lengths,
                                     decoder_sequence_lengths=target_seq_lengths,
                                     training=False)
            # Compute loss
            loss = self.criterion(y_true=target_seqs[:, 1:],  # skip <BOS> tag for targets
                                  y_pred=predictions[:, :-1, :])  # skip <EOS> tag for predictions
            losses.update_state(loss)

        logging.info(f'{Fore.GREEN}Average Validation Loss {losses.result():.4f}')
        losses.reset_states()
