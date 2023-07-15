# standard imports
import math
import time
import logging

# third-party imports
from colorama import Fore, init  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.metrics import Mean  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore

# module imports
from model import Transformer
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy

# logging house-keeping
init(autoreset=True)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """ Utility class to train a model."""

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
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        print_frequency: int
    ) -> None:
        """
        Trains the model for the number of specified epochs.

        :param start_epoch: starting epoch
        :param epochs: total number of training epochs
        :param d_model: size of the vectors throughout the Transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """
        for epoch in range(start_epoch, epochs):
            self.train_loader.create_batches()
            self.train_one_epoch(epoch, d_model, warmup_steps, batches_per_step, print_frequency, epochs)
            self.val_loader.create_batches()
            self.validate_one_epoch()
            self.save_checkpoint()

    def save_checkpoint(self, prefix=''):
        """Saves a tensorflow checkpoint for the model."""
        self.model.save_weights(prefix + 'transformer_checkpoint')

    def train_one_epoch(
        self,
        epoch: int,
        d_model: int,
        warmup_steps: int,
        batches_per_step: int,
        print_frequency: int,
        epochs: int,
    ) -> None:
        """
        Trains the model for one epoch.

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
                             f'{Fore.GREEN}Loss {loss:.4f} '
                             f'{Fore.YELLOW}Average Loss {losses.result():.4f}')
                with self.summary_writer.as_default():  # log metrics for TensorBoard
                    tf.summary.scalar('Loss', losses.result(), step=step)
                    tf.summary.scalar('Learning Rate', self.optimizer.learning_rate, step=step)

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
