# standard imports
import math
import time

# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore

# module imports
from model import Transformer
from utils import AverageMeter
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy


class Trainer:
    """ Utility class to train a model."""

    def __init__(
        self,
        model: Transformer,
        optimizer: Optimizer,
        criterion: LabelSmoothedCrossEntropy,
        train_loader: SequenceLoader,
        val_loader: SequenceLoader
    ):
        """
        Initializes the trainer with tensorflow objects.

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

    @staticmethod
    def schedule_learning_rate(step: int, d_model: int, warmup_steps: int) -> float:
        """
        The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

        :param step: training step number
        :param d_model: size of vectors throughout the transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper
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
    ):
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
            self.train_one_epoch(epoch, d_model, warmup_steps, batches_per_step, print_frequency)
            self.val_loader.create_batches()
            self.validate_one_epoch()
            self.save_checkpoint(epoch, self.model, self.optimizer)

    def save_checkpoint(self, epoch, model, optimizer, prefix=''):
        """Saves a tensorflow checkpoint for the specified epoch."""
        self.model.save(prefix + 'transformer_checkpoint', save_format='tf')

    @tf.function
    def train_one_epoch(self, epoch: int, d_model: int, warmup_steps: int, batches_per_step: int, print_frequency: int):
        """
        Trains the model for one epoch.

        :param epoch: the current training epoch
        :param d_model: size of the vectors throughout the Transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        :param batches_per_step: perform a training step (update parameters), once every so many batches
        :param print_frequency: print status once every so many steps
        """
        step = 1
        data_time = AverageMeter()
        step_time = AverageMeter()
        losses = AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (source_seqs, target_seqs, source_seq_lengths, target_seq_lengths) in enumerate(self.train_loader):
            with tf.GradientTape() as tape:
                predictions = self.model(encoder_sequences=source_seqs,
                                         decoder_sequences=target_seqs,
                                         encoder_sequence_lengths=source_seq_lengths,
                                         decoder_sequence_lengths=target_seq_lengths,
                                         training=True)
                loss = self.criterion(y_true=target_seqs[:, 1:], # skip <BOS> tag for targets
                                      y_pred=predictions[:, :-1, :])  # skip <EOS> tag for predictions

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if (i + 1) % batches_per_step == 0:
                self.optimizer.learning_rate = self.schedule_learning_rate(step, d_model, warmup_steps)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                step += 1

            data_time.update(time.time() - start_data_time)
            losses.update(loss.numpy(), (target_seq_lengths - 1).numpy().sum())

            if step % print_frequency == 0:
                step_time.update(time.time() - start_step_time)
                print(f'Epoch {epoch + 1} '
                      f'Batch {i + 1}/{self.train_loader.n_batches}-----'
                      f'Step {step}-----'
                      f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                      f'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            start_data_time = time.time()
            start_step_time = time.time()

    @tf.function
    def validate_one_epoch(self):
        """Validate the model."""
        losses = AverageMeter()

        for i, (source_seqs, target_seqs, source_seq_lengths, target_seq_lengths) in enumerate(self.val_loader):
            # Forward prop
            predictions = self.model(encoder_sequences=source_seqs,
                                     decoder_sequences=target_seqs,
                                     encoder_sequence_lengths=source_seq_lengths,
                                     decoder_sequence_lengths=target_seq_lengths,
                                     training=False)
            # Compute loss
            loss = self.criterion(targets=target_seqs[:, 1:], inputs=predictions)  # skip <BOS> tag

            # Keep track of losses
            losses.update(loss.numpy(), (target_seq_lengths - 1).numpy().sum())

        # Print status
        print(f'Validation Loss {losses.val:.4f} ({losses.avg:.4f})')
