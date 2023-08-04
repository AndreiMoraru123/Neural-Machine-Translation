# third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import schedules


class WarmupLearningRateSchedule(schedules.LearningRateSchedule):
    """Custom learning rate scheduling class that mimics the one in Attention Is All You Need."""

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        """
        Initializes the learning rate scheduler.

        :param d_model: size of vectors throughout the transformer
        :param warmup_steps: number of warmup steps where learning rate is increased linearly
        """
        super(WarmupLearningRateSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.total_steps = 0

    def increment_step(self):
        """Keras resets the schedule at every epoch, and this is undesired as the lr would start warming up again."""
        self.total_steps += 1  # so we keep track of the total number of steps taken instead of the relative steps

    def __call__(self, step: int) -> tf.Tensor:
        """
        Changes the learning rate according to the paper formula for the given step.

        lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
        :param step: step for the scheduler, required by Keras, even though we don't actually use it
        :return: the updated learning rate
        """
        self.increment_step()
        step = tf.cast(self.total_steps, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def fast_positional_encoding(d_model: int, max_length: int = 100) -> tf.Tensor:
    """
    Computes positional encodings for the tokens.

    :param d_model: size of the vectors throughout the transformer
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a Tensor of shape (1, max_length, d_model)
    """

    position = np.arange(max_length)[:, np.newaxis]
    division_term = np.exp(np.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model))

    positional_encoding = np.zeros((max_length, d_model))
    positional_encoding[:, 0::2] = np.sin(position * division_term)
    positional_encoding[:, 1::2] = np.cos(position * division_term)

    return tf.constant(positional_encoding[np.newaxis, ...], dtype=tf.float32)
