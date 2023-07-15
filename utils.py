# third-party imports
import numpy as np
import tensorflow as tf  # type: ignore


def fast_positional_encoding(d_model: int, max_length: int = 100) -> tf.Tensor:
    """
    Computes positional encodings for the tokens.

    :param d_model: size of the vectors throughout the transformer
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a Tensor of shape (1, max_length, d_model)
    """

    position = np.arange(max_length)[:, np.newaxis]
    division_term = np.exp(np.arange(0., d_model, 2) * -(np.log(10000.0) / d_model))

    positional_encoding = np.zeros((max_length, d_model))
    positional_encoding[:, 0::2] = np.sin(position * division_term)
    positional_encoding[:, 1::2] = np.cos(position * division_term)

    return tf.constant(positional_encoding[np.newaxis, ...], dtype=tf.float32)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
