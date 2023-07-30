# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras import losses  # type:ignore


class LabelSmoothedCrossEntropy(losses.Loss):
    """
    Cross Entropy Loss with label_smoothing as a form of regularization.
    """

    def __init__(self, eps: float = 0.1, **kwargs):
        """
        Initializes the loss.

        :param eps: smoothing coefficient
        """
        super(LabelSmoothedCrossEntropy, self).__init__(**kwargs)
        self.eps = eps

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the loss.

        :param y_true: gold target sequences, a tensor of shape (N, pad_length)
        :param y_pred: decoded target language sequences, a Tensor of shape (N, pad_length , vocab_size)
        :return: mean label-smoother cross-entropy scalar loss
        """

        # Create mask for padded positions
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, y_pred.dtype)

        # Flatten inputs and targets
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true_flat = tf.reshape(y_true, [-1])

        # Smoothed one-hot vectors for the gold sequences
        depth = tf.cast(y_pred.shape[-1], tf.int32)
        target_vector = tf.one_hot(y_true_flat, depth)
        target_vector = target_vector * (1.0 - self.eps) + self.eps / y_pred.shape[-1]

        # Compute smoothed cross-entropy loss
        loss = -1 * target_vector * tf.nn.log_softmax(y_pred_flat, axis=-1)

        # Apply mask
        mask_flat = tf.reshape(mask, [-1])
        mask_flat_expanded = tf.expand_dims(mask_flat, axis=-1)
        loss *= mask_flat_expanded  # mask gets broadcast in TensorFlow

        # Compute mean loss
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask_flat)

        return loss
