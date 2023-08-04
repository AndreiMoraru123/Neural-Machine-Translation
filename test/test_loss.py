# third-party imports
import pytest
import tensorflow as tf

# module imports
from loss import LabelSmoothedCrossEntropy


@pytest.fixture(
    name="loss_input_data", params=[(10, 25, 10000), "zero_loss", "high_loss"]
)
def input_data_for_loss(request):

    if isinstance(request.param, tuple):
        batch_size, seq_len, vocab_size = request.param

        # Decoded sequences
        inputs = tf.random.uniform(
            (batch_size, seq_len, vocab_size),
            minval=0,
            maxval=vocab_size,
            dtype=tf.float32,
        )

        # target sequences
        targets = tf.random.uniform(
            (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
        )

        return inputs, targets

    elif request.param == "zero_loss":
        # Targets and inputs are identical
        targets = tf.constant([[1, 2, 0], [3, 0, 0]], dtype=tf.int32)
        inputs = tf.one_hot(targets, depth=4, dtype=tf.float32)

        return inputs, targets

    elif request.param == "high_loss":
        # Targets and inputs are different
        targets = tf.constant([[1, 2, 0], [3, 0, 0]], dtype=tf.int32)
        inputs = tf.one_hot(3 - targets, depth=4, dtype=tf.float32)

        return inputs, targets


@pytest.fixture(name="loss_fn")
def label_smoothed_cross_entropy_loss():
    loss = LabelSmoothedCrossEntropy()
    return loss


def test_label_smoothed_cross_entropy_initialization(loss_fn):
    """Test the loss initialization."""
    assert isinstance(
        loss_fn, LabelSmoothedCrossEntropy
    ), "loss is not a LabelSmoothedCrossEntropy instance"


def test_label_smoothed_cross_entropy_loss(loss_input_data, loss_fn):
    """Tests the forward pass of the label smoother CE loss"""
    inputs, targets = loss_input_data

    # Compute loss
    loss = loss_fn(y_true=targets, y_pred=inputs)

    # Assert the loss is scalar
    assert tf.rank(loss) == 0, f"Expected scalar loss, but got {tf.rank(loss)}"

    # Get depth for one-hot encoding
    depth = tf.cast(inputs.shape[-1], dtype=tf.int32)

    # If inputs and targets are identical, loss should be close to zero
    if tf.reduce_all(tf.equal(inputs, tf.one_hot(targets, depth=depth))):
        assert loss < 1.0, f"Expected near-zero loss, but got {loss}"

    # If inputs and targets are completely different, loss should be high
    if tf.reduce_all(tf.equal(inputs, tf.one_hot(3 - targets, depth=depth))):
        assert loss > 1.0, f"Expected high loss, but got {loss}"
