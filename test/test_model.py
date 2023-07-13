# third-party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore

# module imports
from model import MultiHeadAttention, FeedForward


@pytest.fixture(name='mha_layer_config')
def multi_head_attention_layer_configuration():
    """Configuration parameters for initializing MultiHeadAttention."""
    config = {
        "d_model": 128,
        "n_heads": 8,
        "d_queries": 16,
        "d_values": 16,
        "dropout": 0.1,
        "in_decoder": True,
    }
    return config


@pytest.fixture(name="ffn_layer_config")
def feed_forward_layer_configuration():
    """Configuration parameters for initializing FeedForward."""
    config = {
        "d_model": 128,
        "d_inner": 512,
        "dropout": 0.1,
    }
    return config


@pytest.fixture(name="mha_layer")
def multi_head_attention_layer(mha_layer_config):
    """Multi Head Attention layer."""
    layer = MultiHeadAttention(**mha_layer_config)
    return layer


@pytest.fixture(name="ffn_layer")
def feed_forward_layer(ffn_layer_config):
    """Feed Forward layer."""
    layer = FeedForward(**ffn_layer_config)
    return layer


@pytest.fixture(name="mha_input_data", params=[(10, 25, 128)])
def input_data_for_multi_head_attention_layer(request):
    batch_size, seq_len, d_model = request.param

    query_sequences = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    key_value_sequences = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    key_value_sequence_lengths = tf.ones((batch_size,), dtype=tf.int32) * seq_len

    return query_sequences, key_value_sequences, key_value_sequence_lengths


@pytest.fixture(name="ffn_input_data", params=[(10, 25, 128)])
def input_data_for_feed_forward_layer(request):
    batch_size, seq_len, d_model = request.param
    sequences = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    return sequences


def test_multi_head_attention_layer_initialization(mha_layer):
    """Tests initialization of a MultiHeadAttention Layer."""
    assert isinstance(mha_layer, MultiHeadAttention), "Layer is not a MultiHeadAttention instance."


def test_feed_forward_layer_initialization(ffn_layer):
    """Tests initialization of a FeedForward Layer."""
    assert isinstance(ffn_layer, FeedForward), "Layer is not a FeedForward instance."


def test_multi_head_attention_layer_forward_pass(mha_layer, mha_input_data):
    """Tests forward pass of a MultiHeadAttention Layer."""
    query_sequences, key_value_sequences, key_value_sequence_lengths = mha_input_data

    # Assert skip connection compatibility
    assert mha_layer.d_model == query_sequences.shape[2], "Will not be able to add residual skip connection otherwise"
    output = mha_layer(query_sequences, key_value_sequences, key_value_sequence_lengths)

    # Assert the output shape is as expected
    expected_shape = (query_sequences.shape[0], query_sequences.shape[1], mha_layer.d_model)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # Assert the output is different from the input (i.e., the layer is doing something)
    assert not tf.reduce_all(tf.equal(query_sequences, output)), "The output is the same as the input."


def test_feed_forward_layer_forward_pass(ffn_layer, ffn_input_data):
    """Tests forward pass of a Feed Forward Layer."""
    sequences = ffn_input_data

    # Assert skip connection compatibility
    assert ffn_layer.d_model == sequences.shape[2], "Will not be able to add residual skip connection otherwise"
    output = ffn_layer(sequences)

    # Assert the output shape is as expected
    expected_shape = (sequences.shape[0], sequences.shape[1], ffn_layer.d_model)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # Assert the output is different from the input (i.e., the layer is doing something)
    assert not tf.reduce_all(tf.equal(sequences, output)), "The output is the same as the input."
