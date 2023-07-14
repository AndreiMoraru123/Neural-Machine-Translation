# third-party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore

# module imports
from model import MultiHeadAttention, FeedForward, Encoder, Decoder, Transformer


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


@pytest.fixture(name='encoder_layer_config')
def encoder_layer_configuration():
    """Configuration parameters for initializing Encoder."""
    config = {
        "vocab_size": 10000,
        "positional_encoding": tf.random.uniform((1, 100, 128)),
        "d_model": 128,
        "n_heads": 8,
        "d_queries": 16,
        "d_values": 16,
        "d_inner": 512,
        "n_layers": 6,
        "dropout": 0.1,
    }
    return config


@pytest.fixture(name='decoder_layer_config')
def decoder_layer_configuration():
    """Configuration parameters for initializing Decoder."""
    config = {
        "vocab_size": 10000,
        "positional_encoding": tf.random.uniform((1, 100, 128)),
        "d_model": 128,
        "n_heads": 8,
        "d_queries": 16,
        "d_values": 16,
        "d_inner": 512,
        "n_layers": 6,
        "dropout": 0.1,
    }
    return config


@pytest.fixture(name='transformer_config')
def transformer_configuration():
    """Configuration parameters for initializing Transformer."""
    config = {
        "vocab_size": 10000,
        "positional_encoding": tf.random.uniform((1, 100, 128)),
        "d_model": 128,
        "n_heads": 8,
        "d_queries": 16,
        "d_values": 16,
        "d_inner": 512,
        "n_layers": 6,
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


@pytest.fixture(name="encoder_layer")
def encoder_layer(encoder_layer_config):
    """Encoder layer."""
    layer = Encoder(**encoder_layer_config)
    return layer


@pytest.fixture(name="decoder_layer")
def decoder_layer(decoder_layer_config):
    """Decoder layer."""
    layer = Decoder(**decoder_layer_config)
    return layer


@pytest.fixture(name='transformer')
def transformer_layer(transformer_config):
    """Initialize Transformer with the given configuration."""
    transformer = Transformer(**transformer_config)
    return transformer


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


@pytest.fixture(name="encoder_input_data", params=[(10, 25)])
def input_data_for_encoder_layer(request):
    batch_size, seq_len = request.param

    encoder_sequences = tf.random.uniform((batch_size, seq_len), minval=0, maxval=10000, dtype=tf.int32)
    encoder_sequence_lengths = tf.ones((batch_size,), dtype=tf.int32) * seq_len

    return encoder_sequences, encoder_sequence_lengths


@pytest.fixture(name="decoder_input_data", params=[(10, 25)])
def input_data_for_decoder_layer(request, decoder_layer_config):
    batch_size, seq_len = request.param

    # Token sequences
    decoder_sequences = tf.random.uniform((batch_size, seq_len), minval=0, maxval=10000, dtype=tf.int32)
    decoder_sequence_lengths = tf.ones((batch_size,), dtype=tf.int32) * seq_len

    # For simplicity, let's assume encoder_sequences and their lengths are same as decoder ones
    encoder_token_sequences = decoder_sequences
    encoder_sequence_lengths = decoder_sequence_lengths

    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(decoder_layer_config['vocab_size'], decoder_layer_config['d_model'])

    # Apply embedding for encoder sequences, as the decoder sequences already are getting embedded in the decoder
    encoder_sequences = embedding_layer(encoder_token_sequences)

    return decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths


@pytest.fixture(name='transformer_input_data', params=[(10, 25)])
def input_data_for_transformer(request, transformer_config):
    """Generate input data for Transformer."""
    batch_size, seq_len = request.param

    # Token sequences
    encoder_sequences = tf.random.uniform((batch_size, seq_len), minval=0, maxval=10000, dtype=tf.int32)
    encoder_sequence_lengths = tf.ones((batch_size,), dtype=tf.int32) * seq_len
    decoder_sequences = tf.random.uniform((batch_size, seq_len), minval=0, maxval=10000, dtype=tf.int32)
    decoder_sequence_lengths = tf.ones((batch_size,), dtype=tf.int32) * seq_len

    return encoder_sequences, encoder_sequence_lengths, decoder_sequences, decoder_sequence_lengths


def test_multi_head_attention_layer_initialization(mha_layer):
    """Tests initialization of a MultiHeadAttention Layer."""
    assert isinstance(mha_layer, MultiHeadAttention), "Layer is not a MultiHeadAttention instance."


def test_feed_forward_layer_initialization(ffn_layer):
    """Tests initialization of a FeedForward Layer."""
    assert isinstance(ffn_layer, FeedForward), "Layer is not a FeedForward instance."


def test_encoder_layer_initialization(encoder_layer):
    """Tests initialization of an Encoder Layer."""
    assert isinstance(encoder_layer, Encoder), "Layer is not an Encoder instance."


def test_decoder_layer_initialization(decoder_layer):
    """Tests initialization of a Decoder Layer."""
    assert isinstance(decoder_layer, Decoder), "Layer is not a Decoder instance."


def test_transformer_initialization(transformer):
    """Tests initialization of a Transformer."""
    assert isinstance(transformer, Transformer), "Layer is not a Transformer instance."


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


def test_encoder_layer_forward_pass(encoder_layer, encoder_input_data):
    """Tests forward pass of an Encoder Layer."""
    encoder_sequences, encoder_sequence_lengths = encoder_input_data

    output = encoder_layer(encoder_sequences, encoder_sequence_lengths)

    # Assert the output shape is as expected
    expected_shape = (encoder_sequences.shape[0], encoder_sequences.shape[1], encoder_layer.d_model)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"


def test_decoder_layer_forward_pass(decoder_layer, decoder_input_data):
    """Tests forward pass of a Decoder Layer."""
    decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths = decoder_input_data

    output = decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths)

    # Assert the output shape is as expected
    expected_shape = (decoder_sequences.shape[0], decoder_sequences.shape[1], decoder_layer.vocab_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"


def test_transformer_forward_pass(transformer, transformer_input_data):
    """Tests forward pass of a Transformer."""
    encoder_sequences, encoder_sequence_lengths, decoder_sequences, decoder_sequence_lengths = transformer_input_data

    output = transformer(encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths)

    # Assert the output shape is as expected
    expected_shape = (decoder_sequences.shape[0], decoder_sequences.shape[1], transformer.vocab_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

