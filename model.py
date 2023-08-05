# third-party imports
import einops
import tensorflow as tf
from tensorflow.keras import layers, Model


class MultiHeadAttention(layers.Layer):
    """MHA transformer layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        dropout: float,
        in_decoder: bool = False,
        **kwargs
    ):
        """

        :param d_model: size of a sequence of queries (and keys & values for convenience)
        :param n_heads: number of heads
        :param d_queries: size of the query vectors (and key vectors)
        :param d_values: size of the value vectors (and output vectors)
        :param dropout: probability of dropout
        :param in_decoder: whether we are decoding (masking attention) or not
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries
        self.in_decoder = in_decoder

        # Attention distribution logic
        assert (
            self.d_queries % self.n_heads == 0
        ), "d_queries must be divisible by n_heads"
        assert (
            self.d_values % self.n_heads == 0
        ), "d_values must be divisible by n_heads"
        assert self.d_keys % self.n_heads == 0, "d_keys must be divisible by n_heads"

        # Projection Layers
        self.cast_queries = layers.Dense(n_heads * d_queries)
        self.cast_keys_values = layers.Dense(n_heads * (d_queries + d_values))
        self.cast_output = layers.Dense(d_model)

        self.layer_norm = layers.LayerNormalization()

        self.dropout = layers.Dropout(dropout)

    def call(
        self,
        query_sequences: tf.Tensor,
        key_value_sequences: tf.Tensor,
        key_value_sequence_lengths: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        """
        Forward pass for all the heads.

        :param training: training mode (apply dropout) or inference mode (not apply dropout)
        :param query_sequences: input query sequences, a Tensor of shape (N, query_sequence_pad_length, d_model)
        :param key_value_sequences: the sequences to be queried against, (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, meant to ignore pads, (N)
        :return: attention-weighted output sequences for the query sequences, (N, query_sequence_pad_length, d_model)
        """
        query_sequence_pad_length = tf.shape(query_sequences)[1]
        key_value_sequence_pad_length = tf.shape(key_value_sequences)[1]

        def tensors_are_equal(tensor1: tf.Tensor, tensor2: tf.Tensor) -> bool:
            """
            Tensorflow has no straightforward way of determining whether two tensors have both the same dimensionality
            and elements, so this helper function does exactly that.

            :param tensor1: Tensor checking for
            :param tensor2: Tensor checking against
            :return: True if two tensors have the same size and elements, False otherwise (like torch.equal)
            """
            shape_check = tf.reduce_all(tf.shape(tensor1) == tf.shape(tensor2))
            if shape_check:
                # Check for eager execution and only then perform an element-wise check
                if tf.executing_eagerly():
                    elements_check = tf.reduce_all(tf.math.equal(tensor1, tensor2))
                    return elements_check
                else:
                    # In graph mode, we assume that having the same shape means they are equal
                    # This might not always be the case, of course, but TensorFlow can't help itself but build the graph
                    return True
            return False

        self_attention = tensors_are_equal(key_value_sequences, query_sequences)

        query_sequences = self.layer_norm(
            query_sequences
        )  # (N, query_sequence_pad_length, d_model)

        # (N, key_value_sequence_pad_length, d_model)
        key_value_sequences = tf.cond(
            self_attention,  # is this self attention or is it cross attention?
            true_fn=lambda: self.layer_norm(
                key_value_sequences
            ),  # need layer normalization
            false_fn=lambda: key_value_sequences,
        )  # they have already been normalized

        input_to_add = query_sequences  # tf Tensors are immutable, so no worries, assigning creates a copy by itself

        queries = self.cast_queries(
            query_sequences
        )  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys_values = self.cast_keys_values(key_value_sequences)
        keys, values = tf.split(
            keys_values,
            [self.n_heads * self.d_keys, self.n_heads * self.d_values],
            axis=-1,
        )

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        # And then, for convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)

        queries = einops.rearrange(
            queries,
            "b q (h d) -> b q h d",  # (N, query_sequence_pad_length, n_heads, d_queries)
            h=self.n_heads,
            d=self.d_queries,
        )
        keys = einops.rearrange(
            keys,
            "b kv (h d) -> b kv h d",  # (N, key_value_sequence_pad_length, n_heads, d_keys)
            h=self.n_heads,
            d=self.d_keys,
        )
        values = einops.rearrange(
            values,
            "b kv (h d) -> b kv h d",  # (N, key_value_sequence_pad_length, n_heads, d_values)
            h=self.n_heads,
            d=self.d_values,
        )

        # We want to parallelize the attentions, so we extend the heads as batches, so the 4D tensors become 3D.

        queries = einops.rearrange(
            queries,
            "b q h d -> (b h) q d",  # (N * n_heads, query_sequence_pad_length, d_queries)
        )
        keys = einops.rearrange(
            keys,
            "b kv h d -> (b h) kv d",  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        )
        values = einops.rearrange(
            values,
            "b kv h d -> (b h) kv d",  # (N * n_heads, key_value_sequence_pad_length, d_values)
        )

        attention_weights = tf.linalg.matmul(
            queries, keys, transpose_b=True
        )  # dot product
        # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        attention_weights = (
            1.0 / tf.math.sqrt(tf.cast(self.d_keys, dtype=tf.float32))
        ) * attention_weights  # scale

        # Use broadcasting for comparison to mask paddings
        range_tensor = tf.range(
            key_value_sequence_pad_length, dtype=tf.int32
        )  # (key_value_sequence_pad_length)

        # Repeat key_value_sequence_lengths to match the number of heads
        lengths_tensor = tf.repeat(
            key_value_sequence_lengths, self.n_heads
        )  # (N * n_heads)

        # Use broadcasting for comparison -> (N * n_heads, 1, key_value_sequence_pad_length)
        not_pad_in_keys = range_tensor[None, None, :] < lengths_tensor[:, None, None]

        # Repeat along the query_sequence_pad_length dimension to match the shape of attention_weights, i.e.
        # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = tf.repeat(not_pad_in_keys, query_sequence_pad_length, axis=1)

        attention_weights = tf.where(not_pad_in_keys, attention_weights, -float("inf"))

        def mask_future():
            """Masks future unseen tokens for decoding."""
            not_future_mask = tf.cast(
                tf.linalg.band_part(tf.ones_like(attention_weights), -1, 0), tf.bool
            )
            return tf.where(not_future_mask, attention_weights, -float("inf"))

        # Decide whether attention weights stay the same (encoding) or get masked (decoding)
        attention_weights = tf.cond(
            tf.logical_and(self.in_decoder, self_attention),
            true_fn=mask_future,
            false_fn=lambda: attention_weights,
        )

        attention_weights = tf.nn.softmax(
            attention_weights
        )  # softmax along the key dimension
        attention_weights = self.dropout(attention_weights, training=training)

        sequences = tf.linalg.matmul(
            attention_weights, values
        )  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch dimension and number of heads to restore original axes
        sequences = einops.rearrange(
            sequences,
            "(b h) q d -> b q h d",
            h=self.n_heads,  # (N, query_sequence_pad_length, n_heads, d_values)
        )

        # Concatenate the n_heads subspaces  (N, query_sequence_pad_length, n_heads * d_values)
        sequences = einops.rearrange(sequences, "b q h d -> b q (h d)")
        # Transform the concatenated subspace sequences into a single output of size d_model
        sequences = self.cast_output(
            sequences
        )  # (N, query_sequence_pad_length, d_model)
        # Dropout and residual connection
        sequences = self.dropout(sequences, training=training) + input_to_add

        return sequences


class FeedForward(layers.Layer):
    """The Feed Forward Network transformer layer."""

    def __init__(self, d_model: int, d_inner: int, dropout: float, **kwargs):
        """
        Initializes the Feed Forward Layer.

        :param d_model: input and output sizes for this sublayer.
        :param d_inner: in-between linear transforms dimension.
        :param dropout: dropout probability.
        """
        super(FeedForward, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = layers.LayerNormalization()
        self.fc1 = layers.Dense(d_inner)
        self.fc2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, sequences: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Forward pass of the feed forward layer.

        :param training: training mode (apply dropout) or inference mode (do not apply dropout)
        :param sequences: input sequences a Tensor of shape (N, pad_length, d_model)
        :return: output sequences, a Tensor of shape (N, pad_length, d_model)
        """
        input_to_add = sequences  # (N, pad_length, d_model)
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Force the model to learn into a different dimensionality
        sequences = self.dropout(
            tf.nn.relu(self.fc1(sequences)), training=training
        )  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)
        sequences = (
            self.dropout(sequences, training=training) + input_to_add
        )  # (N, pad_length, d_model)

        return sequences


class Encoder(layers.Layer):
    """Encoder Transformer for source language."""

    def __init__(
        self,
        vocab_size: int,
        positional_encoding: tf.Tensor,
        d_model: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        d_inner: int,
        n_layers: int,
        dropout: float,
        **kwargs
    ):
        """
        Initializes the Encoder.

        :param vocab_size: the size of the shared vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model in the Encoder
        :param n_heads: number of heads in the multi-head attention layer
        :param d_queries: size of the query vectors (and key vectors) in the multi-head attention layyer
        :param d_values: size of the value vectors in the multi-head attention
        :param d_inner: in-between linear transforms dimension in the feed forward layer
        :param n_layers: number of [multi head attention + feed forward] layers in the Encoder
        :param dropout: dropout probability
        """
        super(Encoder, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.encoder_layers = [self._make_layer() for _ in range(n_layers)]
        self.dropout_layer = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()

    def _make_layer(self):
        """Creates a single encoder layer by combining MHA + FFN sub layers."""

        multi_head_attention = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            in_decoder=False,
        )
        feed_forward_network = FeedForward(
            d_model=self.d_model, d_inner=self.d_inner, dropout=self.dropout
        )

        return [multi_head_attention, feed_forward_network]

    def call(
        self,
        encoder_sequences: tf.Tensor,
        encoder_sequence_lengths: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        """
        Forward pass of the Encoder.

        :param encoder_sequences: the source language sequences, a Tensor of shape (N, pad_length)
        :param encoder_sequence_lengths: true lengths of these sequences, a Tensor of shape (N)
        :param training: training mode (apply dropout) or inference mode (do not apply dropout)
        :return: encoded source language sequences, a Tensor of shape (N, pad_length, d_model)
        """

        pad_length = tf.shape(encoder_sequences)[
            1
        ]  # for this batch only, varies across batches
        encoder_sequences = self.embedding(encoder_sequences) * tf.math.sqrt(
            tf.cast(self.d_model, tf.float32)
        )
        encoder_sequences += self.positional_encoding[
            :, :pad_length
        ]  # (N, pad_length, d_model)

        encoder_sequences = self.dropout_layer(
            encoder_sequences, training=training
        )  # (N, pad_length, d_model)

        for encoder_layer in self.encoder_layers:
            # Multi Head Attention layer
            encoder_sequences = encoder_layer[0](
                query_sequences=encoder_sequences,
                key_value_sequences=encoder_sequences,
                key_value_sequence_lengths=encoder_sequence_lengths,
                training=training,
            )
            # Feed Forward layer
            encoder_sequences = encoder_layer[1](
                sequences=encoder_sequences, training=training
            )

        encoder_sequences = self.layer_norm(
            encoder_sequences
        )  # (N, pad_length, d_model)

        return encoder_sequences


class Decoder(layers.Layer):
    """Decoder Transformer for target language."""

    def __init__(
        self,
        vocab_size: int,
        positional_encoding: tf.Tensor,
        d_model: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        d_inner: int,
        n_layers: int,
        dropout: float,
        **kwargs
    ):
        """
        Initializes the Decoder.

        :param vocab_size: the size of the shared vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model in the Encoder
        :param n_heads: number of heads in the multi-head attention layer
        :param d_queries: size of the query vectors (and key vectors) in the multi-head attention layyer
        :param d_values: size of the value vectors in the multi-head attention
        :param d_inner: in-between linear transforms dimension in the feed forward layer
        :param n_layers: number of [multi head attention + feed forward] layers in the Encoder
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.decoder_layers = [self._make_layer() for _ in range(n_layers)]
        self.dropout_layer = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
        self.fc = layers.Dense(vocab_size)

    def _make_layer(self):
        """Creates a single encoder layer by combining MHA + FFN sub layers."""

        multi_head_attention_self = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            in_decoder=True,
        )
        multi_head_attention_cross = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            in_decoder=True,
        )
        feed_forward_network = FeedForward(
            d_model=self.d_model, d_inner=self.d_inner, dropout=self.dropout
        )

        return [
            multi_head_attention_self,
            multi_head_attention_cross,
            feed_forward_network,
        ]

    def call(
        self,
        decoder_sequences: tf.Tensor,
        decoder_sequence_lengths: tf.Tensor,
        encoder_sequences: tf.Tensor,
        encoder_sequence_lengths: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        """
        Forward pass of the Decoder.

        :param decoder_sequences: the source language sequences, a Tensor of shape (N, pad_length)
        :param decoder_sequence_lengths: true lengths of these sequences, a Tensor of shape (N)
        :param encoder_sequences: encoded source language sequences, a Tensor of shape (N, encoder_pad_length, d_model)
        :param encoder_sequence_lengths: true lengths of these sequences, a Tensor of shape (N)
        :param training: training mode (apply dropout) or inference mode (do not apply dropout)
        :return: decoded target language sequence, a Tensor of shape (N, pad_length, vocab_size)
        """

        pad_length = tf.shape(decoder_sequences)[
            1
        ]  # for this batch only, varies across batches
        decoder_sequences = self.embedding(decoder_sequences) * tf.math.sqrt(
            tf.cast(self.d_model, tf.float32)
        )
        decoder_sequences += self.positional_encoding[
            :, :pad_length, :
        ]  # (N, pad_length, d_model)

        decoder_sequences = self.dropout_layer(
            decoder_sequences, training=training
        )  # (N, pad_length, d_model)

        for decoder_layer in self.decoder_layers:
            # Multi Head Self Attention layer
            decoder_sequences = decoder_layer[0](
                query_sequences=decoder_sequences,
                key_value_sequences=decoder_sequences,
                key_value_sequence_lengths=decoder_sequence_lengths,
                training=training,
            )
            # Multi Head Cross Attention layer
            decoder_sequences = decoder_layer[1](
                query_sequences=decoder_sequences,
                key_value_sequences=encoder_sequences,
                key_value_sequence_lengths=encoder_sequence_lengths,
                training=training,
            )
            # Feed Forward layer
            decoder_sequences = decoder_layer[2](
                sequences=decoder_sequences, training=training
            )

        decoder_sequences = self.layer_norm(
            decoder_sequences
        )  # (N, pad_length, d_model)

        # Compute across vocabulary dimension
        decoder_sequences = self.fc(decoder_sequences)  # (N, pad_length, d_model)

        return decoder_sequences


class Transformer(Model):
    """The Transformer network."""

    def __init__(
        self,
        vocab_size: int,
        positional_encoding: tf.Tensor,
        d_model: int = 512,
        n_heads: int = 8,
        d_queries: int = 64,
        d_values: int = 64,
        d_inner: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initializes the transformer network.

        :param vocab_size: size of the shared vocabulary, i.e. total number of word tokens
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of the vectors throughout the transformer model
        :param n_heads: number of heads in the multi-head attention layer
        :param d_queries: size of the query vectors (and key vectors) in the multi-head attention layer
        :param d_values: size of the value vectors in the multi-head attention layer
        :param d_inner: in between linear transforms size in the feed forward layer
        :param n_layers: number of layers in the Encoder and Decoder
        :param dropout: dropout probability
        """
        super(Transformer, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers

        self.encoder = Encoder(
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            d_inner=d_inner,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            d_inner=d_inner,
            n_layers=n_layers,
            dropout=dropout,
        )

    def call(
        self,
        encoder_sequences: tf.Tensor,
        decoder_sequences: tf.Tensor,
        encoder_sequence_lengths: tf.Tensor,
        decoder_sequence_lengths: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass of the Transformer network.

        :param encoder_sequences: source language sequences, a tensor of size (N, encoder_sequence_pad_length)
        :param decoder_sequences: target language sequences, a tensor of size (N, decoder_sequence_pad_length)
        :param encoder_sequence_lengths: true lengths of source language sequences, a tensor of size (N)
        :param decoder_sequence_lengths: true lengths of target language sequences, a tensor of size (N)
        :param training: training mode (apply dropout) or inference mode (not apply dropout)
        :return: decoded target language sequences, a tensor of size (N, decoder_sequence_pad_length, vocab_size)
        """

        encoder_sequences = self.encoder(
            encoder_sequences,  # (N, encoder_sequence_pod_length, d_model)
            encoder_sequence_lengths,
            training=training,
        )

        decoder_sequences = self.decoder(
            decoder_sequences,  # (N, decoder_sequence_pad_length, vocab_size)
            decoder_sequence_lengths,
            encoder_sequences,
            encoder_sequence_lengths,
            training=training,
        )

        return decoder_sequences
