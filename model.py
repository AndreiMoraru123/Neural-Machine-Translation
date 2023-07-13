# standard imports
import math

# third-party imports
import einops
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore


class MultiHeadAttention(layers.Layer):
    """MHA transformer layer."""

    def __init__(self, d_model: int, n_heads: int, d_queries: int, d_values: int,
                 dropout: float, in_decoder: bool = False, **kwargs):
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
        assert self.d_queries % self.n_heads == 0, 'd_queries must be divisible by n_heads'
        assert self.d_values % self.n_heads == 0, 'd_values must be divisible by n_heads'
        assert self.d_keys % self.n_heads == 0, 'd_keys must be divisible by n_heads'

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
        key_value_sequence_lengths: tf.Tensor
    ) -> tf.Tensor:
        """
        Forward pass for all the heads.

        :param query_sequences: input query sequences, a Tensor of shape (N, query_sequence_pad_length, d_model)
        :param key_value_sequences: the sequences to be queried against, (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, meant to ignore pads, (N)
        :return: attention-weighted output sequences for the query sequences, (N, query_sequence_pad_length, d_model)
        """
        query_sequence_pad_length = tf.shape(query_sequences)[1]
        key_value_sequence_pad_length = tf.shape(key_value_sequences)[1]

        self_attention = tf.reduce_all(tf.equal(key_value_sequences, query_sequences))

        query_sequences = self.layer_norm(query_sequences)  # (N, query_sequence_pad_length, d_model)

        # (N, key_value_sequence_pad_length, d_model)
        key_value_sequences = tf.cond(self_attention,  # is this self attention or is it cross attention?
                                      true_fn=lambda: self.layer_norm(key_value_sequences),  # need layer normalization
                                      false_fn=lambda: key_value_sequences)  # they have already been normalized

        input_to_add = query_sequences  # tf Tensors are immutable, so no worries, assigning creates a copy by itself

        queries = self.cast_queries(query_sequences)  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys_values = self.cast_keys_values(key_value_sequences)
        keys, values = tf.split(keys_values, [self.n_heads * self.d_keys, self.n_heads * self.d_values], axis=-1)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        # And then, for convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)

        queries = einops.rearrange(
            queries, 'b q (h d) -> b q h d',  # (N, query_sequence_pad_length, n_heads, d_queries)
            h=self.n_heads, d=self.d_queries
        )
        keys = einops.rearrange(
            keys, 'b kv (h d) -> b kv h d',  # (N, key_value_sequence_pad_length, n_heads, d_keys)
            h=self.n_heads, d=self.d_keys
        )
        values = einops.rearrange(
            values, 'b kv (h d) -> b kv h d',  # (N, key_value_sequence_pad_length, n_heads, d_values)
            h=self.n_heads, d=self.d_values
        )

        # We want to parallelize the attentions, so we extend the heads as batches, so the 4D tensors become 3D.

        queries = einops.rearrange(
            queries, 'b q h d -> (b h) q d'  # (N * n_heads, query_sequence_pad_length, d_queries)
        )
        keys = einops.rearrange(
            keys, 'b kv h d -> (b h) kv d'  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        )
        values = einops.rearrange(
            values, 'b kv h d -> (b h) kv d'  # (N * n_heads, key_value_sequence_pad_length, d_values)
        )

        attention_weights = tf.linalg.matmul(queries, keys, transpose_b=True)  # dot product
        # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        attention_weights = (1. / math.sqrt(self.d_keys)) * attention_weights  # scale

        # Use broadcasting for comparison to mask paddings
        range_tensor = tf.range(key_value_sequence_pad_length, dtype=tf.int32)  # (key_value_sequence_pad_length)

        # Repeat key_value_sequence_lengths to match the number of heads
        lengths_tensor = tf.repeat(key_value_sequence_lengths, self.n_heads)  # (N * n_heads)

        # Use broadcasting for comparison -> (N * n_heads, 1, key_value_sequence_pad_length)
        not_pad_in_keys = range_tensor[None, None, :] < lengths_tensor[:, None, None]

        # Repeat along the query_sequence_pad_length dimension to match the shape of attention_weights, i.e.
        # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = tf.repeat(not_pad_in_keys, query_sequence_pad_length, axis=1)

        attention_weights = tf.where(not_pad_in_keys, attention_weights, -float('inf'))

        def mask_future():
            """Masks future unseen tokens for decoding."""
            not_future_mask = tf.cast(tf.linalg.band_part(tf.ones_like(attention_weights), -1, 0), tf.bool)
            return tf.where(not_future_mask, attention_weights, -float('inf'))

        # Decide whether attention weights stay the same (encoding) or get masked (decoding)
        attention_weights = tf.cond(tf.logical_and(self.in_decoder, self_attention),
                                    true_fn=mask_future, false_fn=lambda: attention_weights)

        attention_weights = tf.nn.softmax(attention_weights)  # softmax along the key dimension
        attention_weights = self.dropout(attention_weights)

        sequences = tf.linalg.matmul(attention_weights, values)  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch dimension and number of heads to restore original axes
        sequences = einops.rearrange(
            sequences, '(b h) q d -> b q h d', h=self.n_heads,  # (N, query_sequence_pad_length, n_heads, d_values)
        )

        # Concatenate the n_heads subspaces  (N, query_sequence_pad_length, n_heads * d_values)
        sequences = einops.rearrange(sequences, 'b q h d -> b q (h d)')
        # Transform the concatenated subspace sequences into a single output of size d_model
        sequences = self.cast_output(sequences)  # (N, query_sequence_pad_length, d_model)
        # Dropout and residual connection
        sequences = self.dropout(sequences) + input_to_add

        return sequences


class FeedForward(layers.Layer):
    """The Feed Forward Network transformer layer."""

    def __init__(self, d_model: int, d_inner: int, dropout: float, **kwargs):
        """

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

    def call(self, sequences: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the feed forward layer.
        :param sequences: input sequences a Tensor of shape (N, pad_length, d_model)
        :return: output sequences, a Tensor of shape (N, pad_length, d_model)
        """

        input_to_add = sequences  # (N, pad_length, d_model)
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Force the model to learn into a different dimensionality
        sequences = self.dropout(tf.nn.relu(self.fc1(sequences)))  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        sequences = self.dropout(sequences) + input_to_add # (N, pad_length, d_model)

        return sequences