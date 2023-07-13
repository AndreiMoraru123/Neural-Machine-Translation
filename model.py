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

        batch_size = tf.shape(query_sequences)[0]  # (N)
        query_sequence_pad_length = tf.shape(query_sequences)[1]
        key_value_sequence_pad_length = tf.shape(key_value_sequences)[1]

        self_attention = tf.reduce_all(tf.equal(key_value_sequences, query_sequences))

        query_sequences = self.layer_norm(query_sequences)  # (N, query_sequence_pad_length, d_model)

        # (N, key_value_sequence_pad_length, d_model)
        key_value_sequences = tf.cond(self_attention,  # is this self attention or is it cross attention?
                                      true_fn=lambda: self.layer_norm(key_value_sequences),  # need layer normalization
                                      false_fn=lambda: key_value_sequences)  # they have already been normalized

        input_to_add = query_sequences  # tf Tensors are immutable, so no worries

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

        # Mask the paddings in the keys
        range_tensor = tf.range(key_value_sequence_pad_length, dtype=tf.int32)  # (key_value_sequence_pad_length)
        # Expand range tensor
        range_tensor = tf.expand_dims(range_tensor, axis=0)  # (1, key_value_sequence_pad_length
        # Repeat range tensor to shape (N * query_sequence_pad_length * n_heads, key_value_sequence_pad_length)
        range_tensor = tf.tile(range_tensor, [tf.reduce_prod(attention_weights.shape[:-1]), 1])
        # Reshape into the final form (N * query_sequence_pad_length, key_value_sequence_pad_length)
        range_tensor = tf.reshape(range_tensor, attention_weights.shape)

        lengths_tensor = tf.repeat(tf.expand_dims(key_value_sequence_lengths, -1),  # (N, key_value_sequence_pad_length)
                                 key_value_sequence_pad_length, axis=-1)
        # Expand to (N, 1, query_sequence_pad_length * key_value_sequence_pad_length)
        lengths_tensor = tf.repeat(tf.expand_dims(lengths_tensor, 1), query_sequence_pad_length, axis=-1)
        # Expand to (N, n_heads, 1, query_sequence_pad_length * key_value_sequence_pad_length)
        lengths_tensor = tf.repeat(tf.expand_dims(lengths_tensor, 1), self.n_heads, axis=1)
        # Broadcast back to (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        lengths_tensor = tf.reshape(lengths_tensor, attention_weights.shape)

        not_pad_in_keys = range_tensor < lengths_tensor

        attention_weights = tf.where(not_pad_in_keys, attention_weights, -float('inf'))

        def mask_future():
            not_future_mask = tf.cast(tf.linalg.band_part(tf.ones_like(attention_weights), -1, 0), tf.bool)
            return tf.where(not_future_mask, attention_weights, -float('inf'))

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



