
```python
# PyTorch  

N = 2  # batch size
n_heads = 2
query_sequence_pad_length = 4
key_value_sequence_pad_length = 3
key_value_sequence_lengths = torch.LongTensor([3, 2])

not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand(N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(n_heads).unsqueeze(1).unsqueeze(2).expand(N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

print(not_pad_in_keys)
print(not_pad_in_keys.shape)
```

```
tensor([[[ True, True, True], [ True, True, True], [ True, True, True], [ True, True, True]], [[ True, True, True], [ True, True, True], [ True, True, True], [ True, True, True]], [[ True, True, False], [ True, True, False], [ True, True, False], [ True, True, False]], [[ True, True, False], [ True, True, False], [ True, True, False], [ True, True, False]]]) torch.Size([4, 4, 3])
```

```python
# TensorFlow

N = 2  # batch size
n_heads = 2
query_sequence_pad_length = 4
key_value_sequence_pad_length = 3
key_value_sequence_lengths = tf.constant([3, 2], dtype=tf.int32)

attention_weights = tf.random.normal([N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length])  

@tf.function
def compute_mask():
    # Use broadcasting for comparison
    range_tensor = tf.range(key_value_sequence_pad_length, dtype=tf.int32)  # (key_value_sequence_pad_length)
    
    # Repeat key_value_sequence_lengths to match the number of heads
    lengths_tensor = tf.repeat(key_value_sequence_lengths, n_heads)  # (N * n_heads)

    # Use broadcasting for comparison
    not_pad_in_keys = range_tensor[None, None, :] < lengths_tensor[:, None, None]  # (N * n_heads, 1, key_value_sequence_pad_length)

    # Repeat along the query_sequence_pad_length dimension to match the shape of attention_weights
    not_pad_in_keys = tf.repeat(not_pad_in_keys, query_sequence_pad_length, axis=1)  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

    return not_pad_in_keys

print(compute_mask())
```

```
tf.Tensor( [[[ True True True] [ True True True] [ True True True] [ True True True]] [[ True True True] [ True True True] [ True True True] [ True True True]] [[ True True False] [ True True False] [ True True False] [ True True False]] [[ True True False] [ True True False] [ True True False] [ True True False]]], shape=(4, 4, 3), dtype=bool)
```