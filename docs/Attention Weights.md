
```python
# PyTorch

attention_weights = torch.Tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
not_pad_in_keys = torch.BoolTensor([[True, True, False], [True, True, True], [True, False, True]])

# Apply padding masking
attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf'))
print(attention_weights)

# Apply future masking
not_future_mask = torch.ones_like(attention_weights).tril().bool()
attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf'))
print(attention_weights)
```

```
tensor([[0.1000, 0.2000, -inf], [0.3000, 0.4000, 0.3000], [0.2000, -inf, 0.5000]]) tensor([[0.1000, -inf, -inf], [0.3000, 0.4000, -inf], [0.2000, -inf, 0.5000]])
```

```python
# TensorFlow

attention_weights = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]], dtype=tf.float32)
not_pad_in_keys = tf.constant([[True, True, False], [True, True, True], [True, False, True]])

# Apply padding masking
attention_weights = tf.where(not_pad_in_keys, attention_weights, -float('inf'))
print(attention_weights)

# Apply future masking
def mask_future():
    not_future_mask = tf.cast(tf.linalg.band_part(tf.ones_like(attention_weights), -1, 0), tf.bool)
    return tf.where(not_future_mask, attention_weights, -float('inf'))

attention_weights = tf.cond(tf.constant(True), true_fn=mask_future, false_fn=lambda: attention_weights)
print(attention_weights)
```

```
tf.Tensor( [[ 0.1 0.2 -inf] [ 0.3 0.4 0.3] [ 0.2 -inf 0.5]], shape=(3, 3), dtype=float32) tf.Tensor( [[ 0.1 -inf -inf] [ 0.3 0.4 -inf] [ 0.2 -inf 0.5]], shape=(3, 3), dtype=float32)
```
