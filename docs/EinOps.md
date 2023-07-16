```python
# PyTorch

queries = torch.randn(2, 3, 4)

queries = queries.view(2, 3, 2, 2)

queries.shape  # (N, query_sequence_pad_length, n_heads, d_queries)
```
```
torch.Size([2, 3, 2, 2])
```

```python
queries = queries.permute(0, 2, 1, 3).contiguous()

queries = queries.view(-1, 3, 2)

queries.shape  # (N * n_heads, query_sequence_pad_length, d_queries)
```
```
torch.Size([4, 3, 2])
```

```python
# TensorFlow + EinOps

queries = tf.random.normal([2, 3, 4])

queries = einops.rearrange(queries, 'b q (h d) -> b q h d', h=2)

queries.get_shape()
```
```
TensorShape([2, 3, 2, 2])
```

```python
queries = einops.rearrange(queries, 'b q h d -> (b h) q d')

queries.get_shape() # (N * n_heads, query_sequence_pad_length, d_queries)
```
```
TensorShape([4, 3, 2])
```
