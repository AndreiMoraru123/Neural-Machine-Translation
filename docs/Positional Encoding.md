
```python
# PyTorch

def get_positional_encoding(d_model, max_length=100):

    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum seq length up to which positional encodings are computed
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """

    pos_enc = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                pos_enc[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                pos_enc[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    pos_enc = pos_enc.unsqueeze(0)  # (1, max_length, d_model)

    return pos_enc
```

Notice we have two computations:

```python
math.pow(10000, j / d_model)
```

when `j` is even, and:

```python
math.pow(10000, (j - 1) / d_model)
```

when `j` is odd.

Notice that `(j-1)` occurs only when `j` is odd, so we can account for the difference by starting at `0`:

```python
np.exp(np.arange(0., d_model, 2) * -np.log(10000.0 / d_model))
```


The property $a^b = e^{ln(a^b)}$  can be used here, so:

```python
math.pow(10000, j / d_model)
```

becomes:

```python
math.exp(math.log(math.pow(10000, j / d_model)))
```

and we know that $log(a^b) = b * log(a)$, so we can further simplify it to:

```python
math.exp((j / d_model) * math.log(10000))
```

and lastly we compute this for all the even `j` values:

```python
np.exp(np.arange(0., d_model, 2) * -(np.log(10000.0) / d_model))
```

What about when `j` is odd?

The division term is always going to have an even value, so it stays the same for both:

```python
positional_encoding[:, 0::2] = np.sin(position * div_term)  # for even indices
positional_encoding[:, 1::2] = np.cos(position * div_term)  # for odd indices
```

So the final equivalent implementation becomes:

```python
import numpy as np  
import tensorflow as tf  
  
  
def get_positional_encoding(d_model: int, max_length: int = 100) -> tf.Tensor:

	"""  
	Computes positional encodings for the tokens.  
	  
	:param d_model: size of the vectors throughout the transformer
	:param max_length: maximum seq length up to which positional encodings are computed
	:return: positional encoding, a Tensor of shape (1, max_length, d_model)  
	"""
	  
	position = np.arange(max_length)[:, np.newaxis]  
	division_term = np.exp(np.arange(0., d_model, 2) * -(np.log(10000.0) / d_model))  
	  
	positional_encoding = np.zeros((max_length, d_model))  
	positional_encoding[:, 0::2] = np.sin(position * division_term)  
	positional_encoding[:, 1::2] = np.cos(position * division_term)

	return tf.constant(positional_encoding[np.newaxis, ...], dtype=tf.float32)
```
