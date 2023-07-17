# standard imports
import os

# third-party imports
import tensorflow as tf  # type: ignore

# module imports
from model import Transformer
from evaluator import Evaluator
from dataloader import SequenceLoader
from utils import fast_positional_encoding

# Data Path
data_folder = "data"

# Model params
d_model = 512  # size of vectors throughout the transformer model
n_heads = 8  # number of heads in the multi-head attention
d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
d_values = 64  # size of value vectors in the multi-head attention
d_inner = 2048  # an intermediate size in the position-wise FC
n_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
tokens_in_batch = 600  # batch size in target language tokens
positional_encoding = fast_positional_encoding(d_model=d_model, max_length=160)
path_to_checkpoint = "checkpoints/transformer_checkpoint_30000"

val_loader = SequenceLoader(data_folder=data_folder,
                            source_suffix="en",
                            target_suffix="de",
                            split="val",
                            tokens_in_batch=tokens_in_batch)

model = Transformer(n_heads=n_heads, d_model=d_model, d_queries=d_queries, d_values=d_values,
                    d_inner=d_inner, n_layers=n_layers, dropout=dropout,
                    vocab_size=val_loader.bpe_model.vocab_size(),
                    positional_encoding=positional_encoding)


evaluator = Evaluator(model=model, bpe_model_path=os.path.join(data_folder, "bpe.model"))
evaluator.load_checkpoint(path_to_checkpoint)

if __name__ == "__main__":
    best_hypothesis, all_hypotheses = evaluator.translate("The Hotel.")
    print(best_hypothesis)
    print(all_hypotheses)

