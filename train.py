# third-party imports
import tensorflow as tf  # type: ignore

# module imports
from trainer import Trainer
from model import Transformer
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy
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
positional_encoding = fast_positional_encoding(d_model=d_model, max_length=160)

# Training params
tokens_in_batch = 500  # batch size in target language tokens
batches_per_step = 2500 // tokens_in_batch  # perform a training step (update parameters), once every so many batches
print_frequency = 20  # print status once every so many steps
save_every = 10000  # save every this many number of steps
n_epochs = 3  # number of training epochs
warmup_steps = 8000  # number of warmup steps where learning rate is increased linearly;
betas = (0.9, 0.98)  # beta coefficients in the Adam optimizer
epsilon = 1e-9  # epsilon term in the Adam optimizer
label_smoothing = 0.1  # label smoothing coefficient in the Cross Entropy loss

# Initialize the sequence loaders for training and validation
train_loader = SequenceLoader(data_folder="data",
                              source_suffix="en",
                              target_suffix="de",
                              split="train",
                              tokens_in_batch=tokens_in_batch)

val_loader = SequenceLoader(data_folder="data",
                            source_suffix="en",
                            target_suffix="de",
                            split="val",
                            tokens_in_batch=tokens_in_batch)

# Initialize the Transformer model
model = Transformer(n_heads=n_heads, d_model=d_model, d_queries=d_queries, d_values=d_values,
                    d_inner=d_inner, n_layers=n_layers, dropout=dropout,
                    vocab_size=train_loader.bpe_model.vocab_size(),
                    positional_encoding=positional_encoding)

# Initialize the Adam optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                             decay_steps=10000,
                                                             decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                     beta_1=betas[0],
                                     beta_2=betas[1],
                                     epsilon=epsilon)

# Initialize the Smooth Label Cross-Entropy loss
criterion = LabelSmoothedCrossEntropy(eps=label_smoothing)

# Initialize the trainer
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  log_dir="logs")

if __name__ == "__main__""":
    # Start training
    trainer.train(start_epoch=0,
                  epochs=n_epochs,
                  d_model=d_model,
                  warmup_steps=warmup_steps,
                  batches_per_step=batches_per_step,
                  print_frequency=print_frequency,
                  save_every=save_every)
