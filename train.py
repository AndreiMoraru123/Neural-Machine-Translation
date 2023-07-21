# third-party imports
import tensorflow as tf  # type: ignore

# module imports
from trainer import Trainer
from model import Transformer
from dataloader import SequenceLoader
from loss import LabelSmoothedCrossEntropy
from utils import WarmupLearningRateSchedule, fast_positional_encoding

# Paths
data_folder = "data"
log_dir = "logs"

# Model params
d_model = 512  # size of vectors throughout the transformer model
n_heads = 8  # number of heads in the multi-head attention
d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
d_values = 64  # size of value vectors in the multi-head attention
d_inner = 2048  # an intermediate size in the position-wise FC
n_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
positional_encoding = fast_positional_encoding(d_model=d_model, max_length=100)

# Training params
tokens_in_batch = 900  # batch size in target language tokens
batches_per_step = 10000 // tokens_in_batch  # perform a training step (update parameters) once every so many batches
print_frequency = 50  # print status once every so many steps
save_every = 20000  # save every this many number of steps
n_epochs = 3  # number of training epochs
warmup_steps = 16000  # number of warmup steps where learning rate is increased linearly
betas = (0.9, 0.98)  # beta coefficients in the Adam optimizer
epsilon = 1e-9  # epsilon term in the Adam optimizer
label_smoothing = 0.1  # label smoothing coefficient in the Cross Entropy loss
path_to_checkpoint = ""

train_loader = SequenceLoader(data_folder=data_folder,
                              source_suffix="en",
                              target_suffix="de",
                              split="train",
                              tokens_in_batch=tokens_in_batch)

val_loader = SequenceLoader(data_folder=data_folder,
                            source_suffix="en",
                            target_suffix="de",
                            split="val",
                            tokens_in_batch=tokens_in_batch)

model = Transformer(n_heads=n_heads, d_model=d_model, d_queries=d_queries, d_values=d_values,
                    d_inner=d_inner, n_layers=n_layers, dropout=dropout,
                    vocab_size=train_loader.bpe_model.vocab_size(),
                    positional_encoding=positional_encoding)

lr_schedule = WarmupLearningRateSchedule(d_model=d_model, warmup_steps=4000)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                     beta_1=betas[0],
                                     beta_2=betas[1],
                                     epsilon=epsilon)

criterion = LabelSmoothedCrossEntropy(eps=label_smoothing)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  log_dir=log_dir)

if __name__ == "__main__""":
    # Runs and manages the whole training pipeline
    trainer.train(start_epoch=0, epochs=n_epochs, batches_per_step=batches_per_step,
                  print_frequency=print_frequency, save_every=save_every,
                  path_to_checkpoint=path_to_checkpoint)
