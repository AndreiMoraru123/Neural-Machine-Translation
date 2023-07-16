# standard imports
import os
import codecs
from random import shuffle
from itertools import groupby

# third-party imports
import tensorflow as tf  # type: ignore
import youtokentome  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


class SequenceLoader(object):
    """
    An iterator for loading batches of data into the transformer model.

    For training:

        Each batch contains tokens_in_batch target language tokens (approximately),
        target language sequences of the same length to minimize padding and therefore memory usage,
        source language sequences of very similar lengths to minimize padding and therefore memory usage.
        Batches are also shuffled.

    For validation and testing:

        Each batch contains a single source-target pair, in the same order as in the files from which they were read.
    """

    def __init__(self, data_folder, source_suffix, target_suffix, split, tokens_in_batch):
        """
        Sequence constructor.

        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language
        :param target_suffix: the filename suffix for the target language
        :param split: train, val or test
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.n_batches = None
        self.all_batches = None
        self.current_batch = None

        self.tokens_in_batch = tokens_in_batch
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix

        assert split.lower() in {"train", "test", "val"}, "no such split"

        self.split = split.lower()
        self.for_training = self.split == "train"
        self.bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

        with codecs.open(os.path.join(data_folder, ".".join([split, source_suffix])), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")
        with codecs.open(os.path.join(data_folder, ".".join([split, target_suffix])), "r", encoding="utf-8") as f:
            target_data = f.read().split("\n")
        assert len(source_data) == len(target_data), "different number of source and target sequences"

        source_lengths = [len(s) for s in self.bpe_model.encode(source_data, bos=False, eos=False)]
        target_lengths = [len(t) for t in self.bpe_model.encode(target_data, bos=True, eos=True)]

        self.data = list(zip(source_data, target_data, source_lengths, target_lengths))

        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        self.create_batches()

    def create_batches(self):
        """
        Prepares batches for one epoch.
        """
        if self.for_training:
            chunks = [list(g) for _, g in groupby(self.data, key=lambda x: x[3])]

            self.all_batches = list()  # create batches with the same target sequence length
            for chunk in chunks:
                chunk.sort(key=lambda x: x[2])  # sort so that a batch would also have similar source sequence lengths
                # div the expected batch size (tokens) by sequence length in this chunk to get # of sequences per batch
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            shuffle(self.all_batches)  # shuffle batches
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

        else:
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

    def __iter__(self):
        """ Required by iterator."""
        return self

    def __next__(self):
        """
        Next in iterator.

        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, a tensor of size (N)
        """
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
        except IndexError:
            raise StopIteration

        source_data = self.bpe_model.encode(source_data, output_type=youtokentome.OutputType.ID, bos=False, eos=False)
        target_data = self.bpe_model.encode(target_data, output_type=youtokentome.OutputType.ID, bos=True, eos=True)

        source_data = pad_sequences(sequences=source_data, padding='post', value=self.bpe_model.subword_to_id('<PAD>'))
        target_data = pad_sequences(sequences=target_data, padding='post', value=self.bpe_model.subword_to_id('<PAD>'))

        source_data = tf.convert_to_tensor(source_data, dtype=tf.int32)
        target_data = tf.convert_to_tensor(target_data, dtype=tf.int32)

        source_lengths = tf.convert_to_tensor(source_lengths, dtype=tf.int32)
        target_lengths = tf.convert_to_tensor(target_lengths, dtype=tf.int32)

        return source_data, target_data, source_lengths, target_lengths
