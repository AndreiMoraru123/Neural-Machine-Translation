
# third-party imports
import youtokentome  # type: ignore
import tensorflow as tf  # type: ignore
from youtokentome import BPE

# module imports
from model import Transformer


class Evaluator:
    """Utility class to evaluate a language model for the task of translation."""

    def __init__(self, model: Transformer, bpe_model: BPE):
        """
        Initializes the Evaluator

        :param model: the Transformer model
        :param bpe_model: the Byte-Pair Encoding model
        """

        self.model = model
        self.bpe_model = bpe_model

    def load_checkpoint(self, checkpoint_dir: str):
        """
        Loads the model from the latest checkpoint.

        :param checkpoint_dir: path to the directory containing the checkpoints
        """

        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        # Dummy data to build the model
        dummy_encoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_decoder_sequences = tf.zeros((1, 1), dtype=tf.int32)
        dummy_encoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)
        dummy_decoder_sequence_lengths = tf.zeros((1,), dtype=tf.int32)

        # Calling the model on the dummy data to build it
        self.model(encoder_sequences=dummy_encoder_sequences,
                   decoder_sequences=dummy_decoder_sequences,
                   encoder_sequence_lengths=dummy_encoder_sequence_lengths,
                   decoder_sequence_lengths=dummy_decoder_sequence_lengths,
                   training=False)

        self.model.load_weights(checkpoint_path)
