# standard imports
import math
import logging
from typing import Union, Tuple, List, Dict

# third-party imports
import sacrebleu
import youtokentome  # type: ignore
import tensorflow as tf  # type: ignore
from tqdm import tqdm  # type: ignore
from colorama import Fore, init  # type: ignore

# module imports
from model import Transformer
from dataloader import SequenceLoader

# logging house-keeping
init(autoreset=True)
logging.basicConfig(level=logging.INFO)


class Evaluator:
    """Utility class to evaluate a language model for the task of translation."""

    def __init__(self, model: Transformer, test_loader: SequenceLoader, bpe_model_path: str):
        """
        Initializes the Evaluator.

        :param model: the Transformer model
        :param test_loader: the sequence loader in test configuration
        :param bpe_model_path: the path to the Byte-Pair Encoding model
        """

        self.model = model
        self.test_loader = test_loader
        self.bpe_model = youtokentome.BPE(model=bpe_model_path)

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

    def evaluate(self, length_norm_coefficient: float = 0.6, k: int = 5):
        """
        Evaluates the model using BLUE score.

        :param length_norm_coefficient: coefficient for normalizing decoded sequences' scores by their lengths
        :param k: beam size, when k = 1 the translation is equivalent to performing greedy decoding
        :return:
        """
        hypotheses = []
        references = []
        COLORS = [Fore.GREEN, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA]
        for _, (source_seqs, target_seqs, source_seq_lengths, target_seq_lengths) in enumerate(
            tqdm(self.test_loader, total=self.test_loader.n_batches)
        ):
            hypotheses.append(self.translate(source_sequence=source_seqs,
                                             length_norm_coefficient=length_norm_coefficient,
                                             k=k)[0])
            references.extend(self.test_loader.bpe_model.decode(target_seqs.numpy().tolist(), ignore_ids=[0, 2, 3]))

        for i, (print_text, sacrebleu_text) in enumerate(zip([
            "13a tokenization, cased",
            "13a tokenization, caseless",
            "International tokenization, cased",
            "International tokenization, caseless"
        ], [
            sacrebleu.corpus_bleu(hypotheses, [references]),
            sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True),
            sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'),
            sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True)
        ])):
            colored_log_message = f'{COLORS[i]}{print_text}'
            colored_sacrebleu = f'{COLORS[i]}{sacrebleu_text}'

            logging.info(colored_log_message)
            logging.info(colored_sacrebleu)

    def translate(
        self,
        source_sequence: Union[tf.Tensor, str],
        length_norm_coefficient: float = 0.6,
        k: int = 5
    ) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Translates a source language sequence into the target language, with beam search decoding.

        :param source_sequence: the source language sequence, either a string or a Tensor of byte pair encoded indices
        :param length_norm_coefficient: coefficient for normalizing decoded sequences' scores by their lengths
        :param k: beam size, when k = 1 the translation is equivalent to performing greedy decoding
        :return: the best hypothesis as well as all candidate hypotheses
        """

        n_completed_hypotheses = min(k, 10)  # minimum number of hypotheses to complete

        vocab_size = self.bpe_model.vocab_size()

        if isinstance(source_sequence, str):
            encoder_sequences = self.bpe_model.encode(source_sequence,
                                                      output_type=youtokentome.OutputType.ID,
                                                      bos=False, eos=False)
            encoder_sequences = tf.expand_dims(encoder_sequences, 0)  # (1, source_sequence_length)
        else:
            encoder_sequences = source_sequence

        encoder_sequence_lengths = tf.constant([encoder_sequences.shape[1]], dtype=tf.int32)  # (1)

        encoder_sequences = self.model.encoder(encoder_sequences=encoder_sequences,
                                               encoder_sequence_lengths=encoder_sequence_lengths,
                                               training=False)  # (1, source_sequence_length, d_model)

        hypotheses = tf.constant([[self.bpe_model.subword_to_id('<BOS>')]])  # (1, 1)
        hypotheses_lengths = tf.constant([hypotheses.shape[1]], dtype=tf.int32)  # (1)
        hypotheses_scores = tf.zeros(1)

        completed_hypotheses = []
        completed_hypotheses_scores = []

        step = 1

        while True:
            s = tf.shape(hypotheses)[0]
            decoder_sequences = self.model.decoder(decoder_sequences=hypotheses,  # (s, step, vocab_size)
                                                   decoder_sequence_lengths=hypotheses_lengths,
                                                   encoder_sequences=tf.repeat(encoder_sequences, s, axis=0),
                                                   encoder_sequence_lengths=tf.repeat(encoder_sequence_lengths, s))

            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = tf.nn.log_softmax(scores, axis=-1)  # (s, vocab_size)

            scores = tf.expand_dims(hypotheses_scores, 1) + scores  # (s, vocab_size)

            top_k_hypotheses_scores, unrolled_indices = tf.math.top_k(tf.reshape(scores, [-1]), k)

            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            top_k_hypotheses = tf.concat([tf.gather(hypotheses, prev_word_indices),
                                          tf.expand_dims(next_word_indices, 1)], axis=1)  # (k, step + 1)

            complete = tf.equal(next_word_indices, self.bpe_model.subword_to_id('<EOS>'))  # (k)

            completed_hypotheses.extend(tf.boolean_mask(top_k_hypotheses, complete).numpy().tolist())

            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((tf.boolean_mask(top_k_hypotheses_scores, complete) /
                                                norm).numpy().tolist())

            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            hypotheses = tf.boolean_mask(top_k_hypotheses, ~complete)  # (s, step + 1)
            hypotheses_scores = tf.boolean_mask(top_k_hypotheses_scores, ~complete)  # (s)
            hypotheses_lengths = tf.fill([tf.shape(hypotheses)[0]], tf.shape(hypotheses)[1])  # (s)

            if step > 100:
                break
            step += 1

        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.numpy().tolist()
            completed_hypotheses_scores = hypotheses_scores.numpy().tolist()

        all_hypotheses = []
        for i, h in enumerate(self.bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses
