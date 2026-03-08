import numpy as np
from typing import Set, Dict, List


class Word2VecModel:
    def __init__(self, vocabulary: Set[str], emb_size: int):
        """
        :param vocabulary: A set of unique words (tokens) in the vocabulary
        :param emb_size: The size of the embedding
        """
        self.vocab_map: Dict[str, int] = dict(zip(vocabulary, range(len(vocabulary))))
        self.vocab_size: int = len(vocabulary)
        self.emb_size: int = emb_size

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialises the "input" and "output" representation matrices, as described in the paper.
        """
        self.input_word_weights: np.ndarray = np.random.uniform(low=-0.5 / self.emb_size, high=0.5 / self.emb_size, size=(self.vocab_size, self.emb_size))
        self.output_word_weights: np.ndarray = np.random.uniform(low=-0.5 / self.emb_size, high=0.5 / self.emb_size, size=(self.vocab_size, self.emb_size))

    def one_hot_encode(self, seq: List[str]) -> np.ndarray:
        """
        Encodes the sequence of string tokens as one-hot matrix.
        :param seq: The sequence of string tokens
        :return: A matrix containing the one-hot encoding of the provided sequence
        """
        # make sure all words are in the vocabulary
        if any(token not in self.vocab_map for token in seq):
            raise KeyError(f"Unknown token")

        int_tokens: np.ndarray = np.array(list(map(lambda token: self.vocab_map[token], seq)))
        n = int_tokens.size

        one_hot: np.ndarray = np.zeros((n, self.vocab_size))
        one_hot[np.arange(n), int_tokens] = 1
        return one_hot

    def get_embeddings(self, seq: List[str]) -> np.ndarray:
        """
        Calculates the embeddings for the provided sequence of string tokens.
        :param seq: The sequence of string tokens
        :return: A matrix of size len(seq) by emb_size, containing token embeddings
        """

        one_hot: np.ndarray = self.one_hot_encode(seq)
        emb: np.ndarray = one_hot @ self.input_word_weights
        return emb
