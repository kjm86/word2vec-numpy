import numpy as np
from typing import Set, Dict, List


class Word2VecModel:
    def __init__(self, vocabulary: Set[str]):
        """
        :param vocabulary: A set of words (tokens) in the vocabulary
        """
        self.vocab_map: Dict[str, int] = dict(zip(vocabulary, range(len(vocabulary))))

    def one_hot_encode(self, seq: List[str]) -> np.ndarray:
        """
        Encodes the sequence of string tokens as one-hot matrix.
        :param seq:
        :return:
        """

        n = len(seq)
        V = len(self.vocab_map)
        int_tokens: np.ndarray = np.array(list(map(lambda token: self.vocab_map[token], seq)))
        one_hot: np.ndarray = np.zeros((n, V))
        one_hot[np.arange(n), int_tokens] = 1
        return one_hot
