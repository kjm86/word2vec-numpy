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

    def _calculate_frequencies(self, one_hot_encodings: List[np.ndarray]) -> np.ndarray:
        """
        Calculates the frequencies of each token in the data.
        :param one_hot_encodings: List of one-hot encodings of tokens in each sequence
        :return: Frequencies for every token in the vocabulary
        """
        # count up how many times each token appeared in the data
        counts_per_seq: np.ndarray = np.array([np.sum(one_hot, axis=0) for one_hot in one_hot_encodings])
        total_counts: np.ndarray = np.sum(counts_per_seq, axis=0)

        return total_counts / np.sum(total_counts)

    def _calculate_neg_sampling_distr(self, freq: np.ndarray) -> np.ndarray:
        """
        Calculates the distribution used for negative sampling.
        :param freq: Frequencies of words in the data
        :return: Probability distribution for negative sampling
        """
        # calculate the distribution, specified in the paper as the unigram distribution raised to the 3/4 power
        neg_sampling_distr: np.ndarray = np.pow(freq, 0.75)
        neg_sampling_distr /= np.sum(neg_sampling_distr)

        return neg_sampling_distr

    def _calculate_discard_probabilities(self, freq: np.ndarray) -> np.ndarray:
        """
        Calculates the probabilities used for discarding more common tokens.
        :param freq: Frequencies of words in the data
        :return: Probability values for discarding tokens
        """
        # calculate the distribution, as specified in the paper
        t = 0.005
        discard_probabilities: np.ndarray = 1 - np.sqrt(t / np.maximum(freq, t))

        return discard_probabilities

    def train(self, train_data: List[List[str]], iter_num: int) -> None:
        """
        Main training loop. Performs the specified number of training iterations.
        :param train_data: A list of sentences, divided into lists of string tokens
        :param iter_num: The number of training iterations
        """
        one_hot_encodings = [self.one_hot_encode(seq) for seq in train_data]

        freq = self._calculate_frequencies(one_hot_encodings)
