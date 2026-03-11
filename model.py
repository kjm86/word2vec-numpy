import os.path
from os import PathLike

import numpy as np
from typing import Set, Dict, List

import tqdm


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
        self.input_token_weights: np.ndarray = np.random.uniform(low=-0.5 / self.emb_size, high=0.5 / self.emb_size, size=(self.vocab_size, self.emb_size))
        self.output_token_weights: np.ndarray = np.random.uniform(low=-0.5 / self.emb_size, high=0.5 / self.emb_size, size=(self.vocab_size, self.emb_size))

    def get_int_tokens(self, seq: List[str]) -> np.ndarray:
        """
        Converts the string tokens to integers.
        :param seq: The sequence of string tokens
        :return: An array containing the integer values corresponding to the input tokens
        """
        # make sure all words are in the vocabulary
        if any(token not in self.vocab_map for token in seq):
            raise KeyError(f"Unknown token")

        return np.array(list(map(lambda token: self.vocab_map[token], seq)))

    def get_embeddings(self, seq: List[str]) -> np.ndarray:
        """
        Calculates the embeddings for the provided sequence of string tokens.
        :param seq: The sequence of string tokens
        :return: A matrix of size len(seq) by emb_size, containing token embeddings
        """
        int_tokens: np.ndarray = self.get_int_tokens(seq)
        emb: np.ndarray = self.input_token_weights[int_tokens]
        return emb

    def _calculate_frequencies(self, int_tokens: List[np.ndarray]) -> np.ndarray:
        """
        Calculates the frequencies of each token in the data.
        :param int_tokens: List of int tokens
        :return: Frequencies for every token in the vocabulary
        """
        # count up how many times each token appeared in the data
        total_counts: np.ndarray = np.zeros((self.vocab_size,))
        for seq in int_tokens:
            total_counts += np.bincount(seq, minlength=self.vocab_size)

        return total_counts / np.sum(total_counts)

    def _calculate_neg_sampling_distr(self, freq: np.ndarray) -> np.ndarray:
        """
        Calculates the distribution used for negative sampling.
        :param freq: Frequencies of tokens in the data
        :return: Probability distribution for negative sampling
        """
        # calculate the distribution, specified in the paper as the unigram distribution raised to the 3/4 power
        neg_sampling_distr: np.ndarray = np.pow(freq, 0.75)
        neg_sampling_distr /= np.sum(neg_sampling_distr)

        return neg_sampling_distr

    def _calculate_discard_probabilities(self, freq: np.ndarray, t: float) -> np.ndarray:
        """
        Calculates the probabilities used for discarding more common tokens.
        :param freq: Frequencies of tokens in the data
        :param t: The frequency threshold below which tokens will never get discarded
        :return: Probability values for discarding tokens
        """
        # calculate the distribution, as specified in the paper
        discard_probabilities: np.ndarray = 1 - np.sqrt(t / np.maximum(freq, t))

        return discard_probabilities

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train(self, train_data: List[List[str]], iter_num: int, negative_sample_num: int = 15, context_size: int = 5, learning_rate: float = 0.05, subsampling_threshold: float = 0.005) -> None:
        """
        Main training loop. Performs the specified number of training iterations.
        :param train_data: A list of sentences, divided into lists of string tokens
        :param iter_num: The number of training iterations
        :param negative_sample_num: The number of negative samples for each output/input pair (default: 15)
        :param context_size: The size of the context window (default: 5)
        :param learning_rate: Learning rate for the weight adjustment (default: 0.05)
        :param subsampling_threshold: The frequency threshold below which tokens will never get discarded (default: 0.005)
        """
        all_int_tokens: List[np.ndarray] = [self.get_int_tokens(seq) for seq in train_data]

        freq: np.ndarray = self._calculate_frequencies(all_int_tokens)

        discard_probabilities: np.ndarray = self._calculate_discard_probabilities(freq, subsampling_threshold)
        neg_sampling_distr: np.ndarray = self._calculate_neg_sampling_distr(freq)

        for it in range(iter_num):
            print("Iteration", it + 1)

            rolling_avg_reward: float = 0
            with tqdm.tqdm(all_int_tokens) as t:
                for int_tokens in t:
                    token_discard_prob = discard_probabilities[int_tokens]
                    # has True if token was discarded, and False otherwise
                    discarded: np.ndarray = np.random.random((len(int_tokens),)) < token_discard_prob

                    subsampled_int_tokens: np.ndarray = int_tokens[~discarded]
                    n: int = subsampled_int_tokens.shape[0]

                    # if at most one token, then we skip
                    if n <= 1:
                        continue

                    input_representations: np.ndarray = self.input_token_weights[subsampled_int_tokens]
                    output_representations: np.ndarray = self.output_token_weights[subsampled_int_tokens]

                    input_token_weight_errors: np.ndarray = np.zeros_like(self.input_token_weights)
                    output_token_weight_errors: np.ndarray = np.zeros_like(self.output_token_weights)

                    reward: float = 0

                    for i, input_token in zip(range(n), subsampled_int_tokens):
                        # select the context window
                        positive_sample_tokens: np.ndarray = np.concat((subsampled_int_tokens[i - context_size:i],
                                                                        subsampled_int_tokens[i + 1:i + context_size + 1]))
                        positive_sample_representations: np.ndarray = np.concat((output_representations[i - context_size:i],
                                                                                 output_representations[i + 1:i + context_size + 1]))
                        # actual context size, takes into account the truncation of window at the beginning and end of sentence
                        actual_context_size = positive_sample_representations.shape[0]

                        # generate negative_sample_num negative samples for each output token
                        negative_sample_tokens: np.ndarray = np.random.choice(self.vocab_size, size=negative_sample_num * actual_context_size, p=neg_sampling_distr)
                        negative_sample_representations: np.ndarray = self.output_token_weights[negative_sample_tokens]

                        positive_sample_output: np.ndarray = self._sigmoid(input_representations[i] @ positive_sample_representations.T)
                        negative_sample_output: np.ndarray = self._sigmoid(-input_representations[i] @ negative_sample_representations.T)

                        # maximise the log of sigmoid values
                        input_token_weight_errors[input_token] += (1 - positive_sample_output) @ positive_sample_representations
                        input_token_weight_errors[input_token] += -(1 - negative_sample_output) @ negative_sample_representations

                        # doing this instead of the standard += to count duplicate indices
                        np.add.at(output_token_weight_errors, positive_sample_tokens, np.outer(1 - positive_sample_output, input_representations[i]))
                        np.add.at(output_token_weight_errors, negative_sample_tokens, np.outer(-(1 - negative_sample_output), input_representations[i]))

                        reward += np.sum(positive_sample_output)
                        reward += np.sum(negative_sample_output)

                    reward /= n

                    rolling_avg_reward = 0.95 * rolling_avg_reward + 0.05 * reward

                    t.set_postfix(rolling_average_reward=rolling_avg_reward)

                    self.input_token_weights += learning_rate * input_token_weight_errors / n
                    self.output_token_weights += learning_rate * output_token_weight_errors / n
