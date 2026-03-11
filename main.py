import random
from os import PathLike
from typing import List

from model import Word2VecModel


N = 50000 # number of samples
ITER_NUM = 4 # number of training iterations
EMB_SIZE = 150 # embedding size
LR = 0.1 # learning rate
DATA_PATH = "./wiki2.txt"
MODEL_PATH = "./saved_model"

def load_full_sentences(path: PathLike) -> List[str]:
    """
    Load sentence data from a file.
    :param path: The path to the file
    :return: List of unprocessed sentences
    """
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]

def get_word_sequences(sentences: List[str]) -> List[List[str]]:
    """
    Get the list of words (tokens) for every sentence. Removes anything that isn't alphanumeric characters and converts all characters to lowercase.
    :param sentences: List of unprocessed sentences
    :return: List containing lists of lowercase words
    """
    word_seqs = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = "".join(c for c in sentence if (c.isascii() and c.isalnum()) or c.isspace())
        words = sentence.split()
        if len(words) > 0:
            word_seqs.append(words)
    return word_seqs


def main():
    all_data = get_word_sequences(load_full_sentences(DATA_PATH))
    # discarding short sequences, because the dataset contains also section headers, which aren't real sentences
    all_data = [seq for seq in all_data if len(seq) >= 5]
    training_data = random.choices(all_data, k=N)

    # create vocabulary
    vocabulary = set()
    for line in training_data:
        vocabulary.update(line)

    print(f"Using {N} samples, containing {len(vocabulary)} different words")

    model = Word2VecModel(vocabulary, EMB_SIZE)

    model.train(training_data, ITER_NUM, learning_rate=LR)
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
