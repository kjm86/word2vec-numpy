from model import Word2VecModel
import numpy as np

MODEL_PATH = "./saved_model"

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

w2v = Word2VecModel.load(MODEL_PATH)

# get the top 10 words similar to the word "embedding"
embedding_emb = w2v.get_embedding("embedding")
top10 = w2v.get_top_k_similar(embedding_emb, k=10)
print(top10)
