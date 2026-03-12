from model import Word2VecModel

MODEL_PATH = "./saved_model"

w2v = Word2VecModel.load(MODEL_PATH)

# get the top 10 words similar to the word "embedding"
embedding_emb = w2v.get_embedding("embedding")
top10 = w2v.get_top_k_similar(embedding_emb, k=10)
print(top10)
