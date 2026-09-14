from FlagEmbedding import BGEM3FlagModel

# Загрузка модели (use_fp16=True для ускорения на GPU)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["Как работает BGE M3?", "Определение BM25"]
embeddings = model.encode(sentences, batch_size=12, max_length=8192)['dense_vecs']
print(embeddings.shape) # (2, 1024) - BGE-M3 создает векторы размерностью 1024[reference:6]
