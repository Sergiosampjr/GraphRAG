from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

model = SentenceTransformer(MODEL_NAME)

print("Modelo:", MODEL_NAME)
print("max_seq_length:", model.max_seq_length)

texto = " ".join(["palavra"] * 500)

tokens = model.tokenizer(
    texto,
    truncation=False,
    return_tensors=None
)

print("Tokens para 500 palavras artificiais:", len(tokens["input_ids"]))