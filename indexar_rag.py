from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/juristcu_chunks.parquet")
OUTPUT_DIR = Path("data/rag")

MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 32


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando chunks...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Chunks: {len(df)}")

    print("\nCarregando modelo de embeddings...")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print(f"max_seq_length: {model.max_seq_length}")

    textos = df["text"].tolist()

    print("\nGerando embeddings...")

    embeddings = model.encode(
        textos,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print("\nFormato dos embeddings:")
    print(embeddings.shape)

    print("\nCriando índice FAISS...")

    dimensao = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimensao)

    index.add(embeddings)

    print(f"Vetores indexados: {index.ntotal}")

    # Salva índice FAISS
    faiss.write_index(
        index,
        str(OUTPUT_DIR / "juristcu.faiss")
    )

    # A ordem dessas linhas deve ser exatamente a mesma
    # da ordem dos embeddings inseridos no FAISS.
    metadata = df[
        [
            "chunk_id",
            "doc_id",
            "chunk_index",
            "area",
            "tema",
            "subtema",
            "num_tokens"
        ]
    ].copy()

    metadata.to_parquet(
        OUTPUT_DIR / "metadata.parquet",
        index=False
    )

    print("\n" + "=" * 70)
    print("INDEXAÇÃO CONCLUÍDA")
    print("=" * 70)

    print(f"Modelo: {MODEL_NAME}")
    print(f"Chunks indexados: {index.ntotal}")
    print(f"Dimensão: {dimensao}")

    print(
        f"Índice: "
        f"{OUTPUT_DIR / 'juristcu.faiss'}"
    )

    print(
        f"Metadados: "
        f"{OUTPUT_DIR / 'metadata.parquet'}"
    )


if __name__ == "__main__":
    main()