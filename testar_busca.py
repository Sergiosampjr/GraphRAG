from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


MODEL_NAME = "intfloat/multilingual-e5-base"

INDEX_FILE = Path("data/rag/juristcu.faiss")
METADATA_FILE = Path("data/rag/metadata.parquet")

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

TOP_CHUNKS = 100
TOP_DOCS = 10


def buscar_documentos(
    consulta,
    model,
    index,
    metadata,
    top_chunks=100,
    top_docs=10,
):
    # E5 exige prefixo para queries
    query_text = f"query: {consulta}"

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_chunks,
    )

    resultados = []

    docs_vistos = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        row = metadata.iloc[idx]

        doc_id = int(row["doc_id"])

        # Mantém apenas o chunk mais bem ranqueado
        # de cada documento.
        if doc_id in docs_vistos:
            continue

        docs_vistos.add(doc_id)

        resultados.append(
            {
                "rank": len(resultados) + 1,
                "doc_id": doc_id,
                "chunk_id": row["chunk_id"],
                "chunk_index": int(row["chunk_index"]),
                "score_vetorial": float(score),
                "area": row["area"],
                "tema": row["tema"],
                "subtema": row["subtema"],
            }
        )

        if len(resultados) >= top_docs:
            break

    return pd.DataFrame(resultados)


def main():

    print("Carregando modelo...")

    model = SentenceTransformer(MODEL_NAME)

    print("Carregando índice FAISS...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    print(f"Vetores no índice: {index.ntotal}")
    print(f"Linhas de metadata: {len(metadata)}")

    # Validação importante
    assert index.ntotal == len(metadata), (
        "FAISS e metadata estão desalinhados!"
    )

    print("\nCarregando queries e qrels...")

    consultas = load_dataset(
        "csv",
        data_files=BASE_URL + "query.csv",
        split="train",
    )

    qrels = load_dataset(
        "csv",
        data_files=BASE_URL + "qrel.csv",
        split="train",
    )

    consultas_df = consultas.to_pandas()
    qrels_df = qrels.to_pandas()

    # -----------------------------------------------------
    # QUERY 1
    # -----------------------------------------------------

    query_id = 1

    query_row = consultas_df[
        consultas_df["ID"] == query_id
    ].iloc[0]

    consulta = query_row["TEXT"]

    print("\n" + "=" * 80)
    print("CONSULTA")
    print("=" * 80)

    print(f"Query ID: {query_id}")
    print(f"Texto: {consulta}")
    print(f"Fonte: {query_row['SOURCE']}")

    resultados = buscar_documentos(
        consulta=consulta,
        model=model,
        index=index,
        metadata=metadata,
        top_chunks=TOP_CHUNKS,
        top_docs=TOP_DOCS,
    )

    print("\n" + "=" * 80)
    print("TOP 10 DOCUMENTOS RECUPERADOS PELO RAG")
    print("=" * 80)

    print(
        resultados[
            [
                "rank",
                "doc_id",
                "chunk_id",
                "score_vetorial",
                "area",
                "tema",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # GROUND TRUTH
    # -----------------------------------------------------

    esperado = (
        qrels_df[
            qrels_df["QUERY_ID"] == query_id
        ][
            [
                "DOC_ID",
                "SCORE",
                "ENGINE",
                "RANK",
            ]
        ]
        .sort_values(
            ["SCORE", "RANK"],
            ascending=[False, True],
        )
    )

    print("\n" + "=" * 80)
    print("GROUND TRUTH JURISTCU")
    print("=" * 80)

    print(
        esperado.to_string(index=False)
    )

    # -----------------------------------------------------
    # CRUZAMENTO
    # -----------------------------------------------------

    scores_relevancia = dict(
        zip(
            esperado["DOC_ID"].astype(int),
            esperado["SCORE"],
        )
    )

    resultados["qrel_score"] = (
        resultados["doc_id"]
        .map(scores_relevancia)
    )

    resultados["julgado"] = (
        resultados["qrel_score"]
        .notna()
    )

    print("\n" + "=" * 80)
    print("RAG x QRELS")
    print("=" * 80)

    print(
        resultados[
            [
                "rank",
                "doc_id",
                "score_vetorial",
                "qrel_score",
                "julgado",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()