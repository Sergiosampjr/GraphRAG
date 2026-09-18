from pathlib import Path
import math

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

INDEX_FILE = Path("data/rag/juristcu.faiss")
METADATA_FILE = Path("data/rag/metadata.parquet")

OUTPUT_DIR = Path("outputs/rag")

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

TOP_CHUNKS = 200
K = 10


def buscar_documentos(
    consulta,
    model,
    index,
    metadata,
    top_chunks=200,
    top_docs=10
):
    query_text = f"query: {consulta}"

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_chunks
    )

    documentos = []
    vistos = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        row = metadata.iloc[idx]

        doc_id = int(row["doc_id"])

        if doc_id in vistos:
            continue

        vistos.add(doc_id)

        documentos.append({
            "doc_id": doc_id,
            "score_vetorial": float(score),
            "chunk_id": row["chunk_id"]
        })

        if len(documentos) == top_docs:
            break

    return documentos


def precision_at_k(recuperados, relevantes, k=10):
    recuperados = recuperados[:k]

    acertos = sum(
        1 for doc_id in recuperados
        if doc_id in relevantes
    )

    return acertos / k


def recall_at_k(recuperados, relevantes, k=10):
    if not relevantes:
        return 0.0

    recuperados = recuperados[:k]

    acertos = sum(
        1 for doc_id in recuperados
        if doc_id in relevantes
    )

    return acertos / len(relevantes)


def mrr_at_k(recuperados, relevantes, k=10):

    for rank, doc_id in enumerate(
        recuperados[:k],
        start=1
    ):
        if doc_id in relevantes:
            return 1.0 / rank

    return 0.0


def dcg_at_k(recuperados, relevancias, k=10):

    dcg = 0.0

    for rank, doc_id in enumerate(
        recuperados[:k],
        start=1
    ):

        rel = relevancias.get(doc_id, 0)

        # ganho graduado
        ganho = (2 ** rel) - 1

        dcg += ganho / math.log2(rank + 1)

    return dcg


def ndcg_at_k(recuperados, relevancias, k=10):

    dcg = dcg_at_k(
        recuperados,
        relevancias,
        k
    )

    relevancias_ideais = sorted(
        relevancias.values(),
        reverse=True
    )[:k]

    if not relevancias_ideais:
        return 0.0

    idcg = 0.0

    for rank, rel in enumerate(
        relevancias_ideais,
        start=1
    ):
        ganho = (2 ** rel) - 1

        idcg += ganho / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Carregando modelo...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Carregando índice...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    assert index.ntotal == len(metadata)

    print("Carregando benchmark...")

    queries = load_dataset(
        "csv",
        data_files=BASE_URL + "query.csv",
        split="train"
    ).to_pandas()

    qrels = load_dataset(
        "csv",
        data_files=BASE_URL + "qrel.csv",
        split="train"
    ).to_pandas()

    resultados_metricas = []
    resultados_busca = []

    total = len(queries)

    for contador, (_, query) in enumerate(
        queries.iterrows(),
        start=1
    ):

        query_id = int(query["ID"])
        texto = query["TEXT"]
        source = query["SOURCE"]

        print(
            f"[{contador:03d}/{total}] "
            f"Query {query_id}: {texto}"
        )

        docs = buscar_documentos(
            consulta=texto,
            model=model,
            index=index,
            metadata=metadata,
            top_chunks=TOP_CHUNKS,
            top_docs=K
        )

        recuperados = [
            d["doc_id"]
            for d in docs
        ]

        qrels_query = qrels[
            qrels["QUERY_ID"] == query_id
        ]

        relevancias = {
            int(row["DOC_ID"]): int(row["SCORE"])
            for _, row in qrels_query.iterrows()
        }

        # Mesmo critério do artigo:
        # SCORE != 0 é considerado relevante.
        relevantes = {
            doc_id
            for doc_id, score in relevancias.items()
            if score > 0
        }

        precision = precision_at_k(
            recuperados,
            relevantes,
            K
        )

        recall = recall_at_k(
            recuperados,
            relevantes,
            K
        )

        mrr = mrr_at_k(
            recuperados,
            relevantes,
            K
        )

        ndcg = ndcg_at_k(
            recuperados,
            relevancias,
            K
        )

        resultados_metricas.append({
            "query_id": query_id,
            "query": texto,
            "source": source,
            "precision_at_10": precision,
            "recall_at_10": recall,
            "mrr_at_10": mrr,
            "ndcg_at_10": ndcg,
            "num_relevantes_qrel": len(relevantes)
        })

        for rank, doc in enumerate(
            docs,
            start=1
        ):

            doc_id = doc["doc_id"]

            resultados_busca.append({
                "query_id": query_id,
                "query": texto,
                "source": source,
                "rank": rank,
                "doc_id": doc_id,
                "chunk_id": doc["chunk_id"],
                "score_vetorial": doc["score_vetorial"],
                "qrel_score": relevancias.get(
                    doc_id,
                    np.nan
                ),
                "julgado": doc_id in relevancias
            })

    metricas_df = pd.DataFrame(
        resultados_metricas
    )

    busca_df = pd.DataFrame(
        resultados_busca
    )

    metricas_df.to_csv(
        OUTPUT_DIR / "metricas_por_query.csv",
        index=False
    )

    busca_df.to_csv(
        OUTPUT_DIR / "resultados_top10.csv",
        index=False
    )

    print("\n" + "=" * 75)
    print("RESULTADO GLOBAL")
    print("=" * 75)

    colunas_metricas = [
        "precision_at_10",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10"
    ]

    for coluna in colunas_metricas:
        print(
            f"{coluna:<20}: "
            f"{metricas_df[coluna].mean():.4f}"
        )

    print("\n" + "=" * 75)
    print("RESULTADO POR SOURCE")
    print("=" * 75)

    resumo_source = (
        metricas_df
        .groupby("source")[
            colunas_metricas
        ]
        .mean()
    )

    print(resumo_source)

    print("\n" + "=" * 75)
    print("COBERTURA DOS QRELS NO TOP-10")
    print("=" * 75)

    julgados = busca_df["julgado"].mean()

    print(
        f"Resultados recuperados que possuem qrel: "
        f"{julgados:.2%}"
    )

    print("\nArquivos salvos:")

    print(
        OUTPUT_DIR / "metricas_por_query.csv"
    )

    print(
        OUTPUT_DIR / "resultados_top10.csv"
    )


if __name__ == "__main__":
    main()