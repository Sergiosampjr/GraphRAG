from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/juristcu_chunks.parquet")

OUTPUT_DIR = Path("data/rag")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

FAISS_FILE = OUTPUT_DIR / "juristcu.faiss"
METADATA_FILE = OUTPUT_DIR / "metadata.parquet"

MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 32

# Quantos batches processar antes de salvar um checkpoint.
# Com 32 textos por batch:
#
# 10 batches = 320 chunks por checkpoint
#
CHECKPOINT_EVERY = 10


def salvar_estado(proximo_indice, total_chunks):
    """
    Salva o ponto de retomada da indexação.
    """

    estado = {
        "proximo_indice": int(proximo_indice),
        "total_chunks": int(total_chunks),
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
    }

    arquivo_estado = CHECKPOINT_DIR / "estado.json"

    with open(
        arquivo_estado,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            estado,
            f,
            indent=2,
            ensure_ascii=False
        )


def carregar_estado(total_chunks):
    """
    Verifica se existe uma execução anterior incompleta.
    """

    arquivo_estado = CHECKPOINT_DIR / "estado.json"

    if not arquivo_estado.exists():
        return 0

    with open(
        arquivo_estado,
        "r",
        encoding="utf-8"
    ) as f:
        estado = json.load(f)

    if estado["total_chunks"] != total_chunks:
        raise ValueError(
            "O número de chunks mudou desde o último checkpoint."
        )

    if estado["model_name"] != MODEL_NAME:
        raise ValueError(
            "O modelo de embeddings mudou desde o último checkpoint."
        )

    proximo_indice = int(
        estado["proximo_indice"]
    )

    print("\nCheckpoint encontrado.")

    print(
        f"Retomando a partir do chunk "
        f"{proximo_indice:,}."
    )

    return proximo_indice


def caminho_batch(inicio, fim):
    """
    Nome do arquivo contendo embeddings de um intervalo.
    """

    return CHECKPOINT_DIR / (
        f"embeddings_{inicio:06d}_{fim:06d}.npy"
    )


def gerar_embeddings(
    df,
    model,
    inicio
):
    """
    Gera embeddings em pequenos lotes e salva cada bloco
    periodicamente em arquivos .npy.
    """

    total = len(df)

    buffer_embeddings = []
    inicio_buffer = inicio

    batches_desde_checkpoint = 0

    for batch_inicio in range(
        inicio,
        total,
        BATCH_SIZE
    ):

        batch_fim = min(
            batch_inicio + BATCH_SIZE,
            total
        )

        textos = (
            df.iloc[
                batch_inicio:batch_fim
            ]["text"]
            .tolist()
        )

        embeddings = model.encode(
            textos,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        buffer_embeddings.append(
            embeddings
        )

        batches_desde_checkpoint += 1

        print(
            f"\rProcessados: "
            f"{batch_fim:,}/{total:,} "
            f"({batch_fim / total * 100:.2f}%)",
            end="",
            flush=True
        )

        deve_salvar = (
            batches_desde_checkpoint
            >= CHECKPOINT_EVERY
            or batch_fim == total
        )

        if deve_salvar:

            bloco = np.vstack(
                buffer_embeddings
            )

            arquivo = caminho_batch(
                inicio_buffer,
                batch_fim
            )

            np.save(
                arquivo,
                bloco
            )

            salvar_estado(
                batch_fim,
                total
            )

            print(
                f"\nCheckpoint salvo: "
                f"{arquivo.name}"
            )

            buffer_embeddings = []

            inicio_buffer = batch_fim

            batches_desde_checkpoint = 0


def carregar_todos_embeddings(total_chunks):
    """
    Carrega os arquivos .npy na ordem correta.
    """

    arquivos = sorted(
        CHECKPOINT_DIR.glob(
            "embeddings_*.npy"
        )
    )

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum checkpoint de embeddings encontrado."
        )

    print(
        f"\nCarregando {len(arquivos)} "
        f"arquivos de embeddings..."
    )

    blocos = []

    for arquivo in arquivos:

        print(
            f"  {arquivo.name}"
        )

        embeddings = np.load(
            arquivo
        )

        blocos.append(
            embeddings
        )

    embeddings = np.vstack(
        blocos
    ).astype(
        "float32"
    )

    if len(embeddings) != total_chunks:
        raise ValueError(
            "\nQuantidade de embeddings diferente "
            "da quantidade de chunks.\n"
            f"Embeddings: {len(embeddings)}\n"
            f"Chunks: {total_chunks}"
        )

    return embeddings


def salvar_metadata(df):
    """
    Salva os metadados exatamente na mesma ordem dos chunks.
    """

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
        METADATA_FILE,
        index=False
    )

    print(
        f"Metadados salvos: "
        f"{METADATA_FILE}"
    )


def construir_faiss(embeddings):
    """
    Cria índice FAISS usando produto interno.

    Como os embeddings foram normalizados,
    produto interno equivale à similaridade
    de cosseno.
    """

    dimensao = embeddings.shape[1]

    print("\nCriando índice FAISS...")

    print(
        f"Embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"Dimensão: "
        f"{dimensao}"
    )

    index = faiss.IndexFlatIP(
        dimensao
    )

    index.add(
        embeddings
    )

    print(
        f"Vetores indexados: "
        f"{index.ntotal}"
    )

    faiss.write_index(
        index,
        str(FAISS_FILE)
    )

    print(
        f"Índice salvo: "
        f"{FAISS_FILE}"
    )

    return index


def validar_resultado(
    index,
    df
):
    """
    Validação final para garantir que o FAISS
    está alinhado com os chunks.
    """

    print("\nValidando resultado...")

    if index.ntotal != len(df):
        raise ValueError(
            "Número de vetores no FAISS "
            "é diferente do número de chunks."
        )

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    if len(metadata) != len(df):
        raise ValueError(
            "Metadata possui quantidade "
            "incorreta de linhas."
        )

    print(
        "FAISS e metadata estão alinhados."
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("INDEXAÇÃO RAG - JurisTCU")
    print("=" * 70)

    print("\nCarregando chunks...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    total_chunks = len(df)

    print(
        f"Chunks: "
        f"{total_chunks:,}"
    )

    print("\nCarregando modelo de embeddings...")

    print(
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"max_seq_length: "
        f"{model.max_seq_length}"
    )

    # --------------------------------------------------------
    # Descobre de onde continuar
    # --------------------------------------------------------

    inicio = carregar_estado(
        total_chunks
    )

    # --------------------------------------------------------
    # Gera embeddings faltantes
    # --------------------------------------------------------

    if inicio < total_chunks:

        print("\nGerando embeddings...")

        gerar_embeddings(
            df=df,
            model=model,
            inicio=inicio
        )

    else:

        print(
            "\nTodos os embeddings já foram gerados."
        )

    # --------------------------------------------------------
    # Reúne checkpoints
    # --------------------------------------------------------

    embeddings = carregar_todos_embeddings(
        total_chunks
    )

    print(
        "\nFormato final dos embeddings:"
    )

    print(
        embeddings.shape
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = construir_faiss(
        embeddings
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    salvar_metadata(
        df
    )

    # --------------------------------------------------------
    # Validação
    # --------------------------------------------------------

    validar_resultado(
        index,
        df
    )

    print("\n" + "=" * 70)
    print("INDEXAÇÃO CONCLUÍDA")
    print("=" * 70)

    print(
        f"Modelo: "
        f"{MODEL_NAME}"
    )

    print(
        f"Chunks indexados: "
        f"{index.ntotal:,}"
    )

    print(
        f"Dimensão: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Índice: "
        f"{FAISS_FILE}"
    )

    print(
        f"Metadados: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()