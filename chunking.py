from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


INPUT_FILE = Path("data/processed/juristcu_documentos.parquet")
OUTPUT_FILE = Path("data/processed/juristcu_chunks.parquet")

MODEL_NAME = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 480
CHUNK_OVERLAP = 80


def criar_chunks_tokens(
    texto,
    tokenizer,
    chunk_size=480,
    overlap=80
):
    if not texto:
        return []

    token_ids = tokenizer.encode(
        texto,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []

    inicio = 0

    while inicio < len(token_ids):

        fim = min(
            inicio + chunk_size,
            len(token_ids)
        )

        chunk_tokens = token_ids[inicio:fim]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if fim >= len(token_ids):
            break

        inicio = fim - overlap

    return chunks


def main():

    print("Carregando corpus...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Documentos: {len(df)}")

    print("\nCarregando tokenizer...")
    print(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap: {CHUNK_OVERLAP}")

    registros = []

    print("\nGerando chunks...")

    for _, documento in df.iterrows():

        chunks = criar_chunks_tokens(
            documento["text"],
            tokenizer
        )

        for chunk_index, chunk_text in enumerate(chunks):

            registros.append({
                "chunk_id": (
                    f"{documento['doc_id']}_{chunk_index}"
                ),

                "doc_id": int(documento["doc_id"]),
                "chunk_index": chunk_index,

                # Prefixo exigido pelo E5
                "text": f"passage: {chunk_text}",

                "area": documento["area"],
                "tema": documento["tema"],
                "subtema": documento["subtema"],
                "autor_tese": documento["autor_tese"],
                "tipo_processo": documento["tipo_processo"],
                "indexacao": documento["indexacao"],
                "referencia_legal": documento[
                    "referencia_legal"
                ],
            })

    chunks_df = pd.DataFrame(registros)

    print("\n" + "=" * 70)
    print("CHUNKING CONCLUÍDO")
    print("=" * 70)

    print(f"Documentos: {len(df)}")
    print(f"Chunks: {len(chunks_df)}")

    distribuicao = (
        chunks_df
        .groupby("doc_id")
        .size()
    )

    print(
        f"Média chunks/documento: "
        f"{distribuicao.mean():.2f}"
    )

    print("\nDistribuição:")

    print(
        distribuicao.describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # --------------------------------------------------
    # VALIDAÇÃO REAL
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDAÇÃO")
    print("=" * 70)

    tamanhos = []

    chunks_acima_limite = 0

    for texto in chunks_df["text"]:

        tamanho = len(
            tokenizer.encode(
                texto,
                add_special_tokens=True,
                truncation=False
            )
        )

        tamanhos.append(tamanho)

        if tamanho > 512:
            chunks_acima_limite += 1

    chunks_df["num_tokens"] = tamanhos

    print(
        chunks_df["num_tokens"].describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    print(
        f"\nMaior chunk: "
        f"{chunks_df['num_tokens'].max()} tokens"
    )

    print(
        f"Chunks acima de 512: "
        f"{chunks_acima_limite}"
    )

    if chunks_acima_limite == 0:
        print(
            "\nOK: nenhum chunk ultrapassa "
            "o limite do modelo."
        )
    else:
        print(
            "\nATENÇÃO: existem chunks "
            "acima do limite."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nArquivo salvo em: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()