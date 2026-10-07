import gc
import json
import os
from pathlib import Path

# Limita uso de CPU antes de importar torch / BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAG_PATH = BASE_DIR / "outputs" / "rag" / "resultados_top10.csv"
GRAPHRAG_PATH = BASE_DIR / "outputs" / "graphrag" / "resultados_top10.csv"
HIBRIDO_PATH = BASE_DIR / "outputs" / "hibrido" / "resultados_top10.csv"

METADATA_PATH = BASE_DIR / "data" / "rag" / "metadata.parquet"
FAISS_PATH = BASE_DIR / "data" / "rag" / "juristcu.faiss"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "juristcu_chunks.parquet"

OUTPUT_DIR = BASE_DIR / "outputs" / "generation" / "contexts"
OUTPUT_JSONL = OUTPUT_DIR / "piloto_contextos.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "piloto_contextos.csv"
OUTPUT_QUERIES = OUTPUT_DIR / "piloto_queries.csv"

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"

TOP_K_DOCUMENTOS = 10
QUERIES_POR_FONTE = 2

# Mantemos CPU para evitar tentativa de GPU inexistente/dedicada.
DEVICE = "cpu"

# Tamanho do lote usado apenas para as 6 consultas do piloto.
BATCH_SIZE_QUERY = 1


# ==========================================================
# UTILITÁRIOS
# ==========================================================

def verificar_arquivos():
    arquivos = [
        RAG_PATH,
        GRAPHRAG_PATH,
        HIBRIDO_PATH,
        METADATA_PATH,
        FAISS_PATH,
        CHUNKS_PATH,
    ]

    faltando = [str(p) for p in arquivos if not p.exists()]

    if faltando:
        raise FileNotFoundError(
            "Arquivos necessários não encontrados:\n- "
            + "\n- ".join(faltando)
        )


def normalizar_rankings():
    """
    Carrega e padroniza apenas as colunas necessárias.
    qrels e rótulos de avaliação não entram na geração.
    """

    rag = pd.read_csv(
        RAG_PATH,
        usecols=["query_id", "query", "source", "rank", "doc_id"],
    )

    graphrag = pd.read_csv(
        GRAPHRAG_PATH,
        usecols=[
            "query_id",
            "pergunta",
            "source",
            "rank_final",
            "doc_id",
        ],
    )

    hibrido = pd.read_csv(
        HIBRIDO_PATH,
        usecols=[
            "query_id",
            "pergunta",
            "source",
            "rank_final",
            "doc_id",
        ],
    )

    rag = rag.rename(
        columns={
            "query": "pergunta",
            "rank": "rank_final",
        }
    )
    rag["arquitetura"] = "rag"
    graphrag["arquitetura"] = "graphrag"
    hibrido["arquitetura"] = "hibrido"

    rankings = pd.concat(
        [rag, graphrag, hibrido],
        ignore_index=True,
    )

    rankings["query_id"] = rankings["query_id"].astype(int)
    rankings["doc_id"] = rankings["doc_id"].astype(int)
    rankings["rank_final"] = rankings["rank_final"].astype(int)

    return rankings


def selecionar_queries_piloto(rankings):
    consultas = (
        rankings[
            ["query_id", "pergunta", "source"]
        ]
        .drop_duplicates(subset=["query_id"])
        .sort_values(["source", "query_id"])
        .reset_index(drop=True)
    )

    contagem = consultas.groupby("source")["query_id"].nunique()

    fontes_insuficientes = contagem[
        contagem < QUERIES_POR_FONTE
    ]

    if not fontes_insuficientes.empty:
        raise ValueError(
            "Há fonte(s) com menos consultas do que o necessário:\n"
            + fontes_insuficientes.to_string()
        )

    piloto = (
        consultas
        .groupby("source", group_keys=False)
        .head(QUERIES_POR_FONTE)
        .sort_values(["source", "query_id"])
        .reset_index(drop=True)
    )

    return piloto


def preparar_query_e5(pergunta):
    pergunta = str(pergunta).strip()

    if pergunta.lower().startswith("query:"):
        return pergunta

    return f"query: {pergunta}"


def limpar_passage(texto):
    texto = str(texto).strip()

    if texto.lower().startswith("passage:"):
        return texto[len("passage:"):].strip()

    return texto


# ==========================================================
# MAPA DOC_ID -> POSIÇÕES NO FAISS
# ==========================================================

def construir_mapa_doc_posicoes(metadata):
    """
    Como metadata.parquet possui exatamente uma linha por vetor
    do índice FAISS e a ordem coincide com o índice, o índice
    da linha é a posição do vetor no FAISS.
    """

    mapa = {}

    for posicao, doc_id in enumerate(
        metadata["doc_id"].astype(int).tolist()
    ):
        mapa.setdefault(doc_id, []).append(posicao)

    return mapa


# ==========================================================
# SELEÇÃO DO MELHOR CHUNK
# ==========================================================

def selecionar_melhor_chunk(
    query_embedding,
    doc_id,
    metadata,
    mapa_doc_posicoes,
    index,
):
    doc_id = int(doc_id)

    posicoes = mapa_doc_posicoes.get(doc_id)

    if not posicoes:
        raise ValueError(
            f"Nenhuma posição FAISS encontrada para doc_id={doc_id}"
        )

    # Reconstrói SOMENTE os vetores dos chunks deste documento.
    vetores = np.vstack(
        [
            index.reconstruct(int(pos))
            for pos in posicoes
        ]
    ).astype("float32", copy=False)

    # Os vetores do índice já foram produzidos na indexação.
    # Como a consulta será normalizada, usamos produto interno.
    scores = vetores @ query_embedding

    indice_local = int(np.argmax(scores))
    posicao_vencedora = int(posicoes[indice_local])

    linha = metadata.iloc[posicao_vencedora]

    return {
        "chunk_id": str(linha["chunk_id"]),
        "doc_id": int(linha["doc_id"]),
        "chunk_index": int(linha["chunk_index"]),
        "num_tokens": int(linha["num_tokens"]),
        "score_query_chunk": float(scores[indice_local]),
        "faiss_position": posicao_vencedora,
    }


# ==========================================================
# LEITURA LEVE DOS TEXTOS SELECIONADOS
# ==========================================================

def carregar_textos_chunks_selecionados(chunk_ids):
    """
    Percorre o parquet em lotes e guarda texto apenas dos chunk_ids
    selecionados. Evita carregar todos os 51.818 textos em RAM.
    """

    procurados = set(str(x) for x in chunk_ids)
    encontrados = {}

    parquet = pq.ParquetFile(CHUNKS_PATH)

    for lote in parquet.iter_batches(
        batch_size=2048,
        columns=[
            "chunk_id",
            "text",
        ],
    ):
        df_lote = lote.to_pandas()

        mascara = df_lote["chunk_id"].astype(str).isin(procurados)

        if mascara.any():
            subset = df_lote.loc[
                mascara,
                ["chunk_id", "text"],
            ]

            for _, linha in subset.iterrows():
                chunk_id = str(linha["chunk_id"])
                encontrados[chunk_id] = limpar_passage(
                    linha["text"]
                )

        if len(encontrados) == len(procurados):
            break

        del df_lote
        gc.collect()

    faltando = procurados - set(encontrados)

    if faltando:
        raise ValueError(
            "Não foi possível localizar texto para os chunks: "
            + ", ".join(sorted(faltando))
        )

    return encontrados


# ==========================================================
# MAIN
# ==========================================================

def main():
    verificar_arquivos()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Pode ocorrer se o runtime já inicializou o pool.
        pass

    print("=" * 80)
    print("PREPARAÇÃO LEVE DE CONTEXTOS — PILOTO JURISTCU")
    print("=" * 80)

    # ------------------------------------------------------
    # Rankings
    # ------------------------------------------------------

    print("\n[1/7] Carregando rankings congelados...")
    rankings = normalizar_rankings()

    piloto = selecionar_queries_piloto(rankings)

    print("\nConsultas selecionadas:")
    print(
        piloto[
            ["query_id", "source", "pergunta"]
        ].to_string(index=False)
    )

    piloto.to_csv(
        OUTPUT_QUERIES,
        index=False,
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Metadata + FAISS
    # ------------------------------------------------------

    print("\n[2/7] Carregando metadata do FAISS...")

    metadata = pd.read_parquet(
        METADATA_PATH,
        columns=[
            "chunk_id",
            "doc_id",
            "chunk_index",
            "num_tokens",
        ],
    )

    metadata["doc_id"] = metadata["doc_id"].astype(int)

    print(
        f"metadata: {len(metadata)} linhas"
    )

    print("\n[3/7] Carregando índice FAISS...")

    index = faiss.read_index(str(FAISS_PATH))

    print(
        f"FAISS: {index.ntotal} vetores, dimensão {index.d}"
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS e metadata possuem tamanhos diferentes."
        )

    mapa_doc_posicoes = construir_mapa_doc_posicoes(
        metadata
    )

    # ------------------------------------------------------
    # Descobrir apenas os documentos necessários
    # ------------------------------------------------------

    arquiteturas = ["rag", "graphrag", "hibrido"]

    tarefas = []

    for _, consulta in piloto.iterrows():
        query_id = int(consulta["query_id"])
        pergunta = str(consulta["pergunta"])
        source = str(consulta["source"])

        for arquitetura in arquiteturas:
            ranking_query = (
                rankings[
                    (rankings["query_id"] == query_id)
                    & (
                        rankings["arquitetura"]
                        == arquitetura
                    )
                ]
                .sort_values("rank_final")
                .head(TOP_K_DOCUMENTOS)
            )

            if len(ranking_query) != TOP_K_DOCUMENTOS:
                raise ValueError(
                    f"Esperados {TOP_K_DOCUMENTOS} docs para "
                    f"query_id={query_id}, arquitetura={arquitetura}, "
                    f"mas foram encontrados {len(ranking_query)}."
                )

            tarefas.append(
                {
                    "query_id": query_id,
                    "pergunta": pergunta,
                    "source": source,
                    "arquitetura": arquitetura,
                    "ranking": ranking_query,
                }
            )

    print(
        f"\nPiloto: {len(piloto)} consultas, "
        f"{len(tarefas)} contextos."
    )

    # ------------------------------------------------------
    # Modelo: apenas embeddings das 6 perguntas
    # ------------------------------------------------------

    print(
        "\n[4/7] Carregando multilingual-e5-base em CPU "
        "com apenas 2 threads..."
    )
    print(
        "Nesta versão o modelo NÃO re-embedará os 51.818 chunks."
    )

    modelo = SentenceTransformer(
        MODELO_EMBEDDING,
        device=DEVICE,
    )

    perguntas_unicas = {
        int(linha["query_id"]): str(linha["pergunta"])
        for _, linha in piloto.iterrows()
    }

    query_embeddings = {}

    print(
        f"Gerando somente {len(perguntas_unicas)} embeddings "
        "de consulta..."
    )

    for i, (query_id, pergunta) in enumerate(
        perguntas_unicas.items(),
        start=1,
    ):
        print(
            f"  consulta {i}/{len(perguntas_unicas)} "
            f"(query_id={query_id})"
        )

        emb = modelo.encode(
            [preparar_query_e5(pergunta)],
            batch_size=BATCH_SIZE_QUERY,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32", copy=False)

        query_embeddings[query_id] = emb

    # ------------------------------------------------------
    # Selecionar melhor chunk via vetores JÁ existentes
    # ------------------------------------------------------

    print(
        "\n[5/7] Selecionando melhor chunk por documento "
        "reutilizando vetores do FAISS..."
    )

    registros_parciais = []
    chunk_ids_necessarios = set()

    cache_query_doc = {}

    total = len(tarefas)

    for numero, tarefa in enumerate(
        tarefas,
        start=1,
    ):
        query_id = tarefa["query_id"]
        arquitetura = tarefa["arquitetura"]
        query_embedding = query_embeddings[query_id]

        print(
            f"  [{numero}/{total}] query_id={query_id} | "
            f"{arquitetura}"
        )

        documentos = []

        for _, linha in tarefa["ranking"].iterrows():
            rank = int(linha["rank_final"])
            doc_id = int(linha["doc_id"])

            chave = (query_id, doc_id)

            if chave not in cache_query_doc:
                cache_query_doc[chave] = selecionar_melhor_chunk(
                    query_embedding=query_embedding,
                    doc_id=doc_id,
                    metadata=metadata,
                    mapa_doc_posicoes=mapa_doc_posicoes,
                    index=index,
                )

            melhor = cache_query_doc[chave].copy()
            melhor["rank"] = rank

            documentos.append(melhor)
            chunk_ids_necessarios.add(
                melhor["chunk_id"]
            )

        registros_parciais.append(
            {
                "query_id": query_id,
                "source": tarefa["source"],
                "pergunta": tarefa["pergunta"],
                "arquitetura": arquitetura,
                "documentos": documentos,
            }
        )

    # Libera o modelo antes de carregar os textos.
    del modelo
    query_embeddings.clear()
    gc.collect()

    print(
        f"\nChunks únicos selecionados: "
        f"{len(chunk_ids_necessarios)}"
    )

    # ------------------------------------------------------
    # Textos só dos chunks escolhidos
    # ------------------------------------------------------

    print(
        "\n[6/7] Lendo do parquet somente os textos "
        "dos chunks selecionados..."
    )

    textos = carregar_textos_chunks_selecionados(
        chunk_ids_necessarios
    )

    # ------------------------------------------------------
    # Montar contextos
    # ------------------------------------------------------

    print("\n[7/7] Montando e salvando contextos...")

    registros = []

    for parcial in registros_parciais:
        blocos = []

        documentos_saida = []

        for doc in parcial["documentos"]:
            chunk_id = doc["chunk_id"]
            doc_id = doc["doc_id"]

            texto = textos[chunk_id]

            blocos.append(
                f"[DOC_{doc_id}]\n{texto}"
            )

            documentos_saida.append(
                {
                    "rank": int(doc["rank"]),
                    "doc_id": int(doc_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": int(
                        doc["chunk_index"]
                    ),
                    "num_tokens": int(
                        doc["num_tokens"]
                    ),
                    "score_query_chunk": float(
                        doc["score_query_chunk"]
                    ),
                }
            )

        contexto = "\n\n".join(blocos)

        registro = {
            "query_id": parcial["query_id"],
            "source": parcial["source"],
            "pergunta": parcial["pergunta"],
            "arquitetura": parcial["arquitetura"],
            "num_documentos": len(
                documentos_saida
            ),
            "total_tokens_chunks": int(
                sum(
                    x["num_tokens"]
                    for x in documentos_saida
                )
            ),
            "doc_ids": [
                x["doc_id"]
                for x in documentos_saida
            ],
            "chunk_ids": [
                x["chunk_id"]
                for x in documentos_saida
            ],
            "documentos": documentos_saida,
            "contexto": contexto,
        }

        registros.append(registro)

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as arquivo:
        for registro in registros:
            arquivo.write(
                json.dumps(
                    registro,
                    ensure_ascii=False,
                )
                + "\n"
            )

    df_saida = pd.DataFrame(
        [
            {
                "query_id": r["query_id"],
                "source": r["source"],
                "pergunta": r["pergunta"],
                "arquitetura": r["arquitetura"],
                "num_documentos": r["num_documentos"],
                "total_tokens_chunks": r[
                    "total_tokens_chunks"
                ],
                "doc_ids": json.dumps(
                    r["doc_ids"]
                ),
                "chunk_ids": json.dumps(
                    r["chunk_ids"]
                ),
                "contexto": r["contexto"],
            }
            for r in registros
        ]
    )

    df_saida.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("PREPARAÇÃO CONCLUÍDA")
    print("=" * 80)

    print(
        f"\nConsultas do piloto: "
        f"{piloto['query_id'].nunique()}"
    )
    print(
        f"Contextos gerados: {len(registros)}"
    )

    print("\nArquivos:")
    print(f"- {OUTPUT_QUERIES}")
    print(f"- {OUTPUT_JSONL}")
    print(f"- {OUTPUT_CSV}")

    resumo = (
        df_saida
        .groupby("arquitetura")
        .agg(
            contextos=("query_id", "count"),
            media_docs=("num_documentos", "mean"),
            media_tokens=(
                "total_tokens_chunks",
                "mean",
            ),
            min_tokens=(
                "total_tokens_chunks",
                "min",
            ),
            max_tokens=(
                "total_tokens_chunks",
                "max",
            ),
        )
    )

    print("\nResumo por arquitetura:")
    print(resumo.to_string())


if __name__ == "__main__":
    main()
