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
OUTPUT_JSONL = OUTPUT_DIR / "experimento_contextos.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "experimento_contextos.csv"
OUTPUT_QUERIES = OUTPUT_DIR / "experimento_queries.csv"

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"

TOP_K_DOCUMENTOS = 10
NUM_CONSULTAS_ESPERADO = 150
NUM_ARQUITETURAS = 3
NUM_CONTEXTOS_ESPERADO = (
    NUM_CONSULTAS_ESPERADO * NUM_ARQUITETURAS
)

# CPU para manter o consumo previsível.
DEVICE = "cpu"

# As consultas são curtas; batch pequeno e conservador.
BATCH_SIZE_QUERY = 8


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
    Carrega somente as colunas necessárias para construir os contextos.
    Qrels e rótulos de relevância NÃO entram na geração.
    """

    rag = pd.read_csv(
        RAG_PATH,
        usecols=[
            "query_id",
            "query",
            "source",
            "rank",
            "doc_id",
        ],
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


def obter_todas_consultas(rankings):
    consultas = (
        rankings[
            ["query_id", "pergunta", "source"]
        ]
        .drop_duplicates(subset=["query_id"])
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    if consultas["query_id"].nunique() != NUM_CONSULTAS_ESPERADO:
        raise ValueError(
            f"Eram esperadas {NUM_CONSULTAS_ESPERADO} consultas, "
            f"mas foram encontradas "
            f"{consultas['query_id'].nunique()}."
        )

    return consultas


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
    metadata.parquet possui uma linha por vetor do FAISS e a
    ordem coincide com a posição do vetor no índice.
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

    # Reconstrói apenas os vetores dos chunks do documento.
    vetores = np.vstack(
        [
            index.reconstruct(int(pos))
            for pos in posicoes
        ]
    ).astype("float32", copy=False)

    # Vetores normalizados + produto interno = cosine similarity.
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
# LEITURA DOS TEXTOS SELECIONADOS
# ==========================================================

def carregar_textos_chunks_selecionados(chunk_ids):
    """
    Lê do parquet somente os textos dos chunks selecionados,
    evitando carregar todos os 51.818 textos na RAM.
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

        mascara = (
            df_lote["chunk_id"]
            .astype(str)
            .isin(procurados)
        )

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
        amostra = sorted(faltando)[:20]
        raise ValueError(
            "Não foi possível localizar texto para "
            f"{len(faltando)} chunks. Amostra: "
            + ", ".join(amostra)
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
        pass

    print("=" * 80)
    print("PREPARAÇÃO DE CONTEXTOS — EXPERIMENTO PRINCIPAL JURISTCU")
    print("=" * 80)

    # ------------------------------------------------------
    # Rankings e 150 consultas
    # ------------------------------------------------------

    print("\n[1/7] Carregando rankings congelados...")
    rankings = normalizar_rankings()

    consultas = obter_todas_consultas(rankings)

    print(
        f"Consultas únicas encontradas: "
        f"{consultas['query_id'].nunique()}"
    )

    print("\nDistribuição por source:")
    print(
        consultas["source"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    consultas.to_csv(
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

    print(f"metadata: {len(metadata)} linhas")

    print("\n[3/7] Carregando índice FAISS...")

    index = faiss.read_index(str(FAISS_PATH))

    print(
        f"FAISS: {index.ntotal} vetores, "
        f"dimensão {index.d}"
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS e metadata possuem tamanhos diferentes."
        )

    mapa_doc_posicoes = construir_mapa_doc_posicoes(
        metadata
    )

    # ------------------------------------------------------
    # Criar as 450 tarefas
    # ------------------------------------------------------

    arquiteturas = [
        "rag",
        "graphrag",
        "hibrido",
    ]

    tarefas = []

    for _, consulta in consultas.iterrows():
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
                    f"query_id={query_id}, "
                    f"arquitetura={arquitetura}, "
                    f"mas foram encontrados "
                    f"{len(ranking_query)}."
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

    if len(tarefas) != NUM_CONTEXTOS_ESPERADO:
        raise ValueError(
            f"Eram esperados {NUM_CONTEXTOS_ESPERADO} "
            f"contextos, mas foram criadas "
            f"{len(tarefas)} tarefas."
        )

    print(
        f"\nExperimento: {len(consultas)} consultas × "
        f"{len(arquiteturas)} arquiteturas = "
        f"{len(tarefas)} contextos."
    )

    # ------------------------------------------------------
    # Embeddings somente das 150 perguntas
    # ------------------------------------------------------

    print(
        "\n[4/7] Carregando multilingual-e5-base em CPU "
        "com 2 threads..."
    )

    print(
        "Os 51.818 chunks NÃO serão re-embeddados; "
        "serão reutilizados os vetores do FAISS."
    )

    modelo = SentenceTransformer(
        MODELO_EMBEDDING,
        device=DEVICE,
    )

    perguntas_unicas = {
        int(linha["query_id"]): str(linha["pergunta"])
        for _, linha in consultas.iterrows()
    }

    ids = list(perguntas_unicas.keys())
    textos_query = [
        preparar_query_e5(perguntas_unicas[qid])
        for qid in ids
    ]

    print(
        f"Gerando {len(textos_query)} embeddings "
        "de consulta..."
    )

    embeddings = modelo.encode(
        textos_query,
        batch_size=BATCH_SIZE_QUERY,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)

    query_embeddings = {
        qid: embeddings[i]
        for i, qid in enumerate(ids)
    }

    del embeddings
    gc.collect()

    # ------------------------------------------------------
    # Melhor chunk de cada documento recuperado
    # ------------------------------------------------------

    print(
        "\n[5/7] Selecionando melhor chunk por documento "
        "reutilizando os vetores existentes..."
    )

    registros_parciais = []
    chunk_ids_necessarios = set()

    # Evita recalcular se o mesmo doc aparecer em mais de uma arquitetura
    # para a mesma consulta.
    cache_query_doc = {}

    total = len(tarefas)

    for numero, tarefa in enumerate(
        tarefas,
        start=1,
    ):
        query_id = tarefa["query_id"]
        arquitetura = tarefa["arquitetura"]
        query_embedding = query_embeddings[query_id]

        if numero == 1 or numero % 25 == 0 or numero == total:
            print(
                f"  [{numero}/{total}] "
                f"query_id={query_id} | {arquitetura}"
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

    del modelo
    query_embeddings.clear()
    gc.collect()

    print(
        f"\nPares consulta-documento processados: "
        f"{len(cache_query_doc)}"
    )
    print(
        f"Chunks únicos selecionados: "
        f"{len(chunk_ids_necessarios)}"
    )

    # ------------------------------------------------------
    # Textos apenas dos chunks escolhidos
    # ------------------------------------------------------

    print(
        "\n[6/7] Lendo do parquet somente os textos "
        "dos chunks selecionados..."
    )

    textos = carregar_textos_chunks_selecionados(
        chunk_ids_necessarios
    )

    # ------------------------------------------------------
    # Montar e salvar contextos
    # ------------------------------------------------------

    print("\n[7/7] Montando e salvando os 450 contextos...")

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

    if len(registros) != NUM_CONTEXTOS_ESPERADO:
        raise ValueError(
            f"Eram esperados {NUM_CONTEXTOS_ESPERADO} "
            f"registros finais, mas foram obtidos "
            f"{len(registros)}."
        )

    # Verifica unicidade de (query_id, arquitetura).
    chaves = {
        (r["query_id"], r["arquitetura"])
        for r in registros
    }

    if len(chaves) != NUM_CONTEXTOS_ESPERADO:
        raise ValueError(
            "Há duplicatas em (query_id, arquitetura). "
            f"Chaves únicas: {len(chaves)}."
        )

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
                "num_documentos": r[
                    "num_documentos"
                ],
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
        f"\nConsultas: "
        f"{consultas['query_id'].nunique()}"
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
            consultas_unicas=("query_id", "nunique"),
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

    print("\nValidações finais:")
    print(
        f"- 150 consultas: "
        f"{'OK' if consultas['query_id'].nunique() == 150 else 'ERRO'}"
    )
    print(
        f"- 450 contextos únicos: "
        f"{'OK' if len(chaves) == 450 else 'ERRO'}"
    )
    print(
        f"- 10 documentos por contexto: "
        f"{'OK' if (df_saida['num_documentos'] == 10).all() else 'ERRO'}"
    )


if __name__ == "__main__":
    main()
