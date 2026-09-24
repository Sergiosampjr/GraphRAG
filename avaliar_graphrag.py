from pathlib import Path
import ast
import math
import pickle
import re
import time
import unicodedata
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"

ARQUIVO_FAISS = Path("data/rag/juristcu.faiss")
ARQUIVO_METADATA = Path("data/rag/metadata.parquet")
ARQUIVO_GRAFO = Path("data/graph/juristcu_graph.gpickle")

PASTA_SAIDA = Path("outputs/graphrag")

ARQUIVO_METRICAS = PASTA_SAIDA / "metricas_por_query.csv"
ARQUIVO_RESULTADOS = PASTA_SAIDA / "resultados_top10.csv"
ARQUIVO_RESUMO = PASTA_SAIDA / "resumo_metricas.csv"

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

K = 10

# Número de documentos semanticamente mais bem colocados
# usados como sementes do grafo.
TOP_DOCS_SEMENTE = 10

# Constante do Reciprocal Rank Fusion.
RRF_K = 60

# Relações utilizadas na expansão.
PESOS_RELACOES = {
    "TEM_AREA": 0.10,
    "TEM_TEMA": 0.35,
    "TEM_SUBTEMA": 0.45,
    "INDEXADO_POR": 0.50,
    "CITA": 0.35,
    "TEM_AUTOR_TESE": 0.10,
    "TEM_TIPO_PROCESSO": 0.10,
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar_doc_id(valor):
    """
    Converte diferentes representações de doc_id para int.
    """

    if valor is None:
        return None

    try:
        return int(valor)
    except (TypeError, ValueError):
        pass

    texto = str(valor).strip()

    numeros = re.findall(r"\d+", texto)

    if numeros:
        return int(numeros[-1])

    return None


def obter_tipo_relacao(dados_aresta):
    """
    Obtém o nome/tipo da relação armazenada na aresta.
    """

    for chave in ["relacao", "relation", "tipo", "type"]:
        if chave in dados_aresta:
            return str(dados_aresta[chave])

    return None


def eh_documento(grafo, no):
    """
    Verifica se um nó representa um documento.
    """

    dados = grafo.nodes[no]

    tipo = str(
        dados.get(
            "tipo",
            dados.get(
                "type",
                dados.get("node_type", "")
            )
        )
    ).lower()

    return tipo == "documento"


def extrair_doc_id_do_no(grafo, no):
    """
    Extrai doc_id de um nó Documento.
    """

    dados = grafo.nodes[no]

    for chave in ["doc_id", "id_documento", "document_id", "key"]:
        if chave in dados:
            doc_id = normalizar_doc_id(dados[chave])

            if doc_id is not None:
                return doc_id

    return normalizar_doc_id(no)


# ============================================================
# MÉTRICAS
# ============================================================

def precision_at_k(ranking, relevantes, k=10):
    """
    P@K:
    proporção dos K documentos recuperados que são relevantes.
    """

    ranking_k = ranking[:k]

    if not ranking_k:
        return 0.0

    acertos = sum(
        1
        for doc_id in ranking_k
        if doc_id in relevantes
    )

    return acertos / k


def recall_at_k(ranking, relevantes, k=10):
    """
    R@K:
    proporção dos documentos relevantes conhecidos
    que foram recuperados no Top-K.
    """

    if not relevantes:
        return 0.0

    ranking_k = ranking[:k]

    acertos = sum(
        1
        for doc_id in ranking_k
        if doc_id in relevantes
    )

    return acertos / len(relevantes)


def mrr_at_k(ranking, relevantes, k=10):
    """
    Reciprocal Rank do primeiro documento relevante.
    """

    for posicao, doc_id in enumerate(ranking[:k], start=1):

        if doc_id in relevantes:
            return 1.0 / posicao

    return 0.0


def dcg_at_k(ranking, mapa_relevancia, k=10):
    """
    DCG com ganho exponencial:

        gain = 2^relevancia - 1

    e desconto log2(posição + 1).
    """

    dcg = 0.0

    for posicao, doc_id in enumerate(ranking[:k], start=1):

        relevancia = mapa_relevancia.get(doc_id, 0)

        ganho = (2 ** relevancia) - 1

        desconto = math.log2(posicao + 1)

        dcg += ganho / desconto

    return dcg


def ndcg_at_k(ranking, mapa_relevancia, k=10):
    """
    nDCG@K usando relevância graduada 0-3.
    """

    dcg = dcg_at_k(
        ranking,
        mapa_relevancia,
        k
    )

    relevancias_ideais = sorted(
        mapa_relevancia.values(),
        reverse=True
    )[:k]

    if not relevancias_ideais:
        return 0.0

    idcg = 0.0

    for posicao, relevancia in enumerate(
        relevancias_ideais,
        start=1
    ):

        ganho = (2 ** relevancia) - 1

        desconto = math.log2(posicao + 1)

        idcg += ganho / desconto

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 90)
print("AVALIAÇÃO GRAPHRAG V3 - JurisTCU")
print("=" * 90)


# ------------------------------------------------------------
# FAISS
# ------------------------------------------------------------

print("\nCarregando índice FAISS...")

indice = faiss.read_index(str(ARQUIVO_FAISS))

print(f"Vetores FAISS: {indice.ntotal:,}")


# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------

print("\nCarregando metadata...")

metadata = pd.read_parquet(ARQUIVO_METADATA)

print(f"Linhas metadata: {len(metadata):,}")

if indice.ntotal != len(metadata):
    raise RuntimeError(
        "FAISS e metadata estão desalinhados: "
        f"{indice.ntotal} vetores vs {len(metadata)} linhas."
    )


# ------------------------------------------------------------
# GRAFO
# ------------------------------------------------------------

print("\nCarregando grafo...")

with open(ARQUIVO_GRAFO, "rb") as arquivo:
    grafo = pickle.load(arquivo)

print(f"Nós: {grafo.number_of_nodes():,}")
print(f"Arestas: {grafo.number_of_edges():,}")
print(f"Direcionado: {grafo.is_directed()}")


# ------------------------------------------------------------
# MODELO
# ------------------------------------------------------------

print("\nCarregando modelo E5...")

modelo = SentenceTransformer(MODELO_EMBEDDING)

print(f"Modelo: {MODELO_EMBEDDING}")


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

print("\nCarregando query.csv e qrel.csv...")

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

queries["ID"] = queries["ID"].astype(int)

qrels["QUERY_ID"] = qrels["QUERY_ID"].astype(int)
qrels["DOC_ID"] = qrels["DOC_ID"].astype(int)
qrels["SCORE"] = qrels["SCORE"].astype(int)

print(f"Queries: {len(queries):,}")
print(f"Qrels: {len(qrels):,}")


# ============================================================
# PREPARAR MAPA CHUNK -> DOCUMENTO
# ============================================================

print("\nPreparando doc_ids dos chunks...")

doc_ids_chunks = [
    normalizar_doc_id(valor)
    for valor in metadata["doc_id"].tolist()
]

if any(doc_id is None for doc_id in doc_ids_chunks):
    raise RuntimeError(
        "Foram encontrados doc_ids inválidos no metadata."
    )


# ============================================================
# MAPEAR DOC_ID -> NÓ DO GRAFO
# ============================================================

print("Mapeando documentos do grafo...")

doc_id_para_no = {}

for no in grafo.nodes:

    if not eh_documento(grafo, no):
        continue

    doc_id = extrair_doc_id_do_no(grafo, no)

    if doc_id is not None:
        doc_id_para_no[doc_id] = no

TOTAL_DOCUMENTOS = len(doc_id_para_no)

print(
    f"Documentos mapeados no grafo: "
    f"{TOTAL_DOCUMENTOS:,}"
)


# ============================================================
# CACHE DE FREQUÊNCIA DAS ENTIDADES
# ============================================================

print("\nCalculando frequências das entidades do grafo...")

frequencia_entidade = {}


def obter_frequencia_entidade(no_entidade):
    """
    Número de documentos conectados à entidade por relações
    permitidas.

    O resultado é armazenado em cache.
    """

    if no_entidade in frequencia_entidade:
        return frequencia_entidade[no_entidade]

    documentos = set()

    # Como o grafo é:
    #
    # Documento -> Entidade
    #
    # precisamos olhar os predecessores da entidade.

    for predecessor in grafo.predecessors(no_entidade):

        if not eh_documento(grafo, predecessor):
            continue

        dados_aresta = grafo.get_edge_data(
            predecessor,
            no_entidade
        )

        relacao = obter_tipo_relacao(dados_aresta)

        if relacao not in PESOS_RELACOES:
            continue

        doc_id = extrair_doc_id_do_no(
            grafo,
            predecessor
        )

        if doc_id is not None:
            documentos.add(doc_id)

    frequencia_entidade[no_entidade] = len(documentos)

    return len(documentos)


def calcular_idf(no_entidade):
    """
    Penaliza entidades excessivamente frequentes.

    IDF = log(1 + N / (1 + frequência))
    """

    frequencia = obter_frequencia_entidade(
        no_entidade
    )

    return math.log(
        1.0
        +
        TOTAL_DOCUMENTOS
        /
        (1.0 + frequencia)
    )


# ============================================================
# RANKING SEMÂNTICO
# ============================================================

def gerar_ranking_semantico(pergunta):
    """
    Retorna ranking semântico de todos os documentos.

    O FAISS contém chunks. Para cada documento, mantemos
    somente o maior score entre seus chunks.
    """

    texto_query = f"query: {pergunta}"

    embedding_query = modelo.encode(
        [texto_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    # V3:
    # busca todos os chunks para obter rank semântico real
    # dos documentos.
    scores, indices = indice.search(
        embedding_query,
        indice.ntotal
    )

    melhor_score_documento = {}

    for score, idx_chunk in zip(
        scores[0],
        indices[0]
    ):

        if idx_chunk < 0:
            continue

        doc_id = doc_ids_chunks[idx_chunk]

        score = float(score)

        score_anterior = melhor_score_documento.get(
            doc_id
        )

        if (
            score_anterior is None
            or score > score_anterior
        ):
            melhor_score_documento[doc_id] = score

    ranking = sorted(
        melhor_score_documento.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # doc_id -> posição semântica
    rank_semantico = {
        doc_id: posicao
        for posicao, (doc_id, _) in enumerate(
            ranking,
            start=1
        )
    }

    # doc_id -> score semântico
    score_semantico = dict(ranking)

    return (
        ranking,
        rank_semantico,
        score_semantico
    )


# ============================================================
# EXPANSÃO DO GRAFO
# ============================================================

def gerar_scores_grafo(sementes):
    """
    Expande as sementes através do grafo.

    Para cada:

        documento_semente -> entidade -> documento_candidato

    adicionamos:

        score_semente
        * peso_relação
        * IDF(entidade)

    ao score estrutural do candidato.
    """

    scores_grafo = defaultdict(float)

    for doc_semente, score_semente in sementes:

        no_semente = doc_id_para_no.get(
            doc_semente
        )

        if no_semente is None:
            continue

        # Documento -> Entidade
        for entidade in grafo.successors(
            no_semente
        ):

            dados_aresta_semente = (
                grafo.get_edge_data(
                    no_semente,
                    entidade
                )
            )

            relacao = obter_tipo_relacao(
                dados_aresta_semente
            )

            if relacao not in PESOS_RELACOES:
                continue

            peso_relacao = PESOS_RELACOES[
                relacao
            ]

            idf = calcular_idf(entidade)

            # Entidade <- outros Documentos
            for no_candidato in grafo.predecessors(
                entidade
            ):

                if not eh_documento(
                    grafo,
                    no_candidato
                ):
                    continue

                dados_aresta_candidato = (
                    grafo.get_edge_data(
                        no_candidato,
                        entidade
                    )
                )

                relacao_candidato = (
                    obter_tipo_relacao(
                        dados_aresta_candidato
                    )
                )

                # Só consideramos a mesma relação.
                if relacao_candidato != relacao:
                    continue

                doc_candidato = (
                    extrair_doc_id_do_no(
                        grafo,
                        no_candidato
                    )
                )

                if doc_candidato is None:
                    continue

                contribuicao = (
                    float(score_semente)
                    * peso_relacao
                    * idf
                )

                scores_grafo[
                    doc_candidato
                ] += contribuicao

    return dict(scores_grafo)


# ============================================================
# GRAPHRAG V3
# ============================================================

def recuperar_graphrag(pergunta):
    """
    Pipeline GraphRAG V3:

    pergunta
        ->
    ranking semântico completo
        ->
    Top-10 sementes
        ->
    expansão pelo grafo
        ->
    ranking estrutural
        ->
    RRF
        ->
    Top-10 final.
    """

    # --------------------------------------------------------
    # 1. Ranking semântico
    # --------------------------------------------------------

    (
        ranking_semantico,
        rank_semantico,
        score_semantico
    ) = gerar_ranking_semantico(pergunta)

    # --------------------------------------------------------
    # 2. Sementes
    # --------------------------------------------------------

    sementes = ranking_semantico[
        :TOP_DOCS_SEMENTE
    ]

    ids_sementes = {
        doc_id
        for doc_id, _ in sementes
    }

    # --------------------------------------------------------
    # 3. Grafo
    # --------------------------------------------------------

    scores_grafo = gerar_scores_grafo(
        sementes
    )

    ranking_grafo = sorted(
        scores_grafo.items(),
        key=lambda x: x[1],
        reverse=True
    )

    rank_grafo = {
        doc_id: posicao
        for posicao, (doc_id, _) in enumerate(
            ranking_grafo,
            start=1
        )
    }

    # --------------------------------------------------------
    # 4. Candidatos
    # --------------------------------------------------------

    candidatos = (
        set(scores_grafo.keys())
        |
        ids_sementes
    )

    # --------------------------------------------------------
    # 5. Reciprocal Rank Fusion
    # --------------------------------------------------------

    resultados = []

    for doc_id in candidatos:

        pos_sem = rank_semantico.get(doc_id)

        pos_grafo = rank_grafo.get(doc_id)

        rrf_sem = 0.0
        rrf_grafo = 0.0

        if pos_sem is not None:
            rrf_sem = 1.0 / (
                RRF_K + pos_sem
            )

        if pos_grafo is not None:
            rrf_grafo = 1.0 / (
                RRF_K + pos_grafo
            )

        score_final = (
            rrf_sem
            +
            rrf_grafo
        )

        resultados.append(
            {
                "doc_id": doc_id,
                "score_final": score_final,
                "rank_semantico": pos_sem,
                "score_semantico": (
                    score_semantico.get(doc_id)
                ),
                "rank_grafo": pos_grafo,
                "score_grafo": (
                    scores_grafo.get(
                        doc_id,
                        0.0
                    )
                ),
                "rrf_semantico": rrf_sem,
                "rrf_grafo": rrf_grafo,
                "era_semente": (
                    doc_id in ids_sementes
                ),
            }
        )

    resultados.sort(
        key=lambda x: x["score_final"],
        reverse=True
    )

    return resultados[:K]


# ============================================================
# PREPARAR QRELS
# ============================================================

print("\nPreparando qrels por query...")

qrels_por_query = {}

for query_id, grupo in qrels.groupby(
    "QUERY_ID"
):

    mapa = {}

    for _, linha in grupo.iterrows():

        doc_id = int(linha["DOC_ID"])
        score = int(linha["SCORE"])

        # Proteção caso exista mais de um julgamento
        # para o mesmo documento.
        if (
            doc_id not in mapa
            or score > mapa[doc_id]
        ):
            mapa[doc_id] = score

    qrels_por_query[int(query_id)] = mapa


# ============================================================
# AVALIAR AS 150 QUERIES
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

metricas_queries = []
resultados_top10 = []

tempo_inicio_total = time.perf_counter()

total_queries = len(queries)

print("\n" + "=" * 90)
print("INICIANDO AVALIAÇÃO")
print("=" * 90)


for numero, (_, linha_query) in enumerate(
    queries.iterrows(),
    start=1
):

    query_id = int(linha_query["ID"])
    pergunta = str(linha_query["TEXT"])
    source = str(linha_query["SOURCE"])

    inicio_query = time.perf_counter()

    # --------------------------------------------------------
    # GraphRAG
    # --------------------------------------------------------

    top10 = recuperar_graphrag(
        pergunta
    )

    latencia = (
        time.perf_counter()
        -
        inicio_query
    )

    ranking_docs = [
        resultado["doc_id"]
        for resultado in top10
    ]

    # --------------------------------------------------------
    # Qrels
    # --------------------------------------------------------

    mapa_relevancia = qrels_por_query.get(
        query_id,
        {}
    )

    # JurisTCU:
    # SCORE != 0 é relevante para métricas binárias.
    relevantes = {
        doc_id
        for doc_id, score
        in mapa_relevancia.items()
        if score != 0
    }

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    p10 = precision_at_k(
        ranking_docs,
        relevantes,
        K
    )

    r10 = recall_at_k(
        ranking_docs,
        relevantes,
        K
    )

    mrr10 = mrr_at_k(
        ranking_docs,
        relevantes,
        K
    )

    ndcg10 = ndcg_at_k(
        ranking_docs,
        mapa_relevancia,
        K
    )

    julgados_top10 = sum(
        1
        for doc_id in ranking_docs
        if doc_id in mapa_relevancia
    )

    relevantes_top10 = sum(
        1
        for doc_id in ranking_docs
        if doc_id in relevantes
    )

    judged_at_10 = (
        julgados_top10 / K
    )

    sementes_top10 = sum(
        1
        for resultado in top10
        if resultado["era_semente"]
    )

    novos_grafo_top10 = (
        K - sementes_top10
    )

    # --------------------------------------------------------
    # Salvar métricas da query
    # --------------------------------------------------------

    metricas_queries.append(
        {
            "query_id": query_id,
            "source": source,
            "pergunta": pergunta,
            "p_at_10": p10,
            "r_at_10": r10,
            "mrr_at_10": mrr10,
            "ndcg_at_10": ndcg10,
            "judged_at_10": judged_at_10,
            "julgados_top10": julgados_top10,
            "relevantes_top10": relevantes_top10,
            "sementes_top10": sementes_top10,
            "novos_grafo_top10": novos_grafo_top10,
            "latencia_segundos": latencia,
        }
    )

    # --------------------------------------------------------
    # Salvar cada posição do Top-10
    # --------------------------------------------------------

    for posicao, resultado in enumerate(
        top10,
        start=1
    ):

        doc_id = resultado["doc_id"]

        qrel_score = mapa_relevancia.get(
            doc_id
        )

        resultados_top10.append(
            {
                "query_id": query_id,
                "source": source,
                "pergunta": pergunta,
                "rank_final": posicao,
                "doc_id": doc_id,
                "qrel_score": qrel_score,
                "julgado": (
                    doc_id
                    in mapa_relevancia
                ),
                "relevante_binario": (
                    qrel_score is not None
                    and qrel_score != 0
                ),
                "score_final_rrf": (
                    resultado["score_final"]
                ),
                "rank_semantico": (
                    resultado["rank_semantico"]
                ),
                "score_semantico": (
                    resultado["score_semantico"]
                ),
                "rank_grafo": (
                    resultado["rank_grafo"]
                ),
                "score_grafo": (
                    resultado["score_grafo"]
                ),
                "rrf_semantico": (
                    resultado["rrf_semantico"]
                ),
                "rrf_grafo": (
                    resultado["rrf_grafo"]
                ),
                "era_semente": (
                    resultado["era_semente"]
                ),
            }
        )

    print(
        f"[{numero:03d}/{total_queries:03d}] "
        f"query_id={query_id:<4} | "
        f"P@10={p10:.4f} | "
        f"R@10={r10:.4f} | "
        f"MRR@10={mrr10:.4f} | "
        f"nDCG@10={ndcg10:.4f} | "
        f"novos={novos_grafo_top10} | "
        f"{latencia:.2f}s"
    )


# ============================================================
# DATAFRAMES
# ============================================================

df_metricas = pd.DataFrame(
    metricas_queries
)

df_resultados = pd.DataFrame(
    resultados_top10
)


# ============================================================
# MÉTRICAS GLOBAIS
# ============================================================

p10_global = df_metricas[
    "p_at_10"
].mean()

r10_global = df_metricas[
    "r_at_10"
].mean()

mrr10_global = df_metricas[
    "mrr_at_10"
].mean()

ndcg10_global = df_metricas[
    "ndcg_at_10"
].mean()

judged_global = df_metricas[
    "judged_at_10"
].mean()

latencia_media = df_metricas[
    "latencia_segundos"
].mean()

novos_media = df_metricas[
    "novos_grafo_top10"
].mean()


# ============================================================
# RESULTADOS POR SOURCE
# ============================================================

metricas_por_source = (
    df_metricas
    .groupby("source")
    .agg(
        queries=("query_id", "count"),
        p_at_10=("p_at_10", "mean"),
        r_at_10=("r_at_10", "mean"),
        mrr_at_10=("mrr_at_10", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        judged_at_10=("judged_at_10", "mean"),
        latencia_media=(
            "latencia_segundos",
            "mean"
        ),
        novos_grafo_media=(
            "novos_grafo_top10",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# SALVAR
# ============================================================

df_metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False
)

df_resultados.to_csv(
    ARQUIVO_RESULTADOS,
    index=False
)

resumo = pd.DataFrame(
    [
        {
            "arquitetura": "GraphRAG V3",
            "queries": len(df_metricas),
            "p_at_10": p10_global,
            "r_at_10": r10_global,
            "mrr_at_10": mrr10_global,
            "ndcg_at_10": ndcg10_global,
            "judged_at_10": judged_global,
            "latencia_media_segundos": (
                latencia_media
            ),
            "novos_grafo_top10_media": (
                novos_media
            ),
        }
    ]
)

resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)


# ============================================================
# TEMPO TOTAL
# ============================================================

tempo_total = (
    time.perf_counter()
    -
    tempo_inicio_total
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 90)
print("RESULTADOS GLOBAIS - GRAPHRAG V3")
print("=" * 90)

print(
    f"P@10       : {p10_global:.4f}"
)

print(
    f"R@10       : {r10_global:.4f}"
)

print(
    f"MRR@10     : {mrr10_global:.4f}"
)

print(
    f"nDCG@10    : {ndcg10_global:.4f}"
)

print(
    f"Judged@10  : {judged_global:.4f}"
)

print(
    f"\nLatência média/query: "
    f"{latencia_media:.4f} s"
)

print(
    f"Novos docs do grafo no Top-10, em média: "
    f"{novos_media:.2f}"
)

print(
    f"Tempo total: "
    f"{tempo_total:.2f} s"
)


print("\n" + "=" * 90)
print("RESULTADOS POR SOURCE")
print("=" * 90)

print(
    metricas_por_source.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("ARQUIVOS SALVOS")
print("=" * 90)

print(ARQUIVO_METRICAS)
print(ARQUIVO_RESULTADOS)
print(ARQUIVO_RESUMO)

print("\nAvaliação concluída.")