from pathlib import Path
import math
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_RAG = Path("outputs/rag/resultados_top10.csv")
ARQUIVO_GRAPHRAG = Path("outputs/graphrag/resultados_top10.csv")

PASTA_SAIDA = Path("outputs/hibrido")

ARQUIVO_TOP10 = PASTA_SAIDA / "resultados_top10.csv"
ARQUIVO_METRICAS = PASTA_SAIDA / "metricas_por_query.csv"
ARQUIVO_RESUMO = PASTA_SAIDA / "resumo_metricas.csv"

TOP_K = 10

# Constante clássica do Reciprocal Rank Fusion.
# Mantemos fixa para não ajustar o método usando os qrels.
RRF_K = 60


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def preparar_dataframe_rag(df):
    """
    Padroniza as colunas do RAG.
    """

    df = df.copy()

    obrigatorias = {
        "query_id",
        "query",
        "source",
        "rank",
        "doc_id",
        "qrel_score",
        "julgado",
    }

    faltando = obrigatorias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes no RAG: {sorted(faltando)}"
        )

    df = df.rename(
        columns={
            "query": "pergunta",
            "rank": "rank_rag",
        }
    )

    df["query_id"] = df["query_id"].astype(int)
    df["doc_id"] = df["doc_id"].astype(int)
    df["rank_rag"] = df["rank_rag"].astype(int)

    return df


def preparar_dataframe_graphrag(df):
    """
    Padroniza as colunas do GraphRAG.
    """

    df = df.copy()

    obrigatorias = {
        "query_id",
        "source",
        "pergunta",
        "rank_final",
        "doc_id",
        "qrel_score",
        "julgado",
    }

    faltando = obrigatorias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes no GraphRAG: {sorted(faltando)}"
        )

    df = df.rename(
        columns={
            "rank_final": "rank_graphrag",
        }
    )

    df["query_id"] = df["query_id"].astype(int)
    df["doc_id"] = df["doc_id"].astype(int)
    df["rank_graphrag"] = df["rank_graphrag"].astype(int)

    return df


def calcular_rrf(rank):
    """
    Calcula a contribuição RRF de um ranking.

    RRF(d) = 1 / (k + rank(d))
    """

    if pd.isna(rank):
        return 0.0

    return 1.0 / (RRF_K + int(rank))


def dcg(relevancias):
    """
    Calcula DCG usando ganho exponencial:

        gain = 2^rel - 1

    e desconto:

        log2(rank + 1)
    """

    valor = 0.0

    for i, rel in enumerate(relevancias, start=1):

        ganho = (2 ** rel) - 1

        desconto = math.log2(i + 1)

        valor += ganho / desconto

    return valor


def ndcg_at_k(relevancias_recuperadas, relevancias_ideais, k=10):
    """
    Calcula nDCG@k.
    """

    recuperadas = relevancias_recuperadas[:k]

    dcg_real = dcg(recuperadas)

    ideais = sorted(
        relevancias_ideais,
        reverse=True
    )[:k]

    dcg_ideal = dcg(ideais)

    if dcg_ideal == 0:
        return 0.0

    return dcg_real / dcg_ideal


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 90)
print("AVALIAÇÃO RAG + GRAPHRAG - FUSÃO RRF")
print("=" * 90)

inicio_total = time.perf_counter()

if not ARQUIVO_RAG.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_RAG}"
    )

if not ARQUIVO_GRAPHRAG.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_GRAPHRAG}"
    )


rag_original = pd.read_csv(ARQUIVO_RAG)
graph_original = pd.read_csv(ARQUIVO_GRAPHRAG)


print(f"\nRAG: {len(rag_original)} linhas")
print(f"GraphRAG: {len(graph_original)} linhas")

print("\nColunas RAG:")
print(rag_original.columns.tolist())

print("\nColunas GraphRAG:")
print(graph_original.columns.tolist())


rag = preparar_dataframe_rag(rag_original)
graph = preparar_dataframe_graphrag(graph_original)


# ============================================================
# VALIDAÇÕES
# ============================================================

queries_rag = set(rag["query_id"].unique())
queries_graph = set(graph["query_id"].unique())

if queries_rag != queries_graph:

    somente_rag = sorted(queries_rag - queries_graph)
    somente_graph = sorted(queries_graph - queries_rag)

    raise ValueError(
        "As queries do RAG e GraphRAG não são iguais.\n"
        f"Somente RAG: {somente_rag}\n"
        f"Somente GraphRAG: {somente_graph}"
    )


query_ids = sorted(queries_rag)

print(f"\nQueries encontradas: {len(query_ids)}")


# ============================================================
# CONSTRUÇÃO DOS QRELS A PARTIR DOS RESULTADOS EXISTENTES
# ============================================================

# IMPORTANTE:
#
# Para calcular recall e nDCG corretamente precisamos saber TODOS
# os documentos relevantes de cada query, e não apenas os que
# apareceram no Top-10.
#
# Os arquivos metricas_por_query.csv já possuem o número total
# de relevantes, mas não possuem todos os graus necessários para
# reconstruir o IDCG.
#
# Como o nDCG do híbrido precisa usar exatamente o mesmo conjunto
# de qrels utilizado nas avaliações anteriores, tentaremos carregar
# os qrels do dataset JurisTCU.

print("\nCarregando qrels do JurisTCU...")

try:
    from datasets import load_dataset

    BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

    qrels_dataset = load_dataset(
        "csv",
        data_files=BASE_URL + "qrel.csv",
        split="train",
    )

    qrels = qrels_dataset.to_pandas()

except Exception as erro:
    raise RuntimeError(
        "\nNão foi possível carregar qrel.csv do JurisTCU.\n"
        "O híbrido NÃO deve ser avaliado com apenas os qrels "
        "presentes no Top-10, pois isso distorceria Recall e nDCG.\n"
        f"\nErro original:\n{erro}"
    )


qrels = qrels.rename(
    columns={
        "QUERY_ID": "query_id",
        "DOC_ID": "doc_id",
        "SCORE": "qrel_score",
    }
)

qrels["query_id"] = qrels["query_id"].astype(int)
qrels["doc_id"] = qrels["doc_id"].astype(int)
qrels["qrel_score"] = qrels["qrel_score"].astype(int)


print(f"Qrels carregados: {len(qrels)}")


# ============================================================
# DICIONÁRIOS DE QRELS
# ============================================================

qrels_por_query = {}

for query_id, grupo in qrels.groupby("query_id"):

    mapa = {}

    for _, linha in grupo.iterrows():

        doc_id = int(linha["doc_id"])
        score = int(linha["qrel_score"])

        # Segurança caso haja duplicata para o mesmo documento.
        # Mantemos o maior julgamento.
        mapa[doc_id] = max(
            score,
            mapa.get(doc_id, 0)
        )

    qrels_por_query[int(query_id)] = mapa


# ============================================================
# FUSÃO RRF
# ============================================================

resultados_top10 = []
metricas_queries = []


print("\n" + "=" * 90)
print("PROCESSANDO QUERIES")
print("=" * 90)


for contador, query_id in enumerate(query_ids, start=1):

    inicio_query = time.perf_counter()

    rag_q = (
        rag[rag["query_id"] == query_id]
        .sort_values("rank_rag")
        .copy()
    )

    graph_q = (
        graph[graph["query_id"] == query_id]
        .sort_values("rank_graphrag")
        .copy()
    )


    # --------------------------------------------------------
    # Metadados da query
    # --------------------------------------------------------

    pergunta = rag_q.iloc[0]["pergunta"]
    source = rag_q.iloc[0]["source"]


    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    ranks_rag = {
        int(row["doc_id"]): int(row["rank_rag"])
        for _, row in rag_q.iterrows()
    }

    ranks_graph = {
        int(row["doc_id"]): int(row["rank_graphrag"])
        for _, row in graph_q.iterrows()
    }


    # União dos documentos recuperados pelos dois métodos
    candidatos = set(ranks_rag) | set(ranks_graph)


    linhas_fusao = []


    for doc_id in candidatos:

        rank_rag = ranks_rag.get(doc_id)
        rank_graph = ranks_graph.get(doc_id)

        rrf_rag = calcular_rrf(rank_rag)
        rrf_graph = calcular_rrf(rank_graph)

        score_rrf = rrf_rag + rrf_graph


        if rank_rag is not None and rank_graph is not None:
            origem = "ambos"

        elif rank_rag is not None:
            origem = "rag"

        else:
            origem = "graphrag"


        linhas_fusao.append(
            {
                "doc_id": doc_id,
                "rank_rag": rank_rag,
                "rank_graphrag": rank_graph,
                "rrf_rag": rrf_rag,
                "rrf_graphrag": rrf_graph,
                "score_rrf": score_rrf,
                "origem": origem,
            }
        )


    fusao = pd.DataFrame(linhas_fusao)


    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------
    #
    # Critério principal:
    #   score RRF decrescente
    #
    # Critérios de desempate determinísticos:
    #   melhor rank RAG
    #   melhor rank GraphRAG
    #   doc_id
    #
    # Isso evita resultados não determinísticos.
    # --------------------------------------------------------

    fusao["_rank_rag_sort"] = (
        fusao["rank_rag"]
        .fillna(10**9)
    )

    fusao["_rank_graph_sort"] = (
        fusao["rank_graphrag"]
        .fillna(10**9)
    )

    fusao = fusao.sort_values(
        by=[
            "score_rrf",
            "_rank_rag_sort",
            "_rank_graph_sort",
            "doc_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)


    top10 = fusao.head(TOP_K).copy()

    top10["rank_final"] = np.arange(
        1,
        len(top10) + 1
    )


    # --------------------------------------------------------
    # QRELS
    # --------------------------------------------------------

    mapa_qrels = qrels_por_query.get(
        query_id,
        {}
    )

    relevantes_qrel = {
        doc_id
        for doc_id, score in mapa_qrels.items()
        if score != 0
    }

    total_relevantes = len(relevantes_qrel)


    relevancias_top10 = []
    relevantes_recuperados = 0
    julgados_top10 = 0

    primeira_posicao_relevante = None


    for _, row in top10.iterrows():

        doc_id = int(row["doc_id"])

        julgado = doc_id in mapa_qrels

        score_qrel = (
            int(mapa_qrels[doc_id])
            if julgado
            else 0
        )

        relevante = score_qrel != 0


        if julgado:
            julgados_top10 += 1

        if relevante:

            relevantes_recuperados += 1

            if primeira_posicao_relevante is None:
                primeira_posicao_relevante = int(
                    row["rank_final"]
                )


        relevancias_top10.append(score_qrel)


        resultados_top10.append(
            {
                "query_id": query_id,
                "source": source,
                "pergunta": pergunta,
                "rank_final": int(row["rank_final"]),
                "doc_id": doc_id,

                "rank_rag": (
                    int(row["rank_rag"])
                    if pd.notna(row["rank_rag"])
                    else np.nan
                ),

                "rank_graphrag": (
                    int(row["rank_graphrag"])
                    if pd.notna(row["rank_graphrag"])
                    else np.nan
                ),

                "rrf_rag": float(row["rrf_rag"]),
                "rrf_graphrag": float(row["rrf_graphrag"]),
                "score_final_rrf": float(row["score_rrf"]),

                "origem": row["origem"],

                "qrel_score": (
                    score_qrel
                    if julgado
                    else np.nan
                ),

                "julgado": bool(julgado),
                "relevante_binario": bool(relevante),
            }
        )


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    precision = (
        relevantes_recuperados / TOP_K
    )

    recall = (
        relevantes_recuperados / total_relevantes
        if total_relevantes > 0
        else 0.0
    )

    mrr = (
        1.0 / primeira_posicao_relevante
        if primeira_posicao_relevante is not None
        else 0.0
    )


    relevancias_ideais = list(
        mapa_qrels.values()
    )

    ndcg = ndcg_at_k(
        relevancias_top10,
        relevancias_ideais,
        TOP_K,
    )


    judged_at_10 = (
        julgados_top10 / TOP_K
    )


    latencia_fusao = (
        time.perf_counter() - inicio_query
    )


    metricas_queries.append(
        {
            "query_id": query_id,
            "source": source,
            "pergunta": pergunta,

            "p_at_10": precision,
            "r_at_10": recall,
            "mrr_at_10": mrr,
            "ndcg_at_10": ndcg,
            "judged_at_10": judged_at_10,

            "julgados_top10": julgados_top10,
            "relevantes_top10": relevantes_recuperados,
            "num_relevantes_qrel": total_relevantes,

            # Esta é apenas a latência computacional
            # da fusão dos rankings já existentes.
            "latencia_fusao_segundos": latencia_fusao,
        }
    )


    if contador % 10 == 0 or contador == len(query_ids):

        print(
            f"[{contador:3d}/{len(query_ids)}] "
            f"query_id={query_id}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

df_top10 = pd.DataFrame(resultados_top10)

df_metricas = pd.DataFrame(metricas_queries)


# ============================================================
# RESUMO GLOBAL
# ============================================================

metricas_principais = [
    "p_at_10",
    "r_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "judged_at_10",
]


resumo_global = {
    metrica: df_metricas[metrica].mean()
    for metrica in metricas_principais
}


# ============================================================
# RESULTADOS POR SOURCE
# ============================================================

resumo_source = (
    df_metricas
    .groupby("source")[metricas_principais]
    .mean()
    .reset_index()
)


# ============================================================
# ORIGEM DOS DOCUMENTOS DO TOP-10
# ============================================================

origens = (
    df_top10["origem"]
    .value_counts()
    .to_dict()
)


# ============================================================
# EXIBIÇÃO
# ============================================================

print("\n" + "=" * 90)
print("RESULTADOS GLOBAIS - RAG + GRAPHRAG")
print("=" * 90)

print(f"\nP@10      : {resumo_global['p_at_10']:.4f}")
print(f"R@10      : {resumo_global['r_at_10']:.4f}")
print(f"MRR@10    : {resumo_global['mrr_at_10']:.4f}")
print(f"nDCG@10   : {resumo_global['ndcg_at_10']:.4f}")
print(f"Judged@10 : {resumo_global['judged_at_10']:.4f}")


print("\n" + "=" * 90)
print("RESULTADOS POR SOURCE")
print("=" * 90)

print(
    resumo_source.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("ORIGEM DOS DOCUMENTOS NO TOP-10 HÍBRIDO")
print("=" * 90)

for origem, quantidade in origens.items():

    percentual = (
        quantidade / len(df_top10) * 100
    )

    print(
        f"{origem:10s}: "
        f"{quantidade:4d} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# SALVAMENTO
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


df_top10.to_csv(
    ARQUIVO_TOP10,
    index=False
)

df_metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False
)


linhas_resumo = []

for metrica, valor in resumo_global.items():

    linhas_resumo.append(
        {
            "escopo": "global",
            "grupo": "todos",
            "metrica": metrica,
            "valor": valor,
        }
    )


for _, row in resumo_source.iterrows():

    for metrica in metricas_principais:

        linhas_resumo.append(
            {
                "escopo": "source",
                "grupo": row["source"],
                "metrica": metrica,
                "valor": row[metrica],
            }
        )


df_resumo = pd.DataFrame(
    linhas_resumo
)


df_resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)


tempo_total = (
    time.perf_counter() - inicio_total
)


print("\n" + "=" * 90)
print("ARQUIVOS SALVOS")
print("=" * 90)

print(ARQUIVO_TOP10)
print(ARQUIVO_METRICAS)
print(ARQUIVO_RESUMO)

print(
    f"\nTempo total da fusão/avaliação: "
    f"{tempo_total:.2f}s"
)

print("\nAvaliação híbrida concluída.")