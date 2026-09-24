from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_RAG = Path(
    "outputs/rag/metricas_por_query.csv"
)

ARQUIVO_GRAPHRAG = Path(
    "outputs/graphrag/metricas_por_query.csv"
)

PASTA_SAIDA = Path(
    "outputs/comparacao"
)

ARQUIVO_COMPARACAO = (
    PASTA_SAIDA
    / "comparacao_rag_graphrag.csv"
)

ARQUIVO_RESUMO = (
    PASTA_SAIDA
    / "resumo_comparacao.csv"
)

ARQUIVO_SOURCE = (
    PASTA_SAIDA
    / "comparacao_por_source.csv"
)


METRICAS = [
    "p_at_10",
    "r_at_10",
    "mrr_at_10",
    "ndcg_at_10",
]


# ============================================================
# FUNÇÕES
# ============================================================

def classificar_delta(delta, tolerancia=1e-12):
    """
    Classifica a diferença GraphRAG - RAG.

    delta > 0  -> GraphRAG ganhou
    delta < 0  -> RAG ganhou
    delta == 0 -> empate
    """

    if delta > tolerancia:
        return "GraphRAG"

    if delta < -tolerancia:
        return "RAG"

    return "Empate"


def bootstrap_ic95(valores, n_bootstrap=10000, seed=42):
    """
    Intervalo de confiança bootstrap de 95%
    para a média das diferenças pareadas.
    """

    valores = np.asarray(
        valores,
        dtype=float
    )

    rng = np.random.default_rng(seed)

    medias = np.empty(
        n_bootstrap,
        dtype=float
    )

    n = len(valores)

    for i in range(n_bootstrap):
        amostra = rng.choice(
            valores,
            size=n,
            replace=True
        )

        medias[i] = amostra.mean()

    inferior = np.percentile(
        medias,
        2.5
    )

    superior = np.percentile(
        medias,
        97.5
    )

    return inferior, superior


def permutation_test_pareado(
    diferencas,
    n_permutacoes=10000,
    seed=42
):
    """
    Teste de permutação pareado.

    Hipótese nula:
    não existe diferença sistemática entre
    RAG e GraphRAG.

    Como os dados são pareados, a permutação
    equivale a inverter aleatoriamente o sinal
    das diferenças.
    """

    diferencas = np.asarray(
        diferencas,
        dtype=float
    )

    observado = abs(
        diferencas.mean()
    )

    rng = np.random.default_rng(seed)

    contador = 0

    for _ in range(n_permutacoes):

        sinais = rng.choice(
            [-1, 1],
            size=len(diferencas)
        )

        media_permutada = abs(
            np.mean(
                diferencas * sinais
            )
        )

        if media_permutada >= observado:
            contador += 1

    p_valor = (
        contador + 1
    ) / (
        n_permutacoes + 1
    )

    return p_valor


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 90)
print("COMPARAÇÃO PAREADA RAG vs GRAPHRAG")
print("=" * 90)

print("\nCarregando resultados...")

rag = pd.read_csv(ARQUIVO_RAG)
graphrag = pd.read_csv(ARQUIVO_GRAPHRAG)

print(f"Queries RAG: {len(rag)}")
print(f"Queries GraphRAG: {len(graphrag)}")

print("\nColunas encontradas:")
print("RAG:")
print(rag.columns.tolist())

print("\nGraphRAG:")
print(graphrag.columns.tolist())


# ============================================================
# NORMALIZAR NOMES DE COLUNAS
# ============================================================

def normalizar_colunas_metricas(df, nome_sistema):
    """
    Converte diferentes nomes usados nos scripts anteriores
    para um padrão único:

        query_id
        source
        pergunta
        p_at_10
        r_at_10
        mrr_at_10
        ndcg_at_10
    """

    aliases = {
        # ----------------------------------------------------
        # Query ID
        # ----------------------------------------------------
        "query_id": "query_id",
        "id": "query_id",
        "ID": "query_id",

        # ----------------------------------------------------
        # Texto da query
        # ----------------------------------------------------
        "pergunta": "pergunta",
        "query": "pergunta",
        "text": "pergunta",
        "TEXT": "pergunta",

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------
        "source": "source",
        "SOURCE": "source",

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------
        "p_at_10": "p_at_10",
        "precision_at_10": "p_at_10",
        "precision@10": "p_at_10",
        "P@10": "p_at_10",
        "precision": "p_at_10",

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------
        "r_at_10": "r_at_10",
        "recall_at_10": "r_at_10",
        "recall@10": "r_at_10",
        "R@10": "r_at_10",
        "recall": "r_at_10",

        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------
        "mrr_at_10": "mrr_at_10",
        "MRR@10": "mrr_at_10",
        "mrr@10": "mrr_at_10",
        "mrr": "mrr_at_10",

        # ----------------------------------------------------
        # nDCG
        # ----------------------------------------------------
        "ndcg_at_10": "ndcg_at_10",
        "nDCG@10": "ndcg_at_10",
        "ndcg@10": "ndcg_at_10",
        "ndcg": "ndcg_at_10",
    }

    renomear = {}

    for coluna in df.columns:
        if coluna in aliases:
            renomear[coluna] = aliases[coluna]

    df = df.rename(columns=renomear)

    obrigatorias = [
        "query_id",
        "p_at_10",
        "r_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]

    faltando = [
        coluna
        for coluna in obrigatorias
        if coluna not in df.columns
    ]

    if faltando:
        raise RuntimeError(
            f"\nO arquivo {nome_sistema} ainda não possui "
            f"as colunas necessárias:\n"
            f"{faltando}\n\n"
            f"Colunas disponíveis:\n"
            f"{df.columns.tolist()}"
        )

    return df


rag = normalizar_colunas_metricas(
    rag,
    "RAG"
)

graphrag = normalizar_colunas_metricas(
    graphrag,
    "GraphRAG"
)


# ============================================================
# VALIDAR IDs
# ============================================================

rag["query_id"] = rag["query_id"].astype(int)
graphrag["query_id"] = graphrag["query_id"].astype(int)

if rag["query_id"].duplicated().any():
    raise RuntimeError(
        "Existem query_id duplicados no arquivo RAG."
    )

if graphrag["query_id"].duplicated().any():
    raise RuntimeError(
        "Existem query_id duplicados no arquivo GraphRAG."
    )


# ============================================================
# PREPARAR METADADOS DA QUERY
# ============================================================

# Preferimos pegar source/pergunta do GraphRAG,
# porque esse arquivo foi produzido pelo script mais recente.

metadados = graphrag[
    [
        coluna
        for coluna in [
            "query_id",
            "source",
            "pergunta",
        ]
        if coluna in graphrag.columns
    ]
].copy()


# Caso algum campo não exista no GraphRAG,
# tenta obter do RAG.

if "source" not in metadados.columns and "source" in rag.columns:
    metadados = metadados.merge(
        rag[["query_id", "source"]],
        on="query_id",
        how="left"
    )

if "pergunta" not in metadados.columns and "pergunta" in rag.columns:
    metadados = metadados.merge(
        rag[["query_id", "pergunta"]],
        on="query_id",
        how="left"
    )


# ============================================================
# PREPARAR APENAS AS MÉTRICAS
# ============================================================

metricas_rag = rag[
    [
        "query_id",
        "p_at_10",
        "r_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]
].copy()

metricas_graph = graphrag[
    [
        "query_id",
        "p_at_10",
        "r_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

df = metricas_rag.merge(
    metricas_graph,
    on="query_id",
    how="inner",
    suffixes=(
        "_rag",
        "_graphrag"
    )
)

df = df.merge(
    metadados,
    on="query_id",
    how="left"
)

print(
    f"\nQueries comparadas: {len(df)}"
)

if len(df) != 150:
    print(
        "\nATENÇÃO: o número de queries "
        "comparadas não é 150."
    )


# ============================================================
# GARANTIR SOURCE/PERGUNTA
# ============================================================

if "source" not in df.columns:
    df["source"] = "desconhecido"

if "pergunta" not in df.columns:
    df["pergunta"] = ""

for metrica in METRICAS:

    coluna_rag = (
        f"{metrica}_rag"
    )

    coluna_graph = (
        f"{metrica}_graphrag"
    )

    coluna_delta = (
        f"delta_{metrica}"
    )

    coluna_vencedor = (
        f"vencedor_{metrica}"
    )

    df[coluna_delta] = (
        df[coluna_graph]
        -
        df[coluna_rag]
    )

    df[coluna_vencedor] = (
        df[coluna_delta]
        .apply(
            classificar_delta
        )
    )


# ============================================================
# RESUMO GLOBAL
# ============================================================

resumo = []

print("\n" + "=" * 90)
print("RESULTADOS GLOBAIS")
print("=" * 90)


for metrica in METRICAS:

    coluna_rag = (
        f"{metrica}_rag"
    )

    coluna_graph = (
        f"{metrica}_graphrag"
    )

    coluna_delta = (
        f"delta_{metrica}"
    )

    coluna_vencedor = (
        f"vencedor_{metrica}"
    )

    media_rag = (
        df[coluna_rag].mean()
    )

    media_graph = (
        df[coluna_graph].mean()
    )

    delta_medio = (
        df[coluna_delta].mean()
    )

    ganho_relativo = (
        delta_medio
        /
        media_rag
        * 100
        if media_rag != 0
        else np.nan
    )

    contagem = (
        df[coluna_vencedor]
        .value_counts()
    )

    graph_ganhou = int(
        contagem.get(
            "GraphRAG",
            0
        )
    )

    rag_ganhou = int(
        contagem.get(
            "RAG",
            0
        )
    )

    empates = int(
        contagem.get(
            "Empate",
            0
        )
    )

    ic_inf, ic_sup = (
        bootstrap_ic95(
            df[coluna_delta].values
        )
    )

    p_valor = (
        permutation_test_pareado(
            df[coluna_delta].values
        )
    )

    resumo.append(
        {
            "metrica": metrica,
            "media_rag": media_rag,
            "media_graphrag": media_graph,
            "delta_medio": delta_medio,
            "ganho_relativo_pct": ganho_relativo,
            "graphrag_ganhou": graph_ganhou,
            "rag_ganhou": rag_ganhou,
            "empates": empates,
            "ic95_inferior": ic_inf,
            "ic95_superior": ic_sup,
            "p_valor_permutacao": p_valor,
        }
    )

    print(
        f"\n{metrica}"
    )

    print(
        f"  RAG       : "
        f"{media_rag:.6f}"
    )

    print(
        f"  GraphRAG  : "
        f"{media_graph:.6f}"
    )

    print(
        f"  Delta     : "
        f"{delta_medio:+.6f}"
    )

    print(
        f"  Relativo  : "
        f"{ganho_relativo:+.2f}%"
    )

    print(
        f"  GraphRAG ganhou: "
        f"{graph_ganhou}"
    )

    print(
        f"  RAG ganhou     : "
        f"{rag_ganhou}"
    )

    print(
        f"  Empates        : "
        f"{empates}"
    )

    print(
        f"  IC95% delta    : "
        f"[{ic_inf:.6f}, "
        f"{ic_sup:.6f}]"
    )

    print(
        f"  p permutação   : "
        f"{p_valor:.6f}"
    )


df_resumo = pd.DataFrame(
    resumo
)


# ============================================================
# COMPARAÇÃO POR SOURCE
# ============================================================

resultados_source = []

for source, grupo in df.groupby(
    "source"
):

    for metrica in METRICAS:

        coluna_rag = (
            f"{metrica}_rag"
        )

        coluna_graph = (
            f"{metrica}_graphrag"
        )

        coluna_delta = (
            f"delta_{metrica}"
        )

        coluna_vencedor = (
            f"vencedor_{metrica}"
        )

        contagem = (
            grupo[coluna_vencedor]
            .value_counts()
        )

        resultados_source.append(
            {
                "source": source,
                "metrica": metrica,
                "queries": len(grupo),
                "media_rag": (
                    grupo[
                        coluna_rag
                    ].mean()
                ),
                "media_graphrag": (
                    grupo[
                        coluna_graph
                    ].mean()
                ),
                "delta_medio": (
                    grupo[
                        coluna_delta
                    ].mean()
                ),
                "graphrag_ganhou": int(
                    contagem.get(
                        "GraphRAG",
                        0
                    )
                ),
                "rag_ganhou": int(
                    contagem.get(
                        "RAG",
                        0
                    )
                ),
                "empates": int(
                    contagem.get(
                        "Empate",
                        0
                    )
                ),
            }
        )


df_source = pd.DataFrame(
    resultados_source
)


# ============================================================
# MELHORES E PIORES QUERIES
# ============================================================

print("\n" + "=" * 90)
print("MAIORES GANHOS DO GRAPHRAG EM nDCG@10")
print("=" * 90)

melhores = (
    df.sort_values(
        "delta_ndcg_at_10",
        ascending=False
    )
    .head(10)
)

print(
    melhores[
        [
            "query_id",
            "source",
            "pergunta",
            "ndcg_at_10_rag",
            "ndcg_at_10_graphrag",
            "delta_ndcg_at_10",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("MAIORES PERDAS DO GRAPHRAG EM nDCG@10")
print("=" * 90)

piores = (
    df.sort_values(
        "delta_ndcg_at_10",
        ascending=True
    )
    .head(10)
)

print(
    piores[
        [
            "query_id",
            "source",
            "pergunta",
            "ndcg_at_10_rag",
            "ndcg_at_10_graphrag",
            "delta_ndcg_at_10",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# QUANTAS QUERIES TIVERAM ZERO
# ============================================================

print("\n" + "=" * 90)
print("QUERIES SEM DOCUMENTO RELEVANTE NO TOP-10")
print("=" * 90)

zero_rag = (
    df[
        "p_at_10_rag"
    ] == 0
).sum()

zero_graph = (
    df[
        "p_at_10_graphrag"
    ] == 0
).sum()

print(
    f"RAG      : {zero_rag}"
)

print(
    f"GraphRAG : {zero_graph}"
)


# ============================================================
# SALVAR
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    ARQUIVO_COMPARACAO,
    index=False
)

df_resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)

df_source.to_csv(
    ARQUIVO_SOURCE,
    index=False
)


print("\n" + "=" * 90)
print("ARQUIVOS SALVOS")
print("=" * 90)

print(
    ARQUIVO_COMPARACAO
)

print(
    ARQUIVO_RESUMO
)

print(
    ARQUIVO_SOURCE
)

print(
    "\nComparação concluída."
)