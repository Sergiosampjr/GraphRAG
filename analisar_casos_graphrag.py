from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_COMPARACAO = Path(
    "outputs/comparacao/comparacao_rag_graphrag.csv"
)

ARQUIVO_TOP10_RAG = Path(
    "outputs/rag/resultados_top10.csv"
)

ARQUIVO_TOP10_GRAPHRAG = Path(
    "outputs/graphrag/resultados_top10.csv"
)

PASTA_SAIDA = Path(
    "outputs/comparacao/casos"
)

N_CASOS = 10


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def detectar_coluna(df, candidatos, obrigatoria=True):
    """
    Retorna a primeira coluna existente dentre os candidatos.
    """

    for coluna in candidatos:
        if coluna in df.columns:
            return coluna

    if obrigatoria:
        raise RuntimeError(
            f"Nenhuma destas colunas foi encontrada: "
            f"{candidatos}\n"
            f"Disponíveis: {df.columns.tolist()}"
        )

    return None


def normalizar_top10(df, sistema):
    """
    Normaliza os diferentes formatos dos arquivos Top-10.
    """

    coluna_query = detectar_coluna(
        df,
        ["query_id", "ID", "id"]
    )

    coluna_doc = detectar_coluna(
        df,
        ["doc_id", "DOC_ID", "document_id"]
    )

    coluna_rank = detectar_coluna(
        df,
        [
            "rank_final",
            "rank",
            "posicao",
            "posição"
        ]
    )

    coluna_qrel = detectar_coluna(
        df,
        [
            "qrel_score",
            "score_qrel",
            "relevancia",
            "relevance"
        ],
        obrigatoria=False
    )

    resultado = pd.DataFrame()

    resultado["query_id"] = (
        df[coluna_query].astype(int)
    )

    resultado["doc_id"] = (
        df[coluna_doc].astype(int)
    )

    resultado["rank"] = (
        df[coluna_rank].astype(int)
    )

    if coluna_qrel is not None:
        resultado["qrel_score"] = (
            pd.to_numeric(
                df[coluna_qrel],
                errors="coerce"
            )
        )
    else:
        resultado["qrel_score"] = pd.NA

    resultado["sistema"] = sistema

    return resultado


def mostrar_ranking(df, query_id, nome):
    """
    Imprime Top-10 de uma determinada query.
    """

    ranking = (
        df[
            df["query_id"] == query_id
        ]
        .sort_values("rank")
        .head(10)
    )

    print(f"\n{nome}")
    print("-" * 75)

    for _, linha in ranking.iterrows():

        qrel = linha["qrel_score"]

        if pd.isna(qrel):
            julgamento = "NÃO JULGADO"
        else:
            julgamento = f"SCORE={int(qrel)}"

        print(
            f"{int(linha['rank']):2d}. "
            f"doc_id={int(linha['doc_id']):6d} | "
            f"{julgamento}"
        )


def comparar_documentos(
    query_id,
    rag,
    graph
):
    """
    Identifica documentos mantidos, removidos e promovidos
    pelo GraphRAG.
    """

    rag_q = (
        rag[
            rag["query_id"] == query_id
        ]
        .sort_values("rank")
        .head(10)
    )

    graph_q = (
        graph[
            graph["query_id"] == query_id
        ]
        .sort_values("rank")
        .head(10)
    )

    docs_rag = set(
        rag_q["doc_id"].tolist()
    )

    docs_graph = set(
        graph_q["doc_id"].tolist()
    )

    removidos = docs_rag - docs_graph
    novos = docs_graph - docs_rag
    mantidos = docs_rag & docs_graph

    return (
        mantidos,
        removidos,
        novos
    )


# ============================================================
# CARREGAR ARQUIVOS
# ============================================================

print("=" * 90)
print("ANÁLISE DE CASOS - RAG vs GRAPHRAG")
print("=" * 90)

comparacao = pd.read_csv(
    ARQUIVO_COMPARACAO
)

rag_original = pd.read_csv(
    ARQUIVO_TOP10_RAG
)

graph_original = pd.read_csv(
    ARQUIVO_TOP10_GRAPHRAG
)

print(
    f"\nComparação: {len(comparacao)} queries"
)

print(
    f"Resultados RAG: {len(rag_original)} linhas"
)

print(
    f"Resultados GraphRAG: {len(graph_original)} linhas"
)

print("\nColunas RAG:")
print(rag_original.columns.tolist())

print("\nColunas GraphRAG:")
print(graph_original.columns.tolist())


# ============================================================
# NORMALIZAR
# ============================================================

rag = normalizar_top10(
    rag_original,
    "RAG"
)

graph = normalizar_top10(
    graph_original,
    "GraphRAG"
)


# ============================================================
# IDENTIFICAR CASOS
# ============================================================

melhores = (
    comparacao
    .sort_values(
        "delta_ndcg_at_10",
        ascending=False
    )
    .head(N_CASOS)
)

piores = (
    comparacao
    .sort_values(
        "delta_ndcg_at_10",
        ascending=True
    )
    .head(N_CASOS)
)


# ============================================================
# ANALISAR CASOS
# ============================================================

linhas_saida = []


def analisar_grupo(casos, categoria):

    print("\n")
    print("=" * 90)
    print(categoria.upper())
    print("=" * 90)

    for _, caso in casos.iterrows():

        query_id = int(
            caso["query_id"]
        )

        pergunta = str(
            caso.get(
                "pergunta",
                ""
            )
        )

        source = str(
            caso.get(
                "source",
                ""
            )
        )

        ndcg_rag = float(
            caso["ndcg_at_10_rag"]
        )

        ndcg_graph = float(
            caso["ndcg_at_10_graphrag"]
        )

        delta = float(
            caso["delta_ndcg_at_10"]
        )

        print("\n" + "#" * 90)

        print(
            f"QUERY {query_id}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Pergunta: {pergunta}"
        )

        print(
            f"nDCG RAG      : {ndcg_rag:.6f}"
        )

        print(
            f"nDCG GraphRAG : {ndcg_graph:.6f}"
        )

        print(
            f"Delta          : {delta:+.6f}"
        )

        mostrar_ranking(
            rag,
            query_id,
            "TOP-10 RAG"
        )

        mostrar_ranking(
            graph,
            query_id,
            "TOP-10 GRAPHRAG"
        )

        (
            mantidos,
            removidos,
            novos
        ) = comparar_documentos(
            query_id,
            rag,
            graph
        )

        print("\nALTERAÇÕES NO TOP-10")
        print("-" * 75)

        print(
            "Mantidos:",
            sorted(mantidos)
        )

        print(
            "Removidos pelo GraphRAG:",
            sorted(removidos)
        )

        print(
            "Novos promovidos pelo grafo:",
            sorted(novos)
        )

        # ----------------------------------------------------
        # Informações detalhadas dos documentos novos
        # ----------------------------------------------------

        graph_q = graph[
            graph["query_id"] == query_id
        ]

        rag_q = rag[
            rag["query_id"] == query_id
        ]

        for doc_id in sorted(novos):

            linha_doc = graph_q[
                graph_q["doc_id"] == doc_id
            ]

            if len(linha_doc) > 0:
                qrel = (
                    linha_doc.iloc[0][
                        "qrel_score"
                    ]
                )
            else:
                qrel = pd.NA

            if pd.isna(qrel):
                qrel_texto = "NÃO JULGADO"
            else:
                qrel_texto = (
                    f"SCORE={int(qrel)}"
                )

            print(
                f"  + {doc_id}: "
                f"{qrel_texto}"
            )

        # ----------------------------------------------------
        # Documentos removidos
        # ----------------------------------------------------

        for doc_id in sorted(removidos):

            linha_doc = rag_q[
                rag_q["doc_id"] == doc_id
            ]

            if len(linha_doc) > 0:
                qrel = (
                    linha_doc.iloc[0][
                        "qrel_score"
                    ]
                )
            else:
                qrel = pd.NA

            if pd.isna(qrel):
                qrel_texto = "NÃO JULGADO"
            else:
                qrel_texto = (
                    f"SCORE={int(qrel)}"
                )

            print(
                f"  - {doc_id}: "
                f"{qrel_texto}"
            )

        # ----------------------------------------------------
        # Salvar resumo
        # ----------------------------------------------------

        linhas_saida.append(
            {
                "categoria": categoria,
                "query_id": query_id,
                "source": source,
                "pergunta": pergunta,
                "ndcg_rag": ndcg_rag,
                "ndcg_graphrag": ndcg_graph,
                "delta_ndcg": delta,
                "num_mantidos": len(mantidos),
                "num_removidos": len(removidos),
                "num_novos": len(novos),
                "docs_removidos": (
                    ",".join(
                        map(
                            str,
                            sorted(removidos)
                        )
                    )
                ),
                "docs_novos": (
                    ",".join(
                        map(
                            str,
                            sorted(novos)
                        )
                    )
                ),
            }
        )


# ============================================================
# EXECUTAR ANÁLISE
# ============================================================

analisar_grupo(
    melhores,
    "MAIORES GANHOS"
)

analisar_grupo(
    piores,
    "MAIORES PERDAS"
)


# ============================================================
# ANÁLISE ESPECÍFICA DOS DADOS GRAPHRAG
# ============================================================

print("\n")
print("=" * 90)
print("DIAGNÓSTICO DOS DOCUMENTOS PROMOVIDOS PELO GRAFO")
print("=" * 90)

# O arquivo GraphRAG contém informações extras:
#
# rank_semantico
# rank_grafo
# score_grafo
# era_semente
#
# Vamos aproveitá-las.

colunas_diagnostico = [
    "query_id",
    "doc_id",
    "rank_final",
    "qrel_score",
    "rank_semantico",
    "score_semantico",
    "rank_grafo",
    "score_grafo",
    "rrf_semantico",
    "rrf_grafo",
    "era_semente",
]

colunas_existentes = [
    coluna
    for coluna in colunas_diagnostico
    if coluna in graph_original.columns
]

diagnostico = graph_original[
    colunas_existentes
].copy()

if "era_semente" in diagnostico.columns:

    promovidos = diagnostico[
        diagnostico["era_semente"] == False
    ].copy()

    print(
        f"\nDocumentos não-semente "
        f"promovidos ao Top-10: "
        f"{len(promovidos)}"
    )

    if "qrel_score" in promovidos.columns:

        julgados = promovidos[
            "qrel_score"
        ].notna()

        relevantes = (
            promovidos[
                "qrel_score"
            ].fillna(0) > 0
        )

        print(
            f"Promovidos julgados: "
            f"{julgados.sum()}"
        )

        print(
            f"Promovidos relevantes conhecidos: "
            f"{relevantes.sum()}"
        )

        if julgados.sum() > 0:

            print(
                "Precisão entre promovidos julgados: "
                f"{relevantes.sum() / julgados.sum():.4f}"
            )


# ============================================================
# SALVAR
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

df_saida = pd.DataFrame(
    linhas_saida
)

arquivo_saida = (
    PASTA_SAIDA
    / "casos_extremos.csv"
)

df_saida.to_csv(
    arquivo_saida,
    index=False
)

if (
    "era_semente"
    in diagnostico.columns
):

    arquivo_promovidos = (
        PASTA_SAIDA
        / "documentos_promovidos_grafo.csv"
    )

    promovidos.to_csv(
        arquivo_promovidos,
        index=False
    )


print("\n" + "=" * 90)
print("ARQUIVOS SALVOS")
print("=" * 90)

print(
    arquivo_saida
)

if (
    "era_semente"
    in diagnostico.columns
):
    print(
        arquivo_promovidos
    )

print(
    "\nAnálise de casos concluída."
)