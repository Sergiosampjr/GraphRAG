from pathlib import Path

import pandas as pd


INPUT_DIR = Path("outputs/rag")

METRICAS_FILE = INPUT_DIR / "metricas_por_query.csv"
RESULTADOS_FILE = INPUT_DIR / "resultados_top10.csv"

OUTPUT_DIR = Path("outputs/rag/analise")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Carregando resultados do RAG...")

    metricas = pd.read_csv(
        METRICAS_FILE
    )

    resultados = pd.read_csv(
        RESULTADOS_FILE
    )

    print(f"Queries: {len(metricas)}")
    print(f"Resultados top-10: {len(resultados)}")

    metricas_cols = [
        "precision_at_10",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10"
    ]

    # ============================================================
    # 1. ESTATÍSTICAS GLOBAIS
    # ============================================================

    print("\n" + "=" * 80)
    print("ESTATÍSTICAS GLOBAIS")
    print("=" * 80)

    estatisticas = (
        metricas[metricas_cols]
        .describe()
        .T
    )

    print(estatisticas)

    estatisticas.to_csv(
        OUTPUT_DIR / "estatisticas_globais.csv"
    )

    # ============================================================
    # 2. MÉTRICAS POR SOURCE
    # ============================================================

    print("\n" + "=" * 80)
    print("MÉTRICAS POR SOURCE")
    print("=" * 80)

    resumo_source = (
        metricas
        .groupby("source")[metricas_cols]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(resumo_source)

    resumo_source.to_csv(
        OUTPUT_DIR / "metricas_por_source.csv"
    )

    # ============================================================
    # 3. QUERIES SEM NENHUM ACERTO
    # ============================================================

    sem_acerto = metricas[
        metricas["recall_at_10"] == 0
    ].copy()

    print("\n" + "=" * 80)
    print("QUERIES SEM DOCUMENTO RELEVANTE NO TOP-10")
    print("=" * 80)

    print(
        f"Total: {len(sem_acerto)} "
        f"de {len(metricas)} "
        f"({len(sem_acerto) / len(metricas):.2%})"
    )

    if not sem_acerto.empty:

        print(
            sem_acerto[
                [
                    "query_id",
                    "query",
                    "source"
                ]
            ].to_string(index=False)
        )

    sem_acerto.to_csv(
        OUTPUT_DIR / "queries_sem_acerto.csv",
        index=False
    )

    # ============================================================
    # 4. QUERIES COM PELO MENOS UM ACERTO
    # ============================================================

    com_acerto = metricas[
        metricas["recall_at_10"] > 0
    ].copy()

    print("\n" + "=" * 80)
    print("QUERIES COM PELO MENOS UM DOCUMENTO RELEVANTE")
    print("=" * 80)

    print(
        f"Total: {len(com_acerto)} "
        f"de {len(metricas)} "
        f"({len(com_acerto) / len(metricas):.2%})"
    )

    # ============================================================
    # 5. PRIMEIRO DOCUMENTO RELEVANTE
    # ============================================================

    resultados["relevante"] = (
        resultados["qrel_score"]
        .fillna(0)
        .gt(0)
    )

    primeiros_relevantes = (
        resultados[
            resultados["relevante"]
        ]
        .sort_values(
            ["query_id", "rank"]
        )
        .groupby(
            "query_id",
            as_index=False
        )
        .first()
    )

    print("\n" + "=" * 80)
    print("POSIÇÃO DO PRIMEIRO DOCUMENTO RELEVANTE")
    print("=" * 80)

    if not primeiros_relevantes.empty:

        print(
            primeiros_relevantes["rank"]
            .describe()
        )

    primeiros_relevantes.to_csv(
        OUTPUT_DIR / "primeiro_relevante.csv",
        index=False
    )

    # ============================================================
    # 6. MELHORES QUERIES POR nDCG
    # ============================================================

    melhores = (
        metricas
        .sort_values(
            "ndcg_at_10",
            ascending=False
        )
        .head(15)
    )

    print("\n" + "=" * 80)
    print("15 MELHORES QUERIES POR nDCG@10")
    print("=" * 80)

    print(
        melhores[
            [
                "query_id",
                "query",
                "source",
                "precision_at_10",
                "recall_at_10",
                "mrr_at_10",
                "ndcg_at_10"
            ]
        ].to_string(index=False)
    )

    melhores.to_csv(
        OUTPUT_DIR / "melhores_queries.csv",
        index=False
    )

    # ============================================================
    # 7. PIORES QUERIES POR nDCG
    # ============================================================

    piores = (
        metricas
        .sort_values(
            [
                "ndcg_at_10",
                "mrr_at_10"
            ],
            ascending=True
        )
        .head(15)
    )

    print("\n" + "=" * 80)
    print("15 PIORES QUERIES POR nDCG@10")
    print("=" * 80)

    print(
        piores[
            [
                "query_id",
                "query",
                "source",
                "precision_at_10",
                "recall_at_10",
                "mrr_at_10",
                "ndcg_at_10"
            ]
        ].to_string(index=False)
    )

    piores.to_csv(
        OUTPUT_DIR / "piores_queries.csv",
        index=False
    )

    # ============================================================
    # 8. COBERTURA DOS QRELS
    # ============================================================

    print("\n" + "=" * 80)
    print("COBERTURA DOS QRELS")
    print("=" * 80)

    cobertura_global = (
        resultados["julgado"]
        .astype(bool)
        .mean()
    )

    print(
        f"Cobertura global: "
        f"{cobertura_global:.2%}"
    )

    cobertura_query = (
        resultados
        .groupby(
            [
                "query_id",
                "query",
                "source"
            ]
        )["julgado"]
        .mean()
        .reset_index(
            name="cobertura_qrel"
        )
    )

    cobertura_source = (
        cobertura_query
        .groupby("source")[
            "cobertura_qrel"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )

    print("\nCobertura por source:")
    print(cobertura_source)

    cobertura_query.to_csv(
        OUTPUT_DIR / "cobertura_qrel_por_query.csv",
        index=False
    )

    cobertura_source.to_csv(
        OUTPUT_DIR / "cobertura_qrel_por_source.csv"
    )

    # ============================================================
    # 9. DISTRIBUIÇÃO DOS SCORES QREL RECUPERADOS
    # ============================================================

    print("\n" + "=" * 80)
    print("DISTRIBUIÇÃO DOS SCORES QREL RECUPERADOS")
    print("=" * 80)

    julgados = resultados[
        resultados["qrel_score"].notna()
    ].copy()

    distribuicao_qrel = (
        julgados["qrel_score"]
        .value_counts()
        .sort_index()
    )

    print(distribuicao_qrel)

    distribuicao_qrel.to_csv(
        OUTPUT_DIR / "distribuicao_qrel_recuperados.csv",
        header=["quantidade"]
    )

    # ============================================================
    # 10. ACERTOS POR POSIÇÃO
    # ============================================================

    print("\n" + "=" * 80)
    print("DOCUMENTOS RELEVANTES POR POSIÇÃO DO RANKING")
    print("=" * 80)

    acertos_rank = (
        resultados
        .groupby("rank")["relevante"]
        .agg(
            [
                "sum",
                "mean"
            ]
        )
        .rename(
            columns={
                "sum": "quantidade_relevantes",
                "mean": "proporcao_relevantes"
            }
        )
    )

    print(acertos_rank)

    acertos_rank.to_csv(
        OUTPUT_DIR / "relevancia_por_rank.csv"
    )

    # ============================================================
    # 11. RESUMO FINAL
    # ============================================================

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)

    print(
        f"Queries avaliadas: {len(metricas)}"
    )

    print(
        f"Queries com >= 1 acerto: "
        f"{len(com_acerto)} "
        f"({len(com_acerto)/len(metricas):.2%})"
    )

    print(
        f"Queries sem acerto: "
        f"{len(sem_acerto)} "
        f"({len(sem_acerto)/len(metricas):.2%})"
    )

    print(
        f"Cobertura global dos qrels: "
        f"{cobertura_global:.2%}"
    )

    print("\nMédias globais:")

    for coluna in metricas_cols:

        print(
            f"{coluna:<20}: "
            f"{metricas[coluna].mean():.4f}"
        )

    print("\nArquivos gerados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()