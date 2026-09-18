from pathlib import Path

import pandas as pd


ARQUIVO = Path("data/processed/juristcu_documentos.parquet")


def analisar_coluna(df, coluna, top_n=15):
    print("\n" + "=" * 80)
    print(f"COLUNA: {coluna}")
    print("=" * 80)

    total = len(df)
    nulos = df[coluna].isna().sum()

    # considera também strings vazias como ausentes
    vazios = (
        df[coluna]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    unicos = df[coluna].nunique(dropna=True)

    print(f"Total de registros: {total}")
    print(f"Nulos: {nulos}")
    print(f"Strings vazias: {vazios}")
    print(f"Valores únicos: {unicos}")

    print("\nValores mais frequentes:")

    frequencias = (
        df[coluna]
        .value_counts(dropna=False)
        .head(top_n)
    )

    print(frequencias.to_string())

    print("\nExemplos de valores:")

    exemplos = (
        df[coluna]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(10)
        .tolist()
    )

    for i, valor in enumerate(exemplos, start=1):
        valor_resumido = valor[:300]
        print(f"{i:02d}. {valor_resumido}")


def main():

    print("Carregando documentos...")

    df = pd.read_parquet(ARQUIVO)

    print("\n" + "=" * 80)
    print("VISÃO GERAL")
    print("=" * 80)

    print(f"Documentos: {len(df)}")
    print(f"Colunas: {len(df.columns)}")

    print("\nLista de colunas:")

    for coluna in df.columns:
        print(f"- {coluna}")

    # ------------------------------------------------------------
    # Colunas candidatas ao grafo
    # ------------------------------------------------------------

    colunas_analisar = [
        "area",
        "tema",
        "subtema",
        "autor_tese",
        "funcao_autor",
        "tipo_processo",
        "tipo_recurso",
        "indexacao",
        "referencia_legal",
        "paradigmatico",
    ]

    for coluna in colunas_analisar:

        if coluna in df.columns:
            analisar_coluna(
                df,
                coluna
            )

        else:
            print(
                f"\nATENÇÃO: coluna '{coluna}' "
                f"não encontrada."
            )

    # ------------------------------------------------------------
    # Relação hierárquica AREA -> TEMA -> SUBTEMA
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RELAÇÃO AREA -> TEMA -> SUBTEMA")
    print("=" * 80)

    hierarquia = (
        df[
            [
                "area",
                "tema",
                "subtema"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "area",
                "tema",
                "subtema"
            ]
        )
    )

    print(
        f"Combinações únicas: "
        f"{len(hierarquia)}"
    )

    print("\nPrimeiras 30 combinações:")

    print(
        hierarquia
        .head(30)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Quantidade de temas por área
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEMAS POR ÁREA")
    print("=" * 80)

    temas_area = (
        df
        .groupby("area")["tema"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        temas_area
        .head(20)
        .to_string()
    )

    # ------------------------------------------------------------
    # Quantidade de subtemas por tema
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUBTEMAS POR TEMA")
    print("=" * 80)

    subtemas_tema = (
        df
        .groupby("tema")["subtema"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        subtemas_tema
        .head(20)
        .to_string()
    )

    # ------------------------------------------------------------
    # Documentos por área
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DOCUMENTOS POR ÁREA")
    print("=" * 80)

    docs_area = (
        df["area"]
        .value_counts()
        .head(20)
    )

    print(
        docs_area
        .to_string()
    )

    print("\nAnálise concluída.")


if __name__ == "__main__":
    main()