from pathlib import Path

import pandas as pd


ARQUIVO = Path("data/processed/juristcu_documentos.parquet")


def main():
    print("Carregando corpus processado...")

    df = pd.read_parquet(ARQUIVO)

    # Contagem aproximada por palavras
    df["num_palavras"] = (
        df["text"]
        .fillna("")
        .str.split()
        .str.len()
    )

    print("\n" + "=" * 70)
    print("ESTATÍSTICAS DO CORPUS")
    print("=" * 70)

    print(f"Documentos: {len(df)}")
    print(f"Documentos vazios: {(df['num_palavras'] == 0).sum()}")

    print("\nNúmero de palavras por documento:")
    print(df["num_palavras"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    print("\n" + "=" * 70)
    print("DOCUMENTOS POR FAIXA")
    print("=" * 70)

    faixas = {
        "até 250 palavras": (0, 250),
        "251–500": (251, 500),
        "501–1000": (501, 1000),
        "1001–2000": (1001, 2000),
        "2001–5000": (2001, 5000),
        "mais de 5000": (5001, float("inf")),
    }

    for nome, (inicio, fim) in faixas.items():
        quantidade = (
            (df["num_palavras"] >= inicio)
            & (df["num_palavras"] <= fim)
        ).sum()

        percentual = quantidade / len(df) * 100

        print(
            f"{nome:<20}: "
            f"{quantidade:>6} "
            f"({percentual:6.2f}%)"
        )

    print("\n" + "=" * 70)
    print("5 MAIORES DOCUMENTOS")
    print("=" * 70)

    maiores = (
        df[["doc_id", "num_palavras", "tema", "subtema"]]
        .sort_values("num_palavras", ascending=False)
        .head(5)
    )

    print(maiores.to_string(index=False))


if __name__ == "__main__":
    main()