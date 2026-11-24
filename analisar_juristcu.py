
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTOS = (
    BASE_DIR / "data/processed/juristcu_documentos.parquet"
)

# Arquivos opcionais do benchmark.
# Ajuste os caminhos se estiverem em outra pasta.
QUERIES = BASE_DIR / "data/raw/query.csv"
QRELS = BASE_DIR / "data/raw/qrel.csv"

SAIDA = BASE_DIR / "outputs/eda_juristcu"
SAIDA.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def salvar_tabela(df, nome):
    caminho = SAIDA / nome
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"[SALVO] {caminho}")


def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(
        SAIDA / nome,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()
    print(f"[SALVO] {SAIDA / nome}")


def normalizar_colunas(df):
    def normalizar(nome):
        nome = unicodedata.normalize("NFKD", str(nome))
        nome = "".join(
            c for c in nome if not unicodedata.combining(c)
        )
        return re.sub(r"\s+", "_", nome.strip().lower())

    df = df.copy()
    df.columns = [normalizar(c) for c in df.columns]
    return df






def distribuicao_categoria(df, coluna, nome, top_n=15):
    if coluna not in df.columns:
        return

    serie = (
        df[coluna]
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )

    contagem = serie.value_counts().head(top_n)

    tabela = contagem.rename_axis(coluna).reset_index(
        name="quantidade"
    )

    salvar_tabela(tabela, f"{nome}.csv")

    plt.figure(figsize=(11, 6))
    sns.barplot(
        x=contagem.values,
        y=contagem.index,
        color="steelblue"
    )
    plt.title(f"Distribuição por {coluna}")
    plt.xlabel("Quantidade de documentos")
    plt.ylabel(coluna.replace("_", " ").title())
    salvar_grafico(f"{nome}.png")



# ============================================================
# DOWNLOAD DOS ARQUIVOS QUERY E QREL
# ============================================================

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

QUERIES.parent.mkdir(
    parents=True,
    exist_ok=True
)

if not QUERIES.exists():

    print("\nBaixando query.csv...")

    consultas = load_dataset(
        "csv",
        data_files=BASE_URL + "query.csv",
        split="train"
    )

    consultas.to_pandas().to_csv(
        QUERIES,
        index=False,
        encoding="utf-8"
    )

    print("query.csv salvo com sucesso!")

if not QRELS.exists():

    print("\nBaixando qrel.csv...")

    relevancias = load_dataset(
        "csv",
        data_files=BASE_URL + "qrel.csv",
        split="train"
    )

    relevancias.to_pandas().to_csv(
        QRELS,
        index=False,
        encoding="utf-8"
    )

    print("qrel.csv salvo com sucesso!")


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISE EXPLORATÓRIA — JURISTCU")
print("=" * 70)

if not DOCUMENTOS.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {DOCUMENTOS}"
    )

df = normalizar_colunas(pd.read_parquet(DOCUMENTOS))

print("\nDimensões do dataset:")
print(f"Documentos: {len(df):,}")
print(f"Colunas: {len(df.columns)}")

print("\nColunas disponíveis:")
print(df.columns.tolist())

print("\nTipos de dados:")
print(df.dtypes)


# ============================================================
# 2. QUALIDADE DOS DADOS
# ============================================================

print("\n[2] QUALIDADE DOS DADOS")

qualidade = pd.DataFrame({
    "coluna": df.columns,
    "nulos": df.isna().sum().values,
    "percentual_nulos": (
        df.isna().mean().values * 100
    ).round(2),
    "valores_unicos": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

salvar_tabela(qualidade, "qualidade_dados.csv")

print(qualidade.to_string(index=False))

print("\nLinhas completamente duplicadas:")
print(df.duplicated().sum())

if "doc_id" in df.columns:
    print("IDs duplicados:")
    print(df["doc_id"].duplicated().sum())


# ============================================================
# 3. TAMANHO DOS DOCUMENTOS
# ============================================================

print("\n[3] ESTATÍSTICAS DOS DOCUMENTOS")

if "text" not in df.columns:
    raise ValueError(
        "A coluna 'text' não foi encontrada."
    )

textos = df["text"].fillna("").astype(str)

df["num_palavras"] = textos.str.split().str.len()
df["num_caracteres"] = textos.str.len()

estatisticas = df[
    ["num_palavras", "num_caracteres"]
].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
)

print(estatisticas)

estatisticas.to_csv(
    SAIDA / "estatisticas_tamanho.csv",
    encoding="utf-8-sig"
)

print("\nDocumentos com texto vazio:")
print(textos.str.strip().eq("").sum())

plt.figure(figsize=(10, 5))
sns.histplot(
    df["num_palavras"],
    bins=60,
    color="steelblue"
)
plt.title("Distribuição do tamanho dos documentos")
plt.xlabel("Número de palavras")
plt.ylabel("Frequência")
salvar_grafico("distribuicao_palavras.png")

plt.figure(figsize=(10, 4))
sns.boxplot(
    x=df["num_palavras"],
    color="lightblue"
)
plt.title("Dispersão do tamanho dos documentos")
plt.xlabel("Número de palavras")
salvar_grafico("boxplot_palavras.png")


# ============================================================
# 4. DISTRIBUIÇÃO TEMPORAL
# ============================================================

print("\n[4] DISTRIBUIÇÃO TEMPORAL")

if "ano_acordao" in df.columns:

    anos = pd.to_numeric(
        df["ano_acordao"],
        errors="coerce"
    ).dropna()

    contagem_anos = (
        anos.astype(int)
        .value_counts()
        .sort_index()
    )

    salvar_tabela(
        contagem_anos.rename_axis("ano").reset_index(
            name="quantidade"
        ),
        "distribuicao_anos.csv"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        contagem_anos.index,
        contagem_anos.values,
        marker="o"
    )
    plt.title("Distribuição temporal dos acórdãos")
    plt.xlabel("Ano do acórdão")
    plt.ylabel("Quantidade de documentos")
    salvar_grafico("distribuicao_temporal.png")


# ============================================================
# 5. DISTRIBUIÇÕES JURÍDICAS
# ============================================================

print("\n[5] METADADOS JURÍDICOS")

colunas_categoricas = [
    "area",
    "tema",
    "subtema",
    "colegiado",
    "tipo_processo",
    "tipo_recurso",
    "autor_tese",
    "funcao_autor",
    "paradigmatico",
]

for coluna in colunas_categoricas:
    distribuicao_categoria(
        df,
        coluna,
        f"distribuicao_{coluna}"
    )


# ============================================================
# 6. REFERÊNCIAS LEGAIS
# ============================================================

print("\n[6] REFERÊNCIAS LEGAIS")

if "referencia_legal" in df.columns:

    referencias = (
        df["referencia_legal"]
        .fillna("")
        .astype(str)
    )

    possui_referencia = (
        referencias.str.strip().ne("")
    )

    resumo_referencias = pd.DataFrame({
        "categoria": [
            "Com referência legal",
            "Sem referência legal"
        ],
        "quantidade": [
            int(possui_referencia.sum()),
            int((~possui_referencia).sum())
        ]
    })

    salvar_tabela(
        resumo_referencias,
        "presenca_referencias_legais.csv"
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=resumo_referencias,
        x="categoria",
        y="quantidade",
        color="steelblue"
    )
    plt.title("Presença de referências legais")
    plt.xlabel("")
    plt.ylabel("Documentos")
    salvar_grafico("presenca_referencias_legais.png")


# ============================================================
# 7. CONSULTAS DO BENCHMARK (OPCIONAL)
# ============================================================

print("\n[7] CONSULTAS DO BENCHMARK")

if QUERIES.exists():

    queries = normalizar_colunas(
        pd.read_csv(QUERIES)
    )

    print("Consultas:", len(queries))
    print("Colunas:", queries.columns.tolist())

    coluna_texto = next(
        (
            c for c in [
                "text", "query", "question", "pergunta"
            ]
            if c in queries.columns
        ),
        None
    )

    if coluna_texto:

        queries["num_palavras"] = (
            queries[coluna_texto]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        print(
            queries["num_palavras"].describe()
        )

        salvar_tabela(
            queries[["num_palavras"]],
            "tamanho_consultas.csv"
        )

        plt.figure(figsize=(9, 5))
        sns.histplot(
            queries["num_palavras"],
            bins=20
        )
        plt.title("Distribuição do tamanho das consultas")
        plt.xlabel("Número de palavras")
        salvar_grafico("tamanho_consultas.png")

else:
    print(
        "query.csv não encontrado. "
        "Análise de consultas ignorada."
    )


# ============================================================
# 8. JULGAMENTOS DE RELEVÂNCIA (OPCIONAL)
# ============================================================

print("\n[8] JULGAMENTOS DE RELEVÂNCIA")

if QRELS.exists():

    qrels = normalizar_colunas(
        pd.read_csv(QRELS)
    )

    print("Julgamentos:", len(qrels))
    print("Colunas:", qrels.columns.tolist())

    coluna_relevancia = next(
        (
            c for c in [
                "relevance", "relevance_score",
                "score", "rel", "relevancia"
            ]
            if c in qrels.columns
        ),
        None
    )

    if coluna_relevancia:

        relevancia = pd.to_numeric(
            qrels[coluna_relevancia],
            errors="coerce"
        )

        contagem = (
            relevancia.value_counts()
            .sort_index()
        )

        salvar_tabela(
            contagem.rename_axis(
                "grau_relevancia"
            ).reset_index(name="quantidade"),
            "distribuicao_relevancia.csv"
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=contagem.index.astype(str),
            y=contagem.values,
            color="steelblue"
        )
        plt.title("Distribuição dos graus de relevância")
        plt.xlabel("Grau de relevância")
        plt.ylabel("Quantidade de julgamentos")
        salvar_grafico("distribuicao_relevancia.png")

    coluna_query_id = next(
        (
            c for c in ["query_id", "qid"]
            if c in qrels.columns
        ),
        None
    )

    if coluna_query_id:

        por_consulta = (
            qrels.groupby(coluna_query_id)
            .size()
            .rename("quantidade_julgamentos")
            .reset_index()
        )

        salvar_tabela(
            por_consulta,
            "julgamentos_por_consulta.csv"
        )

        print(
            por_consulta[
                "quantidade_julgamentos"
            ].describe()
        )

else:
    print(
        "qrel.csv não encontrado. "
        "Análise de relevância ignorada."
    )


# ============================================================
# 9. RESUMO FINAL
# ============================================================

resumo = pd.DataFrame([{
    "total_documentos": len(df),
    "total_colunas_originais": len(df.columns) - 2,
    "media_palavras": round(
        df["num_palavras"].mean(), 2
    ),
    "mediana_palavras": round(
        df["num_palavras"].median(), 2
    ),
    "max_palavras": int(
        df["num_palavras"].max()
    ),
    "documentos_vazios": int(
        textos.str.strip().eq("").sum()
    )
}])

salvar_tabela(
    resumo,
    "resumo_geral.csv"
)
print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA")
print(f"Resultados em: {SAIDA}")
print("=" * 70)


duplicados = df[
    df.duplicated(subset=["text"], keep=False)
].sort_values("text")

print("Registros com textos repetidos:", len(duplicados))

print(
    duplicados[
        ["doc_id", "num_acordao", "ano_acordao", "tema"]
    ].to_string(index=False)
)

colunas = [
    "doc_id",
    "num_acordao",
    "ano_acordao",
    "colegiado",
    "area",
    "tema",
    "subtema",
    "tipo_processo",
    "referencia_legal",
]

duplicados = df[
    df.duplicated(subset=["text"], keep=False)
].copy()

for texto, grupo in duplicados.groupby("text", sort=False):

    print("\n" + "=" * 70)
    print("GRUPO DE TEXTOS IDÊNTICOS")
    print("=" * 70)

    print(
        grupo[colunas].to_string(index=False)
    )