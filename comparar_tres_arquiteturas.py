from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVOS = {
    "RAG": Path("outputs/rag/metricas_por_query.csv"),
    "GraphRAG": Path("outputs/graphrag/metricas_por_query.csv"),
    "Hibrido": Path("outputs/hibrido/metricas_por_query.csv"),
}

PASTA_SAIDA = Path("outputs/comparacao_tres")

ARQUIVO_POR_QUERY = PASTA_SAIDA / "comparacao_por_query.csv"
ARQUIVO_GLOBAL = PASTA_SAIDA / "resumo_comparacoes.csv"
ARQUIVO_SOURCE = PASTA_SAIDA / "comparacao_por_source.csv"

N_BOOTSTRAP = 10_000
N_PERMUTACOES = 10_000
SEED = 42


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def carregar_metricas(nome, caminho):
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado para {nome}: {caminho}"
        )

    df = pd.read_csv(caminho)

    aliases = {
        "precision_at_10": "p_at_10",
        "recall_at_10": "r_at_10",
        "mrr_at_10": "mrr_at_10",
        "ndcg_at_10": "ndcg_at_10",
        "query": "pergunta",
    }

    df = df.rename(columns=aliases)

    obrigatorias = {
        "query_id",
        "source",
        "p_at_10",
        "r_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    }

    faltando = obrigatorias - set(df.columns)

    if faltando:
        raise ValueError(
            f"{nome}: colunas ausentes: {sorted(faltando)}"
        )

    df["query_id"] = df["query_id"].astype(int)

    if df["query_id"].duplicated().any():
        duplicadas = (
            df.loc[df["query_id"].duplicated(), "query_id"]
            .tolist()
        )

        raise ValueError(
            f"{nome}: query_id duplicados: {duplicadas}"
        )

    return df


# ============================================================
# BOOTSTRAP PAREADO
# ============================================================

def bootstrap_ic(delta, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    IC bootstrap de 95% para a média das diferenças pareadas.
    """

    delta = np.asarray(delta, dtype=float)

    rng = np.random.default_rng(seed)

    n = len(delta)

    medias = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n
        )

        medias[i] = delta[indices].mean()

    inferior = np.percentile(medias, 2.5)
    superior = np.percentile(medias, 97.5)

    return inferior, superior


# ============================================================
# TESTE DE PERMUTAÇÃO PAREADO
# ============================================================

def teste_permutacao_pareado(
    delta,
    n_permutacoes=N_PERMUTACOES,
    seed=SEED,
):
    """
    Teste bilateral por inversão aleatória do sinal
    das diferenças pareadas.

    H0: diferença média = 0.
    """

    delta = np.asarray(delta, dtype=float)

    rng = np.random.default_rng(seed)

    observado = abs(delta.mean())

    extremos = 0

    for _ in range(n_permutacoes):
        sinais = rng.choice(
            [-1.0, 1.0],
            size=len(delta)
        )

        permutado = abs(
            np.mean(delta * sinais)
        )

        if permutado >= observado:
            extremos += 1

    # Correção +1 evita p = 0.
    p_valor = (
        (extremos + 1)
        /
        (n_permutacoes + 1)
    )

    return p_valor


# ============================================================
# COMPARAÇÃO DE DUAS ARQUITETURAS
# ============================================================

def comparar_par(
    df_a,
    df_b,
    nome_a,
    nome_b,
):
    metricas = [
        "p_at_10",
        "r_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]

    colunas_a = [
        "query_id",
        "source",
        *metricas,
    ]

    colunas_b = [
        "query_id",
        *metricas,
    ]

    a = df_a[colunas_a].copy()
    b = df_b[colunas_b].copy()

    a = a.rename(
        columns={
            m: f"{m}_{nome_a}"
            for m in metricas
        }
    )

    b = b.rename(
        columns={
            m: f"{m}_{nome_b}"
            for m in metricas
        }
    )

    combinado = a.merge(
        b,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    if len(combinado) != len(df_a) or len(combinado) != len(df_b):
        raise ValueError(
            f"{nome_a} × {nome_b}: "
            "quantidade de queries incompatível."
        )

    resumo = []

    for metrica in metricas:
        coluna_a = f"{metrica}_{nome_a}"
        coluna_b = f"{metrica}_{nome_b}"

        valores_a = combinado[coluna_a].to_numpy(dtype=float)
        valores_b = combinado[coluna_b].to_numpy(dtype=float)

        # Delta sempre B - A.
        delta = valores_b - valores_a

        media_a = valores_a.mean()
        media_b = valores_b.mean()
        delta_medio = delta.mean()

        if media_a != 0:
            mudanca_relativa = (
                delta_medio / media_a
            ) * 100
        else:
            mudanca_relativa = np.nan

        vitorias_b = int(
            np.sum(delta > 1e-12)
        )

        vitorias_a = int(
            np.sum(delta < -1e-12)
        )

        empates = int(
            np.sum(np.abs(delta) <= 1e-12)
        )

        ci_inf, ci_sup = bootstrap_ic(delta)

        p_valor = teste_permutacao_pareado(delta)

        resumo.append(
            {
                "comparacao": f"{nome_a} -> {nome_b}",
                "arquitetura_a": nome_a,
                "arquitetura_b": nome_b,
                "metrica": metrica,

                "media_a": media_a,
                "media_b": media_b,

                "delta_b_menos_a": delta_medio,
                "mudanca_relativa_pct": mudanca_relativa,

                "vitorias_a": vitorias_a,
                "vitorias_b": vitorias_b,
                "empates": empates,

                "ic95_inferior": ci_inf,
                "ic95_superior": ci_sup,

                "p_permutacao": p_valor,
            }
        )

        combinado[
            f"delta_{metrica}"
        ] = delta

    combinado["comparacao"] = (
        f"{nome_a} -> {nome_b}"
    )

    return combinado, pd.DataFrame(resumo)


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 100)
print("COMPARAÇÃO ESTATÍSTICA DAS TRÊS ARQUITETURAS")
print("=" * 100)

dados = {}

for nome, caminho in ARQUIVOS.items():
    dados[nome] = carregar_metricas(
        nome,
        caminho
    )

    print(
        f"{nome:10s}: "
        f"{len(dados[nome])} queries"
    )


# ============================================================
# VALIDAÇÃO DOS IDs
# ============================================================

ids_referencia = set(
    dados["RAG"]["query_id"]
)

for nome, df in dados.items():
    ids = set(df["query_id"])

    if ids != ids_referencia:
        raise ValueError(
            f"As queries de {nome} não coincidem "
            "com as queries do RAG."
        )


print(
    f"\nQueries comuns: "
    f"{len(ids_referencia)}"
)


# ============================================================
# COMPARAÇÕES
# ============================================================

pares = [
    ("RAG", "GraphRAG"),
    ("RAG", "Hibrido"),
    ("GraphRAG", "Hibrido"),
]

todos_por_query = []
todos_resumos = []


for nome_a, nome_b in pares:
    print(
        f"\nComparando {nome_a} × {nome_b}..."
    )

    por_query, resumo = comparar_par(
        dados[nome_a],
        dados[nome_b],
        nome_a,
        nome_b,
    )

    todos_por_query.append(por_query)
    todos_resumos.append(resumo)


df_por_query = pd.concat(
    todos_por_query,
    ignore_index=True
)

df_resumo = pd.concat(
    todos_resumos,
    ignore_index=True
)


# ============================================================
# COMPARAÇÃO POR SOURCE
# ============================================================

metricas = [
    "p_at_10",
    "r_at_10",
    "mrr_at_10",
    "ndcg_at_10",
]

linhas_source = []


for nome_a, nome_b in pares:
    df_a = dados[nome_a]
    df_b = dados[nome_b]

    combinado = df_a[
        ["query_id", "source", *metricas]
    ].merge(
        df_b[
            ["query_id", *metricas]
        ],
        on="query_id",
        suffixes=(
            f"_{nome_a}",
            f"_{nome_b}",
        ),
        validate="one_to_one",
    )

    for source, grupo in combinado.groupby("source"):

        for metrica in metricas:
            coluna_a = f"{metrica}_{nome_a}"
            coluna_b = f"{metrica}_{nome_b}"

            media_a = grupo[coluna_a].mean()
            media_b = grupo[coluna_b].mean()

            delta = media_b - media_a

            linhas_source.append(
                {
                    "comparacao": f"{nome_a} -> {nome_b}",
                    "source": source,
                    "metrica": metrica,
                    "media_a": media_a,
                    "media_b": media_b,
                    "delta_b_menos_a": delta,
                }
            )


df_source = pd.DataFrame(
    linhas_source
)


# ============================================================
# EXIBIÇÃO
# ============================================================

print("\n" + "=" * 100)
print("RESULTADOS GLOBAIS")
print("=" * 100)


for comparacao in df_resumo["comparacao"].unique():

    print(
        f"\n{'-' * 100}"
    )

    print(comparacao)

    print(
        f"{'-' * 100}"
    )

    bloco = df_resumo[
        df_resumo["comparacao"] == comparacao
    ]

    for _, row in bloco.iterrows():

        print(
            f"\n{row['metrica']}"
        )

        print(
            f"  {row['arquitetura_a']}: "
            f"{row['media_a']:.6f}"
        )

        print(
            f"  {row['arquitetura_b']}: "
            f"{row['media_b']:.6f}"
        )

        print(
            f"  Delta ({row['arquitetura_b']} - "
            f"{row['arquitetura_a']}): "
            f"{row['delta_b_menos_a']:+.6f}"
        )

        print(
            f"  Mudança relativa: "
            f"{row['mudanca_relativa_pct']:+.2f}%"
        )

        print(
            f"  Vitórias {row['arquitetura_a']}: "
            f"{int(row['vitorias_a'])}"
        )

        print(
            f"  Vitórias {row['arquitetura_b']}: "
            f"{int(row['vitorias_b'])}"
        )

        print(
            f"  Empates: "
            f"{int(row['empates'])}"
        )

        print(
            "  IC95% delta: "
            f"[{row['ic95_inferior']:+.6f}, "
            f"{row['ic95_superior']:+.6f}]"
        )

        print(
            f"  p (permutação): "
            f"{row['p_permutacao']:.6f}"
        )


# ============================================================
# RESULTADOS POR SOURCE
# ============================================================

print("\n" + "=" * 100)
print("COMPARAÇÃO POR SOURCE")
print("=" * 100)

for comparacao in df_source["comparacao"].unique():

    print(f"\n{comparacao}")

    bloco = df_source[
        df_source["comparacao"] == comparacao
    ]

    print(
        bloco.to_string(
            index=False
        )
    )


# ============================================================
# SALVAMENTO
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

df_por_query.to_csv(
    ARQUIVO_POR_QUERY,
    index=False
)

df_resumo.to_csv(
    ARQUIVO_GLOBAL,
    index=False
)

df_source.to_csv(
    ARQUIVO_SOURCE,
    index=False
)


print("\n" + "=" * 100)
print("ARQUIVOS SALVOS")
print("=" * 100)

print(ARQUIVO_POR_QUERY)
print(ARQUIVO_GLOBAL)
print(ARQUIVO_SOURCE)

print(
    "\nComparação das três arquiteturas concluída."
)