from datasets import load_dataset
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

PERGUNTA = "técnica e preço"


# Top-10 RAG original
TOP10_RAG = [
    20584,
    33887,
    20506,
    18324,
    20970,
    31526,
    21207,
    21064,
    21285,
    18894,
]


# Top-10 GraphRAG V3
TOP10_GRAPHRAG = [
    20506,
    33887,
    20584,
    20970,
    21207,
    31526,
    31594,
    21064,
    34211,
    20869,
]


# ============================================================
# CARREGAR QUERY E QREL
# ============================================================

print("=" * 80)
print("VERIFICAÇÃO DE QRELS - JurisTCU")
print("=" * 80)

print("\nCarregando query.csv...")

queries = load_dataset(
    "csv",
    data_files=BASE_URL + "query.csv",
    split="train"
)

print("Carregando qrel.csv...")

qrels = load_dataset(
    "csv",
    data_files=BASE_URL + "qrel.csv",
    split="train"
)

df_queries = queries.to_pandas()
df_qrels = qrels.to_pandas()


# ============================================================
# ENCONTRAR QUERY
# ============================================================

print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)

query_encontrada = df_queries[
    df_queries["TEXT"].str.strip().str.lower()
    == PERGUNTA.strip().lower()
]

if query_encontrada.empty:
    raise RuntimeError(
        f'Query "{PERGUNTA}" não encontrada.'
    )

query_id = int(query_encontrada.iloc[0]["ID"])
source = query_encontrada.iloc[0]["SOURCE"]

print(f"QUERY_ID : {query_id}")
print(f"TEXT     : {PERGUNTA}")
print(f"SOURCE   : {source}")


# ============================================================
# QRELS DA QUERY
# ============================================================

qrels_query = df_qrels[
    df_qrels["QUERY_ID"] == query_id
].copy()

# Garantir tipos comparáveis
qrels_query["DOC_ID"] = qrels_query["DOC_ID"].astype(int)
qrels_query["SCORE"] = qrels_query["SCORE"].astype(int)

print("\nQuantidade de documentos julgados:")
print(len(qrels_query))


# Mapa:
# doc_id -> score
mapa_qrels = dict(
    zip(
        qrels_query["DOC_ID"],
        qrels_query["SCORE"]
    )
)


# ============================================================
# FUNÇÃO DE CLASSIFICAÇÃO
# ============================================================

def descricao_score(score):

    if score is None:
        return "NÃO JULGADO"

    if score == 0:
        return "IRRELEVANTE"

    if score == 1:
        return "RELACIONADO"

    if score == 2:
        return "RELEVANTE"

    if score == 3:
        return "ALTAMENTE RELEVANTE"

    return "DESCONHECIDO"


# ============================================================
# MOSTRAR RANKING
# ============================================================

def mostrar_ranking(nome, ranking):

    print("\n" + "=" * 80)
    print(nome)
    print("=" * 80)

    relevantes = 0
    julgados = 0

    for posicao, doc_id in enumerate(
        ranking,
        start=1
    ):

        score = mapa_qrels.get(doc_id)

        if score is not None:
            julgados += 1

            if score > 0:
                relevantes += 1

        print(
            f"{posicao:2d}. "
            f"doc_id={doc_id:<6} | "
            f"SCORE={str(score):<4} | "
            f"{descricao_score(score)}"
        )

    print("\nResumo:")
    print(f"Julgados   : {julgados}/10")
    print(f"Relevantes : {relevantes}/10")


# ============================================================
# RAG
# ============================================================

mostrar_ranking(
    "TOP-10 RAG",
    TOP10_RAG
)


# ============================================================
# GRAPHRAG
# ============================================================

mostrar_ranking(
    "TOP-10 GRAPHRAG V3",
    TOP10_GRAPHRAG
)


# ============================================================
# DOCUMENTOS QUE SAÍRAM / ENTRARAM
# ============================================================

set_rag = set(TOP10_RAG)
set_graphrag = set(TOP10_GRAPHRAG)

saíram = [
    doc
    for doc in TOP10_RAG
    if doc not in set_graphrag
]

entraram = [
    doc
    for doc in TOP10_GRAPHRAG
    if doc not in set_rag
]


print("\n" + "=" * 80)
print("ALTERAÇÕES NO TOP-10")
print("=" * 80)

print("\nDOCUMENTOS QUE SAÍRAM DO RAG:")

for doc_id in saíram:

    score = mapa_qrels.get(doc_id)

    print(
        f"doc_id={doc_id:<6} | "
        f"SCORE={str(score):<4} | "
        f"{descricao_score(score)}"
    )


print("\nDOCUMENTOS QUE ENTRARAM COM O GRAPHRAG:")

for doc_id in entraram:

    score = mapa_qrels.get(doc_id)

    print(
        f"doc_id={doc_id:<6} | "
        f"SCORE={str(score):<4} | "
        f"{descricao_score(score)}"
    )


# ============================================================
# TODOS OS QRELS CONHECIDOS
# ============================================================

print("\n" + "=" * 80)
print("TODOS OS DOCUMENTOS JULGADOS PARA ESSA QUERY")
print("=" * 80)

qrels_query = qrels_query.sort_values(
    by=["SCORE", "DOC_ID"],
    ascending=[False, True]
)

for _, linha in qrels_query.iterrows():

    print(
        f"doc_id={int(linha['DOC_ID']):<6} | "
        f"SCORE={int(linha['SCORE'])} | "
        f"{descricao_score(int(linha['SCORE']))}"
    )


print("\n" + "=" * 80)
print("VERIFICAÇÃO CONCLUÍDA")
print("=" * 80)