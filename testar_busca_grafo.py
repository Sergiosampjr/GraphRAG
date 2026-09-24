from pathlib import Path
import math
import pickle
from collections import defaultdict

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"

ARQUIVO_FAISS = Path("data/rag/juristcu.faiss")
ARQUIVO_METADATA = Path("data/rag/metadata.parquet")
ARQUIVO_GRAFO = Path("data/graph/juristcu_graph.gpickle")

PERGUNTA = "técnica e preço"

# Quantidade de documentos usados como sementes do grafo
TOP_DOCS_SEMENTE = 10

# Resultado final
TOP_FINAL = 10

# RRF:
# constante fixa usada na fusão dos rankings.
# NÃO vamos ajustar olhando as 150 queries.
RRF_K = 60


# ============================================================
# PESOS DAS RELAÇÕES
# ============================================================

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
    Normaliza doc_id para int sempre que possível.
    """
    if pd.isna(valor):
        return None

    try:
        return int(valor)
    except (ValueError, TypeError):
        return str(valor)


def obter_tipo_relacao(dados_aresta):
    """
    Tenta localizar o nome da relação independentemente
    do nome exato usado no atributo da aresta.
    """
    for chave in ["relacao", "relation", "tipo", "type"]:
        if chave in dados_aresta:
            return str(dados_aresta[chave])

    return None


def obter_label_no(grafo, no):
    """
    Retorna um label legível para impressão.
    """
    dados = grafo.nodes[no]

    for chave in [
        "label",
        "nome",
        "valor",
        "name",
        "text",
        "titulo",
    ]:
        if chave in dados and dados[chave] not in [None, ""]:
            return str(dados[chave])

    return str(no)


def eh_documento(grafo, no):
    """
    Verifica se o nó representa um documento.
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

    return tipo in {
        "documento",
        "document",
        "doc",
    }


def extrair_doc_id_do_no(grafo, no):
    """
    Obtém o doc_id de um nó Documento.
    """

    dados = grafo.nodes[no]

    for chave in ["doc_id", "document_id", "id_documento"]:
        if chave in dados:
            return normalizar_doc_id(dados[chave])

    # fallback:
    # exemplo de ID de nó como "documento:21064"
    texto = str(no)

    if ":" in texto:
        possivel_id = texto.split(":")[-1]

        try:
            return int(possivel_id)
        except ValueError:
            pass

    return normalizar_doc_id(no)


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 80)
print("TESTE DE BUSCA GRAPHRAG V3 - JurisTCU")
print("=" * 80)


# ------------------------------------------------------------
# E5
# ------------------------------------------------------------

print("\nCarregando modelo E5...")

modelo = SentenceTransformer(MODELO_EMBEDDING)

print("Modelo carregado.")


# ------------------------------------------------------------
# FAISS
# ------------------------------------------------------------

print("\nCarregando índice FAISS...")

indice = faiss.read_index(str(ARQUIVO_FAISS))

print(f"Vetores no FAISS: {indice.ntotal}")
print(f"Dimensão dos vetores: {indice.d}")


# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------

print("\nCarregando metadata...")

metadata = pd.read_parquet(ARQUIVO_METADATA)

print(f"Linhas de metadata: {len(metadata)}")

if len(metadata) != indice.ntotal:
    raise RuntimeError(
        "ERRO: quantidade de linhas do metadata "
        "é diferente da quantidade de vetores do FAISS."
    )

if "doc_id" not in metadata.columns:
    raise RuntimeError(
        "ERRO: metadata.parquet não possui coluna 'doc_id'."
    )


# Normaliza apenas uma vez
metadata = metadata.copy()
metadata["doc_id"] = metadata["doc_id"].apply(normalizar_doc_id)


# ------------------------------------------------------------
# GRAFO
# ------------------------------------------------------------

print("\nCarregando grafo...")

with open(ARQUIVO_GRAFO, "rb") as f:
    grafo = pickle.load(f)

print(
    f"Grafo: {grafo.number_of_nodes()} nós / "
    f"{grafo.number_of_edges()} arestas"
)

print(f"Tipo: {type(grafo).__name__}")
print(f"Direcionado: {grafo.is_directed()}")


# ============================================================
# MAPA: doc_id -> nó do grafo
# ============================================================

doc_id_para_no = {}

for no in grafo.nodes:
    if eh_documento(grafo, no):
        doc_id = extrair_doc_id_do_no(grafo, no)

        if doc_id is not None:
            doc_id_para_no[doc_id] = no

print(
    f"Documentos identificados no grafo: "
    f"{len(doc_id_para_no)}"
)


# ============================================================
# 1. RANKING SEMÂNTICO COMPLETO
# ============================================================

def ranking_semantico_documentos(pergunta):
    """
    Calcula o ranking semântico dos documentos.

    IMPORTANTE:
    NÃO recalcula embeddings dos documentos.

    O embedding é calculado somente para a pergunta.

    Depois usamos o próprio índice FAISS, que já possui
    os embeddings dos 51.818 chunks.

    Como cada documento pode possuir vários chunks,
    usamos o MELHOR score de chunk como score do documento.
    """

    texto_query = f"query: {pergunta}"

    embedding_query = modelo.encode(
        [texto_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # --------------------------------------------------------
    # V3:
    # pedimos o ranking de TODOS os chunks.
    #
    # Isso permite conhecer o melhor score semântico
    # de qualquer documento descoberto posteriormente
    # pelo grafo.
    # --------------------------------------------------------

    scores, indices = indice.search(
        embedding_query,
        indice.ntotal
    )

    melhor_score_doc = {}
    melhor_chunk_doc = {}

    for score, idx_chunk in zip(scores[0], indices[0]):

        if idx_chunk < 0:
            continue

        doc_id = metadata.iloc[idx_chunk]["doc_id"]

        if doc_id is None:
            continue

        # Como FAISS retorna em ordem decrescente,
        # a primeira ocorrência tende a ser o melhor chunk.
        # Mesmo assim mantemos a comparação explicitamente.
        if (
            doc_id not in melhor_score_doc
            or float(score) > melhor_score_doc[doc_id]
        ):
            melhor_score_doc[doc_id] = float(score)
            melhor_chunk_doc[doc_id] = int(idx_chunk)

    # Ranking por melhor chunk do documento
    ranking = sorted(
        melhor_score_doc.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    rank_por_doc = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(
            ranking,
            start=1
        )
    }

    return {
        "embedding_query": embedding_query,
        "ranking": ranking,
        "scores": melhor_score_doc,
        "ranks": rank_por_doc,
        "melhor_chunk": melhor_chunk_doc,
    }


# ============================================================
# 2. IDF DA ENTIDADE
# ============================================================

def calcular_idf(total_documentos, frequencia_entidade):
    """
    Penaliza entidades muito genéricas e favorece
    entidades mais específicas.
    """

    return math.log(
        1.0
        + (
            total_documentos
            / (1.0 + frequencia_entidade)
        )
    )


# ============================================================
# 3. EXPANSÃO NO GRAFO
# ============================================================

def expandir_grafo(sementes):
    """
    Expande as sementes através das entidades relacionadas.

    Documento -> entidade -> outros documentos

    Retorna:
        score_grafo_bruto por documento
        evidências
    """

    total_documentos = len(doc_id_para_no)

    scores_grafo = defaultdict(float)
    evidencias = defaultdict(list)

    print("\n" + "=" * 80)
    print("EXPANSÃO DO GRAFO COM IDF")
    print("=" * 80)

    for posicao, (doc_id_semente, score_semente) in enumerate(
        sementes,
        start=1,
    ):

        if doc_id_semente not in doc_id_para_no:
            print(
                f"\nSemente #{posicao}: "
                f"doc_id={doc_id_semente} "
                "não encontrada no grafo."
            )
            continue

        no_documento = doc_id_para_no[doc_id_semente]

        print(
            f"\nSemente #{posicao}: "
            f"doc_id={doc_id_semente} | "
            f"score={score_semente:.6f}"
        )

        # Documento -> entidades
        for entidade in grafo.successors(no_documento):

            dados_aresta = grafo.get_edge_data(
                no_documento,
                entidade
            )

            if not dados_aresta:
                continue

            relacao = obter_tipo_relacao(dados_aresta)

            if relacao not in PESOS_RELACOES:
                continue

            peso_relacao = PESOS_RELACOES[relacao]

            # ------------------------------------------------
            # Entidade -> documentos conectados
            #
            # Como o grafo é:
            # Documento -> Entidade
            #
            # os documentos relacionados à entidade são
            # seus predecessores.
            # ------------------------------------------------

            documentos_conectados = []

            for predecessor in grafo.predecessors(entidade):

                if not eh_documento(grafo, predecessor):
                    continue

                outro_doc_id = extrair_doc_id_do_no(
                    grafo,
                    predecessor
                )

                if outro_doc_id is not None:
                    documentos_conectados.append(
                        (
                            outro_doc_id,
                            predecessor,
                        )
                    )

            frequencia = len(documentos_conectados)

            if frequencia == 0:
                continue

            idf = calcular_idf(
                total_documentos,
                frequencia
            )

            label_entidade = obter_label_no(
                grafo,
                entidade
            )

            print(
                f"  {relacao:20s} | "
                f"{label_entidade} | "
                f"docs={frequencia:4d} | "
                f"idf={idf:.4f}"
            )

            # ------------------------------------------------
            # Pontuação estrutural
            # ------------------------------------------------

            contribuicao = (
                score_semente
                * peso_relacao
                * idf
            )

            for candidato_doc_id, _ in documentos_conectados:

                # Não é necessário excluir sementes:
                # uma semente também pode receber evidência
                # estrutural de OUTRAS sementes.
                if candidato_doc_id == doc_id_semente:
                    continue

                scores_grafo[candidato_doc_id] += contribuicao

                evidencias[candidato_doc_id].append(
                    {
                        "semente": doc_id_semente,
                        "relacao": relacao,
                        "entidade": label_entidade,
                        "frequencia": frequencia,
                        "idf": idf,
                        "contribuicao": contribuicao,
                    }
                )

    return dict(scores_grafo), dict(evidencias)


# ============================================================
# 4. RANKING ESTRUTURAL
# ============================================================

def construir_ranking_grafo(scores_grafo):
    """
    Ordena candidatos pelo score estrutural bruto.
    """

    ranking_grafo = sorted(
        scores_grafo.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    rank_grafo_por_doc = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(
            ranking_grafo,
            start=1
        )
    }

    return ranking_grafo, rank_grafo_por_doc


# ============================================================
# 5. FUSÃO RRF
# ============================================================

def fundir_com_rrf(
    scores_semanticos,
    ranks_semanticos,
    scores_grafo,
    ranks_grafo,
    sementes_ids,
):
    """
    Reciprocal Rank Fusion.

    Em vez de fazer:

        0.70 * score_semantico
        +
        0.30 * score_grafo

    combinamos as POSIÇÕES dos rankings.

    Isso evita comparar diretamente:

        cosine similarity
        versus
        score estrutural IDF

    que possuem escalas diferentes.
    """

    # Candidatos realmente envolvidos na recuperação GraphRAG
    candidatos = set(scores_grafo.keys())
    candidatos.update(sementes_ids)

    resultados = []

    for doc_id in candidatos:

        # --------------------------------------------
        # Ranking semântico REAL
        # --------------------------------------------

        rank_sem = ranks_semanticos.get(doc_id)

        score_sem = scores_semanticos.get(doc_id)

        if rank_sem is None:
            # Em princípio não deveria acontecer,
            # pois fizemos ranking de todo o índice.
            rrf_sem = 0.0
        else:
            rrf_sem = 1.0 / (RRF_K + rank_sem)

        # --------------------------------------------
        # Ranking estrutural
        # --------------------------------------------

        rank_grafo = ranks_grafo.get(doc_id)

        score_grafo = scores_grafo.get(
            doc_id,
            0.0
        )

        if rank_grafo is None:
            rrf_grafo = 0.0
        else:
            rrf_grafo = 1.0 / (
                RRF_K + rank_grafo
            )

        # --------------------------------------------
        # Fusão
        # --------------------------------------------

        score_final = (
            rrf_sem
            + rrf_grafo
        )

        if (
            doc_id in sementes_ids
            and doc_id in scores_grafo
        ):
            origem = "semente+grafo"

        elif doc_id in sementes_ids:
            origem = "semente"

        else:
            origem = "grafo"

        resultados.append(
            {
                "doc_id": doc_id,
                "origem": origem,

                "score_semantico": score_sem,
                "rank_semantico": rank_sem,
                "rrf_semantico": rrf_sem,

                "score_grafo_bruto": score_grafo,
                "rank_grafo": rank_grafo,
                "rrf_grafo": rrf_grafo,

                "score_final": score_final,
            }
        )

    resultados.sort(
        key=lambda x: x["score_final"],
        reverse=True,
    )

    return resultados


# ============================================================
# EXECUÇÃO
# ============================================================

print("\n" + "=" * 80)
print("PERGUNTA")
print("=" * 80)
print(PERGUNTA)


# ------------------------------------------------------------
# ETAPA 1
# Ranking semântico
# ------------------------------------------------------------

print("\nCalculando ranking semântico completo...")

resultado_semantico = ranking_semantico_documentos(
    PERGUNTA
)

ranking_semantico = resultado_semantico["ranking"]
scores_semanticos = resultado_semantico["scores"]
ranks_semanticos = resultado_semantico["ranks"]

print(
    f"Documentos com score semântico: "
    f"{len(ranking_semantico)}"
)


# ------------------------------------------------------------
# ETAPA 2
# Selecionar sementes
# ------------------------------------------------------------

sementes = ranking_semantico[
    :TOP_DOCS_SEMENTE
]

sementes_ids = {
    doc_id
    for doc_id, _ in sementes
}

print("\n" + "=" * 80)
print("DOCUMENTOS-SEMENTE DO FAISS")
print("=" * 80)

for posicao, (doc_id, score) in enumerate(
    sementes,
    start=1,
):
    print(
        f"{posicao:2d}. "
        f"doc_id={doc_id} | "
        f"score={score:.6f}"
    )


# ------------------------------------------------------------
# ETAPA 3
# Expansão estrutural
# ------------------------------------------------------------

scores_grafo, evidencias = expandir_grafo(
    sementes
)

print(
    "\nDocumentos candidatos descobertos pelo grafo: "
    f"{len(scores_grafo)}"
)


# ------------------------------------------------------------
# ETAPA 4
# Ranking estrutural
# ------------------------------------------------------------

ranking_grafo, ranks_grafo = construir_ranking_grafo(
    scores_grafo
)


# ------------------------------------------------------------
# Diagnóstico: melhores candidatos puramente estruturais
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TOP-10 CANDIDATOS ESTRUTURAIS")
print("=" * 80)

for posicao, (doc_id, score_grafo) in enumerate(
    ranking_grafo[:10],
    start=1,
):

    rank_sem = ranks_semanticos.get(doc_id)
    score_sem = scores_semanticos.get(doc_id)

    origem = (
        "semente"
        if doc_id in sementes_ids
        else "novo"
    )

    print(
        f"{posicao:2d}. "
        f"doc_id={doc_id} | "
        f"grafo={score_grafo:.6f} | "
        f"semântico={score_sem:.6f} | "
        f"rank_sem={rank_sem} | "
        f"{origem}"
    )


# ------------------------------------------------------------
# ETAPA 5
# Fusão RRF
# ------------------------------------------------------------

resultados = fundir_com_rrf(
    scores_semanticos=scores_semanticos,
    ranks_semanticos=ranks_semanticos,
    scores_grafo=scores_grafo,
    ranks_grafo=ranks_grafo,
    sementes_ids=sementes_ids,
)


# ------------------------------------------------------------
# TOP-10 FINAL
# ------------------------------------------------------------

top_final = resultados[:TOP_FINAL]

print("\n" + "=" * 80)
print("TOP-10 GRAPHRAG V3 - RRF")
print("=" * 80)

for posicao, resultado in enumerate(
    top_final,
    start=1,
):

    doc_id = resultado["doc_id"]

    print(
        f"\n{posicao:2d}. "
        f"doc_id={doc_id}"
    )

    print(
        f"    origem             : "
        f"{resultado['origem']}"
    )

    print(
        f"    score_semantico    : "
        f"{resultado['score_semantico']:.6f}"
    )

    print(
        f"    rank_semantico     : "
        f"{resultado['rank_semantico']}"
    )

    print(
        f"    score_grafo_bruto  : "
        f"{resultado['score_grafo_bruto']:.6f}"
    )

    print(
        f"    rank_grafo         : "
        f"{resultado['rank_grafo']}"
    )

    print(
        f"    rrf_semantico      : "
        f"{resultado['rrf_semantico']:.8f}"
    )

    print(
        f"    rrf_grafo          : "
        f"{resultado['rrf_grafo']:.8f}"
    )

    print(
        f"    score_final_rrf    : "
        f"{resultado['score_final']:.8f}"
    )

    # --------------------------------------------
    # Evidências estruturais
    # --------------------------------------------

    evidencias_doc = evidencias.get(
        doc_id,
        []
    )

    evidencias_ordenadas = sorted(
        evidencias_doc,
        key=lambda x: x["contribuicao"],
        reverse=True,
    )

    if evidencias_ordenadas:

        print("    evidências:")

        for ev in evidencias_ordenadas[:5]:

            print(
                "      "
                f"semente={ev['semente']} | "
                f"{ev['relacao']} | "
                f"{ev['entidade']} | "
                f"docs={ev['frequencia']} | "
                f"idf={ev['idf']:.4f} | "
                f"+{ev['contribuicao']:.6f}"
            )


# ============================================================
# DIAGNÓSTICO FINAL
# ============================================================

docs_top_final = {
    r["doc_id"]
    for r in top_final
}

novos_documentos = (
    docs_top_final
    - sementes_ids
)

sementes_mantidas = (
    docs_top_final
    & sementes_ids
)

print("\n" + "=" * 80)
print("DIAGNÓSTICO V3")
print("=" * 80)

print(
    f"Sementes FAISS no Top-10 final: "
    f"{len(sementes_mantidas)}"
)

print(
    f"Novos documentos promovidos pelo grafo: "
    f"{len(novos_documentos)}"
)

if novos_documentos:

    print("\nDocumentos novos no Top-10:")

    for doc_id in sorted(novos_documentos):

        resultado = next(
            r
            for r in top_final
            if r["doc_id"] == doc_id
        )

        print(
            f"  doc_id={doc_id} | "
            f"rank_sem={resultado['rank_semantico']} | "
            f"rank_grafo={resultado['rank_grafo']} | "
            f"score_sem={resultado['score_semantico']:.6f}"
        )

else:

    print(
        "\nNenhum documento novo entrou no Top-10."
    )

    print(
        "ATENÇÃO: isso não significa automaticamente "
        "que a V3 está errada."
    )

    print(
        "Agora, diferentemente da V2, os candidatos do "
        "grafo tiveram um score/rank semântico REAL."
    )


print("\n" + "=" * 80)
print("TESTE CONCLUÍDO")
print("=" * 80)