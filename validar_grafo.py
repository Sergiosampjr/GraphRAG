from pathlib import Path
import pickle
from collections import Counter

import networkx as nx


ARQUIVO_GRAFO = Path(
    "data/graph/juristcu_graph.gpickle"
)


def carregar_grafo():
    print("=" * 80)
    print("VALIDAÇÃO DO GRAFO - JurisTCU")
    print("=" * 80)

    if not ARQUIVO_GRAFO.exists():
        raise FileNotFoundError(
            f"Grafo não encontrado: {ARQUIVO_GRAFO}"
        )

    print(f"\nCarregando grafo: {ARQUIVO_GRAFO}")

    with open(ARQUIVO_GRAFO, "rb") as f:
        grafo = pickle.load(f)

    print("Grafo carregado com sucesso.")

    return grafo


def validar_estrutura(grafo):
    print("\n" + "=" * 80)
    print("1. VALIDAÇÃO ESTRUTURAL")
    print("=" * 80)

    print(f"Nós: {grafo.number_of_nodes()}")
    print(f"Arestas: {grafo.number_of_edges()}")

    print(f"Tipo do grafo: {type(grafo).__name__}")
    print(f"É direcionado? {grafo.is_directed()}")
    print(f"É multigrafo? {grafo.is_multigraph()}")

    # --------------------------------------------------------
    # Nós por tipo
    # --------------------------------------------------------

    tipos_nos = Counter()

    for _, dados in grafo.nodes(data=True):
        tipos_nos[dados.get("tipo", "SEM_TIPO")] += 1

    print("\nNós por tipo:")

    for tipo, quantidade in tipos_nos.most_common():
        print(f"{tipo:25s}: {quantidade}")

    # --------------------------------------------------------
    # Relações por tipo
    # --------------------------------------------------------

    tipos_relacoes = Counter()

    for _, _, dados in grafo.edges(data=True):
        tipos_relacoes[dados.get("tipo", "SEM_TIPO")] += 1

    print("\nRelações por tipo:")

    for tipo, quantidade in tipos_relacoes.most_common():
        print(f"{tipo:25s}: {quantidade}")


def validar_nos_isolados(grafo):
    print("\n" + "=" * 80)
    print("2. NÓS ISOLADOS")
    print("=" * 80)

    isolados = list(nx.isolates(grafo))

    print(f"Nós isolados: {len(isolados)}")

    if isolados:
        print("\nPrimeiros nós isolados:")

        for node_id in isolados[:20]:
            print(
                node_id,
                grafo.nodes[node_id]
            )
    else:
        print("Nenhum nó isolado encontrado.")


def validar_documentos(grafo):
    print("\n" + "=" * 80)
    print("3. VALIDAÇÃO DOS DOCUMENTOS")
    print("=" * 80)

    documentos = [
        node_id
        for node_id, dados in grafo.nodes(data=True)
        if dados.get("tipo") == "documento"
    ]

    print(f"Total de documentos: {len(documentos)}")

    documentos_sem_area = 0
    documentos_sem_tema = 0
    documentos_sem_subtema = 0
    documentos_sem_autor = 0
    documentos_sem_tipo_processo = 0

    for documento_id in documentos:

        relacoes = Counter()

        for vizinho in grafo.neighbors(documento_id):

            dados_aresta = grafo.get_edge_data(
                documento_id,
                vizinho
            )

            tipo = dados_aresta.get(
                "tipo",
                "SEM_TIPO"
            )

            relacoes[tipo] += 1

        if relacoes["TEM_AREA"] == 0:
            documentos_sem_area += 1

        if relacoes["TEM_TEMA"] == 0:
            documentos_sem_tema += 1

        if relacoes["TEM_SUBTEMA"] == 0:
            documentos_sem_subtema += 1

        if relacoes["TEM_AUTOR_TESE"] == 0:
            documentos_sem_autor += 1

        if relacoes["TEM_TIPO_PROCESSO"] == 0:
            documentos_sem_tipo_processo += 1

    print(
        f"Documentos sem área: "
        f"{documentos_sem_area}"
    )

    print(
        f"Documentos sem tema: "
        f"{documentos_sem_tema}"
    )

    print(
        f"Documentos sem subtema: "
        f"{documentos_sem_subtema}"
    )

    print(
        f"Documentos sem autor: "
        f"{documentos_sem_autor}"
    )

    print(
        f"Documentos sem tipo de processo: "
        f"{documentos_sem_tipo_processo}"
    )


def mostrar_documento(grafo, doc_id):
    print("\n" + "-" * 80)
    print(f"DOCUMENTO {doc_id}")
    print("-" * 80)

    documento_id = f"documento::{doc_id}"

    if not grafo.has_node(documento_id):

        print("Documento não encontrado no grafo.")
        return

    print("\nAtributos:")

    for chave, valor in grafo.nodes[documento_id].items():
        print(f"{chave}: {valor}")

    print("\nConexões:")

    conexoes = []

    for vizinho in grafo.neighbors(documento_id):

        dados_no = grafo.nodes[vizinho]

        dados_aresta = grafo.get_edge_data(
            documento_id,
            vizinho
        )

        conexoes.append(
            (
                dados_aresta.get("tipo"),
                dados_no.get("tipo"),
                dados_no.get("nome"),
                vizinho
            )
        )

    conexoes.sort(
        key=lambda x: (
            str(x[0]),
            str(x[2])
        )
    )

    for relacao, tipo_no, nome, node_id in conexoes:

        print(
            f"{relacao:20s} "
            f"-> "
            f"{tipo_no:20s} "
            f"| {nome}"
        )


def validar_exemplos_documentos(grafo):
    print("\n" + "=" * 80)
    print("4. EXEMPLOS DE DOCUMENTOS")
    print("=" * 80)

    # Documento que já conhecemos do benchmark JurisTCU
    exemplos = [
        21064
    ]

    # Adiciona outros dois documentos automaticamente
    documentos = [
        dados["doc_id"]
        for _, dados in grafo.nodes(data=True)
        if dados.get("tipo") == "documento"
    ]

    for doc_id in documentos[:2]:
        if doc_id not in exemplos:
            exemplos.append(doc_id)

    for doc_id in exemplos:
        mostrar_documento(
            grafo,
            doc_id
        )


def validar_conceitos(grafo):
    print("\n" + "=" * 80)
    print("5. VALIDAÇÃO DOS CONCEITOS / INDEXAÇÃO")
    print("=" * 80)

    conceitos = []

    for node_id, dados in grafo.nodes(data=True):

        if dados.get("tipo") == "conceito":

            conceitos.append(
                (
                    node_id,
                    dados.get("nome"),
                    grafo.degree(node_id)
                )
            )

    conceitos.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print(
        f"Total de conceitos: "
        f"{len(conceitos)}"
    )

    print("\n20 conceitos com maior grau:")

    for node_id, nome, grau in conceitos[:20]:

        print(
            f"{grau:6d} | {nome}"
        )

    print("\nExemplos de conceitos pouco frequentes:")

    for node_id, nome, grau in conceitos[-20:]:

        print(
            f"{grau:6d} | {nome}"
        )


def validar_referencias_legais(grafo):
    print("\n" + "=" * 80)
    print("6. VALIDAÇÃO DAS REFERÊNCIAS LEGAIS")
    print("=" * 80)

    referencias = []

    for node_id, dados in grafo.nodes(data=True):

        if dados.get("tipo") == "referencia_legal":

            nome = dados.get(
                "nome",
                ""
            )

            referencias.append(
                (
                    node_id,
                    nome,
                    grafo.degree(node_id)
                )
            )

    referencias.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print(
        f"Total de referências legais: "
        f"{len(referencias)}"
    )

    print("\n20 referências com maior grau:")

    for _, nome, grau in referencias[:20]:

        print(
            f"{grau:6d} | {nome}"
        )

    print("\n30 exemplos de referências legais:")

    for _, nome, grau in referencias[:30]:

        print(
            f"{grau:6d} | {nome}"
        )

    # --------------------------------------------------------
    # Busca fragmentos potencialmente suspeitos
    # --------------------------------------------------------

    palavras_suspeitas = {
        "inc.",
        "art.",
        "par.",
        "alínea",
        "congresso nacional",
        "tcu",
        "presidente da república"
    }

    suspeitas = []

    for _, nome, grau in referencias:

        nome_limpo = str(nome).strip().lower()

        if nome_limpo in palavras_suspeitas:
            suspeitas.append(
                (nome, grau)
            )

        elif len(nome_limpo) <= 5:
            suspeitas.append(
                (nome, grau)
            )

    print("\nPossíveis referências fragmentadas:")

    if suspeitas:

        for nome, grau in suspeitas[:50]:

            print(
                f"{grau:6d} | {nome}"
            )

    else:

        print(
            "Nenhum fragmento óbvio "
            "encontrado por esta regra."
        )


def validar_hierarquia(grafo):
    print("\n" + "=" * 80)
    print("7. VALIDAÇÃO DA HIERARQUIA")
    print("=" * 80)

    areas = [
        node_id
        for node_id, dados in grafo.nodes(data=True)
        if dados.get("tipo") == "area"
    ]

    print(
        f"Áreas encontradas: "
        f"{len(areas)}"
    )

    for area_id in sorted(
        areas,
        key=lambda x: grafo.nodes[x].get("nome", "")
    ):

        nome_area = grafo.nodes[area_id].get(
            "nome"
        )

        temas = []

        for vizinho in grafo.neighbors(area_id):

            dados_aresta = grafo.get_edge_data(
                area_id,
                vizinho
            )

            dados_no = grafo.nodes[vizinho]

            if (
                dados_aresta.get("tipo")
                == "POSSUI_TEMA"
                and dados_no.get("tipo")
                == "tema"
            ):

                temas.append(
                    dados_no.get("nome")
                )

        print(
            f"{nome_area:30s}: "
            f"{len(temas)} temas"
        )


def validar_graus(grafo):
    print("\n" + "=" * 80)
    print("8. NÓS COM MAIOR GRAU")
    print("=" * 80)

    graus = sorted(
        grafo.degree,
        key=lambda x: x[1],
        reverse=True
    )

    for node_id, grau in graus[:30]:

        dados = grafo.nodes[node_id]

        print(
            f"{grau:6d} | "
            f"{dados.get('tipo', ''):20s} | "
            f"{dados.get('nome', node_id)}"
        )


def main():

    grafo = carregar_grafo()

    validar_estrutura(grafo)

    validar_nos_isolados(grafo)

    validar_documentos(grafo)

    validar_exemplos_documentos(grafo)

    validar_conceitos(grafo)

    validar_referencias_legais(grafo)

    validar_hierarquia(grafo)

    validar_graus(grafo)

    print("\n" + "=" * 80)
    print("VALIDAÇÃO CONCLUÍDA")
    print("=" * 80)


if __name__ == "__main__":
    main()