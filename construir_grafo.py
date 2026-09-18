from pathlib import Path
import ast
import re
import pickle
import unicodedata
from collections import Counter

import networkx as nx
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_DOCUMENTOS = Path(
    "data/processed/juristcu_documentos.parquet"
)

PASTA_GRAFO = Path("data/graph")

ARQUIVO_GRAFO = PASTA_GRAFO / "juristcu_graph.gpickle"
ARQUIVO_NOS = PASTA_GRAFO / "nos.parquet"
ARQUIVO_ARESTAS = PASTA_GRAFO / "arestas.parquet"


# ============================================================
# FUNÇÕES DE NORMALIZAÇÃO
# ============================================================

def valor_valido(valor):
    """
    Retorna True quando o valor pode ser usado no grafo.
    """

    if valor is None:
        return False

    try:
        if pd.isna(valor):
            return False
    except (TypeError, ValueError):
        pass

    return str(valor).strip() != ""


def normalizar_texto(valor):
    """
    Remove espaços extras preservando o texto apresentado
    ao usuário.
    """

    valor = str(valor).strip()

    return re.sub(
        r"\s+",
        " ",
        valor
    )


def normalizar_id(valor):
    """
    Cria uma versão estável do texto para utilização
    dentro dos identificadores dos nós.

    Exemplo:
        "Licitação de Técnica e Preço"
        ->
        "licitacao_de_tecnica_e_preco"
    """

    texto = normalizar_texto(valor)

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = texto.lower()

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto
    )

    texto = texto.strip("_")

    return texto


# ============================================================
# IDENTIFICADORES DOS NÓS
# ============================================================

def id_documento(doc_id):

    return f"documento::{int(doc_id)}"


def id_area(area):

    return (
        f"area::"
        f"{normalizar_id(area)}"
    )


def id_tema(area, tema):
    """
    Tema é contextualizado pela área.

    Assim, temas de mesmo nome em áreas diferentes
    permanecem entidades diferentes.
    """

    return (
        f"tema::"
        f"{normalizar_id(area)}::"
        f"{normalizar_id(tema)}"
    )


def id_subtema(area, tema, subtema):
    """
    Subtema é contextualizado pela área e pelo tema.

    Evita fundir, por exemplo, o subtema "Requisito"
    pertencente a temas diferentes.
    """

    return (
        f"subtema::"
        f"{normalizar_id(area)}::"
        f"{normalizar_id(tema)}::"
        f"{normalizar_id(subtema)}"
    )


def id_entidade_global(tipo, valor):
    """
    Entidades globais são compartilhadas entre documentos.

    Exemplos:
        autor
        conceito
        tipo_processo
        referencia_legal
    """

    return (
        f"{tipo}::"
        f"{normalizar_id(valor)}"
    )


# ============================================================
# CRIAÇÃO DE NÓS
# ============================================================

def adicionar_area(grafo, area):

    if not valor_valido(area):
        return None

    nome = normalizar_texto(area)
    node_id = id_area(nome)

    if not grafo.has_node(node_id):

        grafo.add_node(
            node_id,
            tipo="area",
            nome=nome
        )

    return node_id


def adicionar_tema(
    grafo,
    area,
    tema
):

    if (
        not valor_valido(area)
        or not valor_valido(tema)
    ):
        return None

    nome_area = normalizar_texto(area)
    nome_tema = normalizar_texto(tema)

    node_id = id_tema(
        nome_area,
        nome_tema
    )

    if not grafo.has_node(node_id):

        grafo.add_node(
            node_id,
            tipo="tema",
            nome=nome_tema,
            area=nome_area
        )

    return node_id


def adicionar_subtema(
    grafo,
    area,
    tema,
    subtema
):

    if (
        not valor_valido(area)
        or not valor_valido(tema)
        or not valor_valido(subtema)
    ):
        return None

    nome_area = normalizar_texto(area)
    nome_tema = normalizar_texto(tema)
    nome_subtema = normalizar_texto(subtema)

    node_id = id_subtema(
        nome_area,
        nome_tema,
        nome_subtema
    )

    if not grafo.has_node(node_id):

        grafo.add_node(
            node_id,
            tipo="subtema",
            nome=nome_subtema,
            area=nome_area,
            tema=nome_tema
        )

    return node_id


def adicionar_entidade_global(
    grafo,
    tipo,
    valor
):

    if not valor_valido(valor):
        return None

    nome = normalizar_texto(valor)

    node_id = id_entidade_global(
        tipo,
        nome
    )

    if not grafo.has_node(node_id):

        grafo.add_node(
            node_id,
            tipo=tipo,
            nome=nome
        )

    return node_id


# ============================================================
# RELAÇÕES
# ============================================================

def adicionar_relacao(
    grafo,
    origem,
    destino,
    tipo_relacao
):

    if origem is None or destino is None:
        return

    grafo.add_edge(
        origem,
        destino,
        tipo=tipo_relacao
    )


# ============================================================
# CAMPOS EM FORMATO DE LISTA
# ============================================================

def separar_lista(valor):
    """
    Separa campos como:

        [Requisito, Legislação, Marco temporal]

    em:

        Requisito
        Legislação
        Marco temporal

    Também aceita listas Python reais ou serializadas.
    """

    if not valor_valido(valor):
        return []

    if isinstance(
        valor,
        (list, tuple, set)
    ):

        return [
            normalizar_texto(item)
            for item in valor
            if valor_valido(item)
        ]

    texto = normalizar_texto(valor)

    # Tenta primeiro interpretar listas Python como:
    #
    # ['A', 'B', 'C']

    try:

        objeto = ast.literal_eval(texto)

        if isinstance(
            objeto,
            (list, tuple, set)
        ):

            return [
                normalizar_texto(item)
                for item in objeto
                if valor_valido(item)
            ]

    except (ValueError, SyntaxError):
        pass

    # Formato observado no JurisTCU:
    #
    # [A, B, C]

    if (
        texto.startswith("[")
        and texto.endswith("]")
    ):

        texto = texto[1:-1].strip()

    if not texto:
        return []

    return [
        normalizar_texto(item)
        for item in texto.split(",")
        if valor_valido(item)
    ]


# ============================================================
# CONSTRUÇÃO DO GRAFO
# ============================================================

def construir_grafo(df):

    # IMPORTANTE:
    #
    # Agora utilizamos um grafo DIRECIONADO.
    #
    # Documento -> Área
    # Área -> Tema
    # Tema -> Subtema
    # Documento -> Conceito
    # etc.

    grafo = nx.DiGraph()

    total = len(df)

    for numero, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        # ====================================================
        # DOCUMENTO
        # ====================================================

        doc_id = int(row["doc_id"])

        documento_id = id_documento(
            doc_id
        )

        grafo.add_node(
            documento_id,
            tipo="documento",
            doc_id=doc_id,
            num_acordao=row["num_acordao"],
            ano_acordao=row["ano_acordao"],
            colegiado=row["colegiado"],
            paradigmatico=row["paradigmatico"]
        )

        # ====================================================
        # TAXONOMIA
        #
        # Área -> Tema -> Subtema
        # ====================================================

        area = row["area"]
        tema = row["tema"]
        subtema = row["subtema"]

        area_id = adicionar_area(
            grafo,
            area
        )

        tema_id = adicionar_tema(
            grafo,
            area,
            tema
        )

        subtema_id = adicionar_subtema(
            grafo,
            area,
            tema,
            subtema
        )

        # Documento -> Área

        adicionar_relacao(
            grafo,
            documento_id,
            area_id,
            "TEM_AREA"
        )

        # Documento -> Tema

        adicionar_relacao(
            grafo,
            documento_id,
            tema_id,
            "TEM_TEMA"
        )

        # Documento -> Subtema

        adicionar_relacao(
            grafo,
            documento_id,
            subtema_id,
            "TEM_SUBTEMA"
        )

        # Área -> Tema

        adicionar_relacao(
            grafo,
            area_id,
            tema_id,
            "POSSUI_TEMA"
        )

        # Tema -> Subtema

        adicionar_relacao(
            grafo,
            tema_id,
            subtema_id,
            "POSSUI_SUBTEMA"
        )

        # ====================================================
        # AUTOR DA TESE
        # ====================================================

        autor_id = adicionar_entidade_global(
            grafo,
            "autor",
            row["autor_tese"]
        )

        adicionar_relacao(
            grafo,
            documento_id,
            autor_id,
            "TEM_AUTOR_TESE"
        )

        # ====================================================
        # TIPO DE PROCESSO
        # ====================================================

        tipo_processo_id = (
            adicionar_entidade_global(
                grafo,
                "tipo_processo",
                row["tipo_processo"]
            )
        )

        adicionar_relacao(
            grafo,
            documento_id,
            tipo_processo_id,
            "TEM_TIPO_PROCESSO"
        )

        # ====================================================
        # CONCEITOS / INDEXAÇÃO
        # ====================================================

        conceitos = separar_lista(
            row["indexacao"]
        )

        for conceito in conceitos:

            conceito_id = (
                adicionar_entidade_global(
                    grafo,
                    "conceito",
                    conceito
                )
            )

            adicionar_relacao(
                grafo,
                documento_id,
                conceito_id,
                "INDEXADO_POR"
            )

        # ====================================================
        # REFERÊNCIAS LEGAIS
        # ====================================================

        referencias = separar_lista(
            row["referencia_legal"]
        )

        for referencia in referencias:

            referencia_id = (
                adicionar_entidade_global(
                    grafo,
                    "referencia_legal",
                    referencia
                )
            )

            adicionar_relacao(
                grafo,
                documento_id,
                referencia_id,
                "CITA"
            )

        # ====================================================
        # PROGRESSO
        # ====================================================

        if (
            numero % 1000 == 0
            or numero == total
        ):

            print(
                f"Processados: "
                f"{numero}/{total}"
            )

    return grafo


# ============================================================
# ESTATÍSTICAS
# ============================================================

def mostrar_estatisticas(grafo):

    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DO GRAFO")
    print("=" * 80)

    print(
        f"Tipo do grafo: "
        f"{type(grafo).__name__}"
    )

    print(
        f"Direcionado: "
        f"{grafo.is_directed()}"
    )

    print(
        f"Número de nós: "
        f"{grafo.number_of_nodes()}"
    )

    print(
        f"Número de arestas: "
        f"{grafo.number_of_edges()}"
    )

    # --------------------------------------------------------
    # NÓS
    # --------------------------------------------------------

    tipos_nos = Counter(
        dados.get(
            "tipo",
            "desconhecido"
        )
        for _, dados
        in grafo.nodes(data=True)
    )

    print("\nNós por tipo:")

    for tipo, quantidade in (
        tipos_nos.most_common()
    ):

        print(
            f"{tipo:25s}: "
            f"{quantidade}"
        )

    # --------------------------------------------------------
    # RELAÇÕES
    # --------------------------------------------------------

    tipos_relacoes = Counter(
        dados.get(
            "tipo",
            "desconhecido"
        )
        for _, _, dados
        in grafo.edges(data=True)
    )

    print("\nRelações por tipo:")

    for tipo, quantidade in (
        tipos_relacoes.most_common()
    ):

        print(
            f"{tipo:25s}: "
            f"{quantidade}"
        )


# ============================================================
# EXPORTAÇÃO
# ============================================================

def exportar_tabelas(grafo):

    # --------------------------------------------------------
    # NÓS
    # --------------------------------------------------------

    print("\nExportando nós...")

    nos = []

    for node_id, atributos in (
        grafo.nodes(data=True)
    ):

        nos.append(
            {
                "node_id": node_id,
                **atributos
            }
        )

    df_nos = pd.DataFrame(nos)

    df_nos.to_parquet(
        ARQUIVO_NOS,
        index=False
    )

    print(
        f"Nós salvos em: "
        f"{ARQUIVO_NOS}"
    )

    # --------------------------------------------------------
    # ARESTAS
    # --------------------------------------------------------

    print("\nExportando arestas...")

    arestas = []

    for origem, destino, atributos in (
        grafo.edges(data=True)
    ):

        arestas.append(
            {
                "origem": origem,
                "destino": destino,
                **atributos
            }
        )

    df_arestas = pd.DataFrame(
        arestas
    )

    df_arestas.to_parquet(
        ARQUIVO_ARESTAS,
        index=False
    )

    print(
        f"Arestas salvas em: "
        f"{ARQUIVO_ARESTAS}"
    )


# ============================================================
# SALVAMENTO
# ============================================================

def salvar_grafo(grafo):

    print("\nSalvando grafo NetworkX...")

    with open(
        ARQUIVO_GRAFO,
        "wb"
    ) as arquivo:

        pickle.dump(
            grafo,
            arquivo,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"Grafo salvo em: "
        f"{ARQUIVO_GRAFO}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONSTRUÇÃO DO GRAFO - JurisTCU")
    print("=" * 80)

    if not ARQUIVO_DOCUMENTOS.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{ARQUIVO_DOCUMENTOS}"
        )

    PASTA_GRAFO.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nCarregando: "
        f"{ARQUIVO_DOCUMENTOS}"
    )

    df = pd.read_parquet(
        ARQUIVO_DOCUMENTOS
    )

    print(
        f"Documentos carregados: "
        f"{len(df)}"
    )

    print(
        "\nConstruindo grafo "
        "direcionado..."
    )

    grafo = construir_grafo(df)

    mostrar_estatisticas(
        grafo
    )

    exportar_tabelas(
        grafo
    )

    salvar_grafo(
        grafo
    )

    print(
        "\nConstrução concluída."
    )


if __name__ == "__main__":
    main()