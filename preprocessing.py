import re
import html
from pathlib import Path

import pandas as pd
from datasets import load_dataset


BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"
OUTPUT_DIR = Path("data/processed")


def limpar_html(texto):
    """
    Remove tags HTML e normaliza espaços.
    """
    if texto is None:
        return ""

    texto = html.unescape(str(texto))

    # Tags que representam separação textual
    texto = re.sub(r"</?(p|br|div|tr|td|table)[^>]*>", " ", texto,
                   flags=re.IGNORECASE)

    # Demais tags HTML
    texto = re.sub(r"<[^>]+>", "", texto)

    # Normalização de espaços
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def valor_seguro(valor):
    """
    Converte valores ausentes em string vazia.
    """
    if valor is None:
        return ""

    return str(valor).strip()


def construir_texto_documento(documento):
    """
    Constrói a representação textual utilizada posteriormente
    pelo pipeline de recuperação.
    """

    partes = []

    area = valor_seguro(documento["AREA"])
    tema = valor_seguro(documento["TEMA"])
    subtema = valor_seguro(documento["SUBTEMA"])

    enunciado = limpar_html(documento["ENUNCIADO"])
    excerto = limpar_html(documento["EXCERTO"])

    if area:
        partes.append(f"Área: {area}")

    if tema:
        partes.append(f"Tema: {tema}")

    if subtema:
        partes.append(f"Subtema: {subtema}")

    if enunciado:
        partes.append(f"Enunciado: {enunciado}")

    if excerto:
        partes.append(f"Excerto: {excerto}")

    return "\n\n".join(partes)


def main():

    print("Carregando JurisTCU...")

    documentos = load_dataset(
        "csv",
        data_files=BASE_URL + "doc.csv",
        split="train"
    )

    print(f"Documentos carregados: {len(documentos)}")

    registros = []

    for documento in documentos:

        registro = {
            # Identificador fundamental para avaliação
            "doc_id": int(documento["KEY"]),

            # Texto usado posteriormente pelo RAG
            "text": construir_texto_documento(documento),

            # Metadados
            "num_acordao": documento["NUMACORDAO"],
            "ano_acordao": documento["ANOACORDAO"],
            "colegiado": valor_seguro(documento["COLEGIADO"]),
            "area": valor_seguro(documento["AREA"]),
            "tema": valor_seguro(documento["TEMA"]),
            "subtema": valor_seguro(documento["SUBTEMA"]),

            "autor_tese": valor_seguro(documento["AUTORTESE"]),
            "funcao_autor": valor_seguro(
                documento["FUNCAOAUTORTESE"]
            ),

            "tipo_processo": valor_seguro(
                documento["TIPOPROCESSO"]
            ),

            "tipo_recurso": valor_seguro(
                documento["TIPORECURSO"]
            ),

            "indexacao": valor_seguro(
                documento["INDEXACAO"]
            ),

            "referencia_legal": valor_seguro(
                documento["REFERENCIALEGAL"]
            ),

            "paradigmatico": valor_seguro(
                documento["PARADIGMATICO"]
            ),
        }

        registros.append(registro)

    df = pd.DataFrame(registros)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "juristcu_documentos.parquet"

    df.to_parquet(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

    print(f"Documentos: {len(df)}")
    print(f"Arquivo: {output_file}")

    print()
    print("Colunas:")
    print(df.columns.tolist())

    print()
    print("Primeiro documento:")
    print(df.iloc[0]["text"][:1500])


if __name__ == "__main__":
    main()