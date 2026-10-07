import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CONTEXTOS_PATH = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "contexts"
    / "experimento_contextos.jsonl"
)

PROMPT_PATH = (
    BASE_DIR
    / "generation"
    / "prompt_generator.txt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "responses"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "experimento_respostas_gemini.jsonl"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "experimento_respostas_gemini.csv"
)

NUM_CONTEXTOS_ESPERADO = 450
SEMENTE_ORDEM = 42
MAX_TENTATIVAS = 4

# Pausa conservadora entre chamadas.
# Não representa uma cota oficial do provedor.
PAUSA_ENTRE_CHAMADAS = 5

# Backoff exponencial em falhas temporárias.
ESPERA_INICIAL_ERRO = 15


# ==========================================================
# AMBIENTE
# ==========================================================

# Carrega .env ANTES de ler GEMINI_MODEL / GEMINI_API_KEY.
load_dotenv()

MODELO_GERADOR = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY não encontrada no arquivo .env."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================================
# UTILITÁRIOS
# ==========================================================

def carregar_prompt():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt do gerador não encontrado: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


PROMPT_TEMPLATE = carregar_prompt()

PROMPT_SHA256 = hashlib.sha256(
    PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


def carregar_jsonl(caminho):
    registros = []

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(
            arquivo,
            start=1,
        ):
            linha = linha.strip()

            if not linha:
                continue

            try:
                registros.append(
                    json.loads(linha)
                )
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f"JSON inválido em {caminho}, "
                    f"linha {numero_linha}: {erro}"
                )

    return registros


def chave_registro(registro):
    return (
        int(registro["query_id"]),
        str(registro["arquitetura"]),
        MODELO_GERADOR,
    )


def carregar_concluidos():
    concluidos = {}

    if not OUTPUT_JSONL.exists():
        return concluidos

    for registro in carregar_jsonl(
        OUTPUT_JSONL
    ):
        chave = (
            int(registro["query_id"]),
            str(registro["arquitetura"]),
            str(registro["modelo_gerador"]),
        )

        if registro.get("status") == "ok":
            concluidos[chave] = registro

    return concluidos


def montar_prompt(registro):
    return PROMPT_TEMPLATE.format(
        pergunta=registro["pergunta"],
        contexto=registro["contexto"],
    )


def hash_contexto(contexto):
    return hashlib.sha256(
        str(contexto).encode("utf-8")
    ).hexdigest()


def extrair_output_text(interaction):
    texto = getattr(
        interaction,
        "output_text",
        None,
    )

    if texto is None:
        raise RuntimeError(
            "A resposta do Gemini não contém output_text."
        )

    texto = str(texto).strip()

    if not texto:
        raise RuntimeError(
            "O Gemini retornou resposta vazia."
        )

    return texto


def chamar_gemini(prompt):
    ultimo_erro = None

    for tentativa in range(
        1,
        MAX_TENTATIVAS + 1,
    ):
        inicio = time.perf_counter()

        try:
            interaction = (
                client.interactions.create(
                    model=MODELO_GERADOR,
                    input=prompt,
                )
            )

            latencia = (
                time.perf_counter()
                - inicio
            )

            resposta = extrair_output_text(
                interaction
            )

            return {
                "resposta": resposta,
                "latencia_segundos": latencia,
                "tentativas": tentativa,
            }

        except Exception as erro:
            ultimo_erro = erro

            if tentativa == MAX_TENTATIVAS:
                break

            espera = (
                ESPERA_INICIAL_ERRO
                * (2 ** (tentativa - 1))
            )

            print(
                f"\nFalha na chamada "
                f"(tentativa {tentativa}/"
                f"{MAX_TENTATIVAS}): "
                f"{type(erro).__name__}: {erro}"
            )
            print(
                f"Aguardando {espera}s..."
            )

            time.sleep(espera)

    raise RuntimeError(
        "Falha após todas as tentativas: "
        f"{type(ultimo_erro).__name__}: "
        f"{ultimo_erro}"
    )


def salvar_jsonl_append(registro):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSONL,
        "a",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write(
            json.dumps(
                registro,
                ensure_ascii=False,
            )
            + "\n"
        )


def reconstruir_csv():
    if not OUTPUT_JSONL.exists():
        return

    registros = carregar_jsonl(
        OUTPUT_JSONL
    )

    linhas = []

    for r in registros:
        linhas.append(
            {
                "query_id": r.get("query_id"),
                "source": r.get("source"),
                "pergunta": r.get("pergunta"),
                "arquitetura": r.get(
                    "arquitetura"
                ),
                "modelo_gerador": r.get(
                    "modelo_gerador"
                ),
                "prompt_sha256": r.get(
                    "prompt_sha256"
                ),
                "contexto_sha256": r.get(
                    "contexto_sha256"
                ),
                "status": r.get("status"),
                "latencia_segundos": r.get(
                    "latencia_segundos"
                ),
                "tentativas": r.get(
                    "tentativas"
                ),
                "num_documentos": r.get(
                    "num_documentos"
                ),
                "total_tokens_chunks": r.get(
                    "total_tokens_chunks"
                ),
                "doc_ids": json.dumps(
                    r.get("doc_ids", [])
                ),
                "chunk_ids": json.dumps(
                    r.get("chunk_ids", [])
                ),
                "resposta": r.get(
                    "resposta"
                ),
                "erro": r.get(
                    "erro"
                ),
            }
        )

    if not linhas:
        return

    campos = list(
        linhas[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos,
        )

        writer.writeheader()
        writer.writerows(linhas)


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera respostas Gemini para os 450 "
            "contextos do experimento principal JurisTCU."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Número máximo de novas respostas "
            "a gerar nesta execução. "
            "Ex.: --limit 5"
        ),
    )

    parser.add_argument(
        "--sem-pausa",
        action="store_true",
        help=(
            "Não aplica a pausa entre chamadas. "
            "Não recomendado para o experimento normal."
        ),
    )

    args = parser.parse_args()

    if not CONTEXTOS_PATH.exists():
        raise FileNotFoundError(
            f"Contextos não encontrados: "
            f"{CONTEXTOS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros = carregar_jsonl(
        CONTEXTOS_PATH
    )

    if len(registros) != NUM_CONTEXTOS_ESPERADO:
        raise ValueError(
            f"Eram esperados {NUM_CONTEXTOS_ESPERADO} "
            f"contextos, mas foram encontrados "
            f"{len(registros)}."
        )

    # Valida unicidade dos 450 pares.
    chaves_contexto = {
        (
            int(r["query_id"]),
            str(r["arquitetura"]),
        )
        for r in registros
    }

    if len(chaves_contexto) != NUM_CONTEXTOS_ESPERADO:
        raise ValueError(
            "Há duplicatas em "
            "(query_id, arquitetura). "
            f"Chaves únicas: {len(chaves_contexto)}."
        )

    # Ordem fixa e embaralhada para evitar que
    # uma arquitetura seja sempre chamada primeiro.
    rng = random.Random(
        SEMENTE_ORDEM
    )

    ordem = list(registros)
    rng.shuffle(ordem)

    concluidos = carregar_concluidos()

    pendentes = [
        r
        for r in ordem
        if chave_registro(r)
        not in concluidos
    ]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit deve ser maior que zero."
            )

        pendentes = pendentes[
            :args.limit
        ]

    print("=" * 80)
    print(
        "GERAÇÃO DE RESPOSTAS — "
        "EXPERIMENTO PRINCIPAL JURISTCU"
    )
    print("=" * 80)

    print(
        f"\nModelo gerador: "
        f"{MODELO_GERADOR}"
    )
    print(
        f"Prompt SHA-256: "
        f"{PROMPT_SHA256}"
    )
    print(
        f"Contextos disponíveis: "
        f"{len(registros)}"
    )
    print(
        f"Já concluídos com sucesso: "
        f"{len(concluidos)}"
    )
    print(
        f"Novas chamadas nesta execução: "
        f"{len(pendentes)}"
    )

    if not pendentes:
        print(
            "\nNada a fazer: todos os "
            "registros selecionados já "
            "foram concluídos."
        )
        reconstruir_csv()
        return

    novos_ok = 0
    novos_erro = 0

    for numero, registro in enumerate(
        pendentes,
        start=1,
    ):
        query_id = int(
            registro["query_id"]
        )
        arquitetura = str(
            registro["arquitetura"]
        )

        print(
            "\n"
            + "=" * 80
        )
        print(
            f"[{numero}/{len(pendentes)}] "
            f"query_id={query_id} | "
            f"arquitetura={arquitetura}"
        )
        print("=" * 80)

        prompt = montar_prompt(
            registro
        )

        resultado_base = {
            "query_id": query_id,
            "source": registro[
                "source"
            ],
            "pergunta": registro[
                "pergunta"
            ],
            "arquitetura": arquitetura,
            "modelo_gerador": (
                MODELO_GERADOR
            ),
            "prompt_sha256": (
                PROMPT_SHA256
            ),
            "contexto_sha256": (
                hash_contexto(
                    registro["contexto"]
                )
            ),
            "num_documentos": registro[
                "num_documentos"
            ],
            "total_tokens_chunks": registro[
                "total_tokens_chunks"
            ],
            "doc_ids": registro[
                "doc_ids"
            ],
            "chunk_ids": registro[
                "chunk_ids"
            ],
            # Guarda exatamente o contexto enviado.
            "contexto": registro[
                "contexto"
            ],
        }

        try:
            resultado_api = chamar_gemini(
                prompt
            )

            saida = {
                **resultado_base,
                "status": "ok",
                "resposta": resultado_api[
                    "resposta"
                ],
                "latencia_segundos": (
                    resultado_api[
                        "latencia_segundos"
                    ]
                ),
                "tentativas": (
                    resultado_api[
                        "tentativas"
                    ]
                ),
                "erro": None,
            }

            salvar_jsonl_append(
                saida
            )

            novos_ok += 1

            print(
                f"OK | "
                f"{saida['latencia_segundos']:.2f}s"
            )

            print(
                "\nResposta:\n"
            )
            print(
                saida["resposta"]
            )

        except Exception as erro:
            saida = {
                **resultado_base,
                "status": "erro",
                "resposta": None,
                "latencia_segundos": None,
                "tentativas": (
                    MAX_TENTATIVAS
                ),
                "erro": (
                    f"{type(erro).__name__}: "
                    f"{erro}"
                ),
            }

            salvar_jsonl_append(
                saida
            )

            novos_erro += 1

            print(
                "\nERRO definitivo:"
            )
            print(
                saida["erro"]
            )

        reconstruir_csv()

        if (
            numero < len(pendentes)
            and not args.sem_pausa
        ):
            print(
                f"\nAguardando "
                f"{PAUSA_ENTRE_CHAMADAS}s..."
            )
            time.sleep(
                PAUSA_ENTRE_CHAMADAS
            )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "EXECUÇÃO FINALIZADA"
    )
    print("=" * 80)

    print(
        f"\nNovas respostas OK: "
        f"{novos_ok}"
    )
    print(
        f"Novos erros: "
        f"{novos_erro}"
    )

    concluidos_finais = carregar_concluidos()

    print(
        f"Total concluído com sucesso: "
        f"{len(concluidos_finais)}/"
        f"{NUM_CONTEXTOS_ESPERADO}"
    )

    print(
        f"\nJSONL: {OUTPUT_JSONL}"
    )
    print(
        f"CSV:   {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
