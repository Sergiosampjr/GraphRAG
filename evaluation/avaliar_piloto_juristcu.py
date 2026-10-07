import argparse
import csv
import json
import random
import time
from pathlib import Path

from avaliar_respostas import (
    MODELO_JUIZ,
    TEMPO_ENTRE_CHAMADAS,
    avaliar_resposta,
)

BASE_DIR = Path(__file__).resolve().parent.parent

RESPOSTAS_PATH = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "responses"
    / "piloto_respostas_gemini.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "evaluations"
)

OUTPUT_JSONL = OUTPUT_DIR / "piloto_avaliacoes_qwen.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "piloto_avaliacoes_qwen.csv"

SEMENTE_ORDEM = 42


def carregar_jsonl(caminho):
    registros = []

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            linha = linha.strip()

            if not linha:
                continue

            try:
                registros.append(json.loads(linha))

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
        str(registro["modelo_gerador"]),
        MODELO_JUIZ,
    )


def carregar_concluidos():
    concluidos = {}

    if not OUTPUT_JSONL.exists():
        return concluidos

    for registro in carregar_jsonl(OUTPUT_JSONL):
        chave = (
            int(registro["query_id"]),
            str(registro["arquitetura"]),
            str(registro["modelo_gerador"]),
            str(registro["modelo_juiz"]),
        )

        if registro.get("status_avaliacao") == "ok":
            concluidos[chave] = registro

    return concluidos


def salvar_jsonl_append(registro):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as arquivo:
        arquivo.write(
            json.dumps(registro, ensure_ascii=False) + "\n"
        )


def reconstruir_csv():
    if not OUTPUT_JSONL.exists():
        return

    registros = carregar_jsonl(OUTPUT_JSONL)

    linhas = []

    for r in registros:
        linhas.append(
            {
                "query_id": r.get("query_id"),
                "source": r.get("source"),
                "pergunta": r.get("pergunta"),
                "arquitetura": r.get("arquitetura"),
                "modelo_gerador": r.get("modelo_gerador"),
                "modelo_juiz": r.get("modelo_juiz"),
                "status_avaliacao": r.get("status_avaliacao"),
                "total_claims": r.get("total_claims"),
                "supported_claims": r.get("supported_claims"),
                "unsupported_claims": r.get("unsupported_claims"),
                "faithfulness": r.get("faithfulness"),
                "unsupported_claim_rate": r.get(
                    "unsupported_claim_rate"
                ),
                "relevance": r.get("relevance"),
                "completeness": r.get("completeness"),
                "correctness": r.get("correctness"),
                "context_sufficient": r.get("context_sufficient"),
                "response_abstained": r.get("response_abstained"),
                "appropriate_abstention": r.get(
                    "appropriate_abstention"
                ),
                "justification": json.dumps(
                    r.get("justification"),
                    ensure_ascii=False,
                )
                if r.get("justification") is not None
                else None,
                "erro_avaliacao": r.get("erro_avaliacao"),
            }
        )

    if not linhas:
        return

    campos = list(linhas[0].keys())

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia as respostas reais do piloto JurisTCU "
            "com o Qwen/Groq usando a rubrica v1."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Número máximo de novas avaliações nesta execução. "
            "Ex.: --limit 2"
        ),
    )

    parser.add_argument(
        "--sem-pausa",
        action="store_true",
        help=(
            "Não aplica pausa entre avaliações. "
            "Não recomendado com o limite gratuito observado."
        ),
    )

    args = parser.parse_args()

    if not RESPOSTAS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de respostas não encontrado: "
            f"{RESPOSTAS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    respostas = carregar_jsonl(RESPOSTAS_PATH)

    respostas_ok = [
        r
        for r in respostas
        if r.get("status") == "ok"
    ]

    if len(respostas_ok) != 18:
        print(
            f"ATENÇÃO: eram esperadas 18 respostas com status=ok, "
            f"mas foram encontradas {len(respostas_ok)}."
        )

    rng = random.Random(SEMENTE_ORDEM)

    ordem = list(respostas_ok)
    rng.shuffle(ordem)

    concluidos = carregar_concluidos()

    pendentes = [
        r
        for r in ordem
        if chave_registro(r) not in concluidos
    ]

    if args.limit is not None:
        pendentes = pendentes[:args.limit]

    print("=" * 80)
    print("AVALIAÇÃO DO PILOTO JURISTCU — LLM-AS-A-JUDGE")
    print("=" * 80)

    print(f"\nModelo juiz: {MODELO_JUIZ}")
    print(f"Respostas disponíveis: {len(respostas_ok)}")
    print(f"Já avaliadas com sucesso: {len(concluidos)}")
    print(
        f"Novas avaliações nesta execução: "
        f"{len(pendentes)}"
    )

    if not pendentes:
        print(
            "\nNada a fazer: todas as respostas selecionadas "
            "já foram avaliadas."
        )
        reconstruir_csv()
        return

    novos_ok = 0
    novos_erro = 0

    for numero, registro in enumerate(pendentes, start=1):
        query_id = int(registro["query_id"])
        arquitetura = str(registro["arquitetura"])

        print("\n" + "=" * 80)
        print(
            f"[{numero}/{len(pendentes)}] "
            f"query_id={query_id} | "
            f"arquitetura={arquitetura}"
        )
        print("=" * 80)

        print("\nPergunta:")
        print(registro["pergunta"])

        try:
            avaliacao = avaliar_resposta(
                pergunta=registro["pergunta"],
                contexto=registro["contexto"],
                resposta=registro["resposta"],
            )

            if avaliacao is None:
                raise RuntimeError(
                    "avaliar_resposta retornou None."
                )

            saida = {
                "query_id": query_id,
                "source": registro["source"],
                "pergunta": registro["pergunta"],
                "arquitetura": arquitetura,
                "modelo_gerador": registro["modelo_gerador"],
                "modelo_juiz": MODELO_JUIZ,
                "status_avaliacao": "ok",
                "total_claims": avaliacao["total_claims"],
                "supported_claims": avaliacao["supported_claims"],
                "unsupported_claims": avaliacao["unsupported_claims"],
                "faithfulness": avaliacao["faithfulness"],
                "unsupported_claim_rate": avaliacao[
                    "unsupported_claim_rate"
                ],
                "relevance": avaliacao["relevance"],
                "completeness": avaliacao["completeness"],
                "correctness": avaliacao["correctness"],
                "context_sufficient": avaliacao["context_sufficient"],
                "response_abstained": avaliacao["response_abstained"],
                "appropriate_abstention": avaliacao[
                    "appropriate_abstention"
                ],
                "justification": avaliacao["justification"],
                "erro_avaliacao": None,
            }

            salvar_jsonl_append(saida)
            reconstruir_csv()

            novos_ok += 1

            print("\nRESULTADO:")
            print(
                json.dumps(
                    saida,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        except Exception as erro:
            saida = {
                "query_id": query_id,
                "source": registro.get("source"),
                "pergunta": registro.get("pergunta"),
                "arquitetura": arquitetura,
                "modelo_gerador": registro.get(
                    "modelo_gerador"
                ),
                "modelo_juiz": MODELO_JUIZ,
                "status_avaliacao": "erro",
                "erro_avaliacao": (
                    f"{type(erro).__name__}: {erro}"
                ),
            }

            salvar_jsonl_append(saida)
            reconstruir_csv()

            novos_erro += 1

            print("\nERRO NA AVALIAÇÃO:")
            print(saida["erro_avaliacao"])

        if (
            numero < len(pendentes)
            and not args.sem_pausa
        ):
            print(
                f"\nAguardando {TEMPO_ENTRE_CHAMADAS} segundos "
                "para respeitar o limite observado da Groq..."
            )

            time.sleep(TEMPO_ENTRE_CHAMADAS)

    print("\n" + "=" * 80)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 80)

    print(f"\nNovas avaliações OK: {novos_ok}")
    print(f"Novos erros: {novos_erro}")

    print(f"\nJSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
