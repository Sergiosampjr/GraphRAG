
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Permite importar o avaliador existente.
EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent

sys.path.insert(0, str(EVALUATION_DIR))

from avaliar_respostas import (
    avaliar_resposta,
    MODELO_JUIZ,
    TEMPO_ENTRE_CHAMADAS,
)

# ==========================================================
# ARQUIVOS
# ==========================================================

RESPOSTAS_PATH = (
    PROJECT_DIR
    / "outputs/generation/responses"
    / "experimento_respostas_gemma3_4b_it_validas.csv"
)

CONTEXTOS_PATH = (
    PROJECT_DIR
    / "outputs/generation/contexts"
    / "experimento_contextos.jsonl"
)

SAIDA_DIR = PROJECT_DIR / "outputs/evaluation"

CHECKPOINT_PATH = (
    SAIDA_DIR / "avaliacoes_gemma_qwen.jsonl"
)

ARQUITETURAS = {"rag", "graphrag", "hibrido"}


def chave(query_id, arquitetura):
    return (
        int(query_id),
        str(arquitetura).strip().lower(),
    )


# ==========================================================
# CARREGAR CONTEXTOS
# ==========================================================

def carregar_contextos():
    contextos = {}

    with CONTEXTOS_PATH.open(encoding="utf-8") as f:
        for numero_linha, linha in enumerate(f, 1):
            if not linha.strip():
                continue

            registro = json.loads(linha)

            identificador = chave(
                registro["query_id"],
                registro["arquitetura"],
            )

            if identificador in contextos:
                raise ValueError(
                    f"Contexto duplicado na linha {numero_linha}: "
                    f"{identificador}"
                )

            contextos[identificador] = registro

    return contextos


# ==========================================================
# CARREGAR CHECKPOINT
# ==========================================================

def carregar_checkpoint():
    concluidos = set()

    if not CHECKPOINT_PATH.exists():
        return concluidos

    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue

            registro = json.loads(linha)

            if registro.get("status") != "ok":
                continue

            identificador = chave(
                registro["query_id"],
                registro["arquitetura"],
            )

            if identificador in concluidos:
                raise ValueError(
                    f"Avaliação duplicada no checkpoint: "
                    f"{identificador}"
                )

            if registro.get("modelo_juiz") != MODELO_JUIZ:
                raise ValueError(
                    "Checkpoint pertence a outro modelo juiz."
                )

            concluidos.add(identificador)

    return concluidos


# ==========================================================
# VALIDAR ENTRADAS
# ==========================================================

def carregar_respostas(contextos):
    df = pd.read_csv(RESPOSTAS_PATH)

    obrigatorias = {
        "query_id",
        "arquitetura",
        "pergunta",
        "resposta",
        "status",
    }

    faltando = obrigatorias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes no CSV: {sorted(faltando)}"
        )

    df["arquitetura"] = (
        df["arquitetura"].astype(str).str.strip().str.lower()
    )

    df["status"] = (
        df["status"].astype(str).str.strip().str.lower()
    )

    if len(df) != 450 or not df["status"].eq("ok").all():
        raise ValueError(
            "O CSV deve conter exatamente 450 respostas OK."
        )

    if set(df["arquitetura"]) != ARQUITETURAS:
        raise ValueError("Arquiteturas inesperadas.")

    if df.duplicated(["query_id", "arquitetura"]).any():
        raise ValueError("Existem respostas duplicadas.")

    if df["pergunta"].isna().any():
        raise ValueError("Existem perguntas vazias.")

    if df["resposta"].isna().any():
        raise ValueError("Existem respostas vazias.")

    if (df["resposta"].astype(str).str.strip() == "").any():
        raise ValueError("Existem respostas em branco.")

    chaves_respostas = {
        chave(r.query_id, r.arquitetura)
        for r in df.itertuples(index=False)
    }

    if chaves_respostas != set(contextos):
        raise ValueError(
            "As combinações de consultas e arquiteturas "
            "não coincidem entre CSV e JSONL."
        )

    for r in df.itertuples(index=False):
        identificador = chave(r.query_id, r.arquitetura)
        contexto = contextos[identificador]

        if str(r.pergunta).strip() != str(
            contexto["pergunta"]
        ).strip():
            raise ValueError(
                f"Pergunta divergente: {identificador}"
            )

        if not str(contexto.get("contexto", "")).strip():
            raise ValueError(
                f"Contexto vazio: {identificador}"
            )

    contagens = df["arquitetura"].value_counts()

    if any(contagens.get(a, 0) != 150 for a in ARQUITETURAS):
        raise ValueError(
            "Esperadas 150 respostas por arquitetura."
        )

    return df


# ==========================================================
# SALVAR RESULTADO
# ==========================================================

def salvar_avaliacao(registro):
    with CHECKPOINT_PATH.open(
        "a", encoding="utf-8"
    ) as arquivo:
        arquivo.write(
            json.dumps(registro, ensure_ascii=False) + "\n"
        )
        arquivo.flush()


# ==========================================================
# EXECUÇÃO
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limite",
        type=int,
        default=3,
        help="Número máximo de novas avaliações. "
             "Use 0 para avaliar todas as pendentes.",
    )

    args = parser.parse_args()

    if args.limite < 0:
        parser.error("--limite não pode ser negativo.")

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    contextos = carregar_contextos()
    respostas = carregar_respostas(contextos)
    concluidos = carregar_checkpoint()

    pendentes = []

    for r in respostas.itertuples(index=False):
        identificador = chave(r.query_id, r.arquitetura)

        if identificador not in concluidos:
            pendentes.append(r)

    print("=" * 65)
    print("AVALIAÇÃO DAS RESPOSTAS DO GEMMA")
    print("=" * 65)

    print("Modelo juiz:", MODELO_JUIZ)
    print("Respostas:", len(respostas))
    print("Contextos:", len(contextos))
    print("Avaliações concluídas:", len(concluidos))
    print("Avaliações pendentes:", len(pendentes))

    if args.limite > 0:
        pendentes = pendentes[:args.limite]

    print("Avaliações nesta execução:", len(pendentes))

    for indice, r in enumerate(pendentes, 1):
        identificador = chave(r.query_id, r.arquitetura)

        contexto = contextos[identificador]

        print("\n" + "=" * 65)
        print(
            f"[{indice}/{len(pendentes)}] "
            f"query_id={r.query_id} | "
            f"arquitetura={r.arquitetura}"
        )
        print("=" * 65)

        try:
            avaliacao = avaliar_resposta(
                pergunta=str(r.pergunta),
                contexto=str(contexto["contexto"]),
                resposta=str(r.resposta),
            )

            if avaliacao is None:
                print(
                    "Avaliação inválida. "
                    "Execução interrompida sem marcar "
                    "o caso como concluído."
                )
                break

            registro = {
                "query_id": int(r.query_id),
                "arquitetura": r.arquitetura,
                "modelo_gerador": str(r.modelo_gerador),
                "modelo_juiz": MODELO_JUIZ,
                "status": "ok",
                "avaliacao": avaliacao,
            }

            salvar_avaliacao(registro)
            concluidos.add(identificador)

            print(
                "Avaliação salva com sucesso."
            )

            print(
                "Faithfulness:",
                avaliacao.get("faithfulness"),
            )

            print(
                "Unsupported Claim Rate:",
                avaliacao.get("unsupported_claim_rate"),
            )

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
            break

        except Exception as erro:
            print(
                "\nErro durante a avaliação:",
                type(erro).__name__,
                str(erro),
            )
            print(
                "Execução interrompida. "
                "As avaliações anteriores foram preservadas."
            )
            break

        if indice < len(pendentes):
            time.sleep(TEMPO_ENTRE_CHAMADAS)

    print("\n" + "=" * 65)
    print("EXECUÇÃO FINALIZADA")
    print("Avaliações salvas:", len(concluidos))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("=" * 65)


if __name__ == "__main__":
    main()
