import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RESPOSTAS_PATH = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "responses"
    / "piloto_respostas_gemini.jsonl"
)

AVALIACOES_PATH = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "evaluations"
    / "piloto_avaliacoes_qwen.jsonl"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "generation"
    / "evaluations"
    / "auditoria_piloto.md"
)


def carregar_jsonl(caminho):
    registros = []

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))

    return registros


def chave(registro):
    return (
        int(registro["query_id"]),
        str(registro["arquitetura"]),
        str(registro["modelo_gerador"]),
    )


def main():
    respostas = carregar_jsonl(RESPOSTAS_PATH)
    avaliacoes = carregar_jsonl(AVALIACOES_PATH)

    respostas_ok = {
        chave(r): r
        for r in respostas
        if r.get("status") == "ok"
    }

    # Mantém apenas a última avaliação OK para cada combinação.
    avaliacoes_ok = {}

    for a in avaliacoes:
        if a.get("status_avaliacao") != "ok":
            continue

        avaliacoes_ok[chave(a)] = a

    faltando = [
        k
        for k in avaliacoes_ok
        if k not in respostas_ok
    ]

    if faltando:
        raise ValueError(
            "Há avaliações sem resposta correspondente: "
            + str(faltando)
        )

    casos = []

    for k, avaliacao in avaliacoes_ok.items():
        resposta = respostas_ok[k]

        casos.append(
            {
                "query_id": int(avaliacao["query_id"]),
                "source": avaliacao["source"],
                "arquitetura": avaliacao["arquitetura"],
                "modelo_gerador": avaliacao["modelo_gerador"],
                "modelo_juiz": avaliacao["modelo_juiz"],
                "pergunta": avaliacao["pergunta"],
                "contexto": resposta["contexto"],
                "resposta": resposta["resposta"],
                "avaliacao": avaliacao,
            }
        )

    # Primeira auditoria: os casos mais informativos,
    # isto é, qualquer nota abaixo de 5 ou claim não suportada.
    selecionados = [
        c
        for c in casos
        if (
            c["avaliacao"].get("relevance") != 5
            or c["avaliacao"].get("completeness") != 5
            or c["avaliacao"].get("correctness") != 5
            or c["avaliacao"].get("unsupported_claims", 0) > 0
            or not c["avaliacao"].get(
                "appropriate_abstention",
                False,
            )
        )
    ]

    # Se não houver casos problemáticos, escolhe um por arquitetura.
    if not selecionados:
        vistos = set()

        for c in sorted(
            casos,
            key=lambda x: (
                x["arquitetura"],
                x["query_id"],
            ),
        ):
            if c["arquitetura"] not in vistos:
                selecionados.append(c)
                vistos.add(c["arquitetura"])

    selecionados = sorted(
        selecionados,
        key=lambda x: (
            x["query_id"],
            x["arquitetura"],
        ),
    )

    linhas = []

    linhas.append("# Auditoria manual do piloto JurisTCU\n")
    linhas.append(
        "Este relatório reúne os casos mais informativos do piloto "
        "para inspeção humana. O objetivo é verificar se as notas "
        "atribuídas pelo LLM-as-a-Judge são coerentes com a pergunta, "
        "o contexto realmente fornecido ao gerador e a resposta gerada.\n"
    )

    linhas.append(
        f"- Avaliações OK disponíveis: **{len(casos)}**\n"
    )
    linhas.append(
        f"- Casos selecionados para auditoria: "
        f"**{len(selecionados)}**\n"
    )

    for numero, caso in enumerate(selecionados, start=1):
        a = caso["avaliacao"]

        linhas.append("\n---\n")
        linhas.append(
            f"\n## Caso {numero} — query_id={caso['query_id']} "
            f"— {caso['arquitetura']}\n"
        )

        linhas.append(
            f"\n**Fonte da consulta:** {caso['source']}\n"
        )
        linhas.append(
            f"\n**Gerador:** {caso['modelo_gerador']}\n"
        )
        linhas.append(
            f"\n**Juiz:** {caso['modelo_juiz']}\n"
        )

        linhas.append("\n### Pergunta\n")
        linhas.append(
            f"\n{caso['pergunta']}\n"
        )

        linhas.append("\n### Contexto fornecido ao Gemini\n")
        linhas.append("\n```text\n")
        linhas.append(caso["contexto"])
        linhas.append("\n```\n")

        linhas.append("\n### Resposta do Gemini\n")
        linhas.append(
            f"\n{caso['resposta']}\n"
        )

        linhas.append("\n### Avaliação do Qwen\n")
        linhas.append(
            f"\n- Relevance: **{a.get('relevance')}**\n"
        )
        linhas.append(
            f"- Completeness: **{a.get('completeness')}**\n"
        )
        linhas.append(
            f"- Correctness: **{a.get('correctness')}**\n"
        )
        linhas.append(
            f"- Total claims: **{a.get('total_claims')}**\n"
        )
        linhas.append(
            f"- Supported claims: **{a.get('supported_claims')}**\n"
        )
        linhas.append(
            f"- Unsupported claims: **{a.get('unsupported_claims')}**\n"
        )
        linhas.append(
            f"- Faithfulness: **{a.get('faithfulness')}**\n"
        )
        linhas.append(
            f"- Unsupported claim rate: "
            f"**{a.get('unsupported_claim_rate')}**\n"
        )
        linhas.append(
            f"- Context sufficient: "
            f"**{a.get('context_sufficient')}**\n"
        )
        linhas.append(
            f"- Response abstained: "
            f"**{a.get('response_abstained')}**\n"
        )
        linhas.append(
            f"- Appropriate abstention: "
            f"**{a.get('appropriate_abstention')}**\n"
        )

        linhas.append("\n### Justificativa do juiz\n")

        justificativa = a.get("justification") or {}

        for campo in [
            "relevance",
            "completeness",
            "correctness",
        ]:
            linhas.append(
                f"\n**{campo}:** "
                f"{justificativa.get(campo)}\n"
            )

        linhas.append(
            "\n**Unsupported claims identificadas:**\n"
        )

        unsupported = justificativa.get(
            "unsupported_claims",
            [],
        )

        if unsupported:
            for item in unsupported:
                linhas.append(f"\n- {item}\n")
        else:
            linhas.append("\n- Nenhuma.\n")

        linhas.append("\n### Checklist humano\n")
        linhas.append(
            "\n- [ ] A resposta realmente responde à pergunta?\n"
        )
        linhas.append(
            "- [ ] As afirmações estão sustentadas pelo contexto?\n"
        )
        linhas.append(
            "- [ ] Existe alguma informação inventada ou extrapolada?\n"
        )
        linhas.append(
            "- [ ] A nota de relevância parece justa?\n"
        )
        linhas.append(
            "- [ ] A nota de completude parece justa?\n"
        )
        linhas.append(
            "- [ ] A nota de correção parece justa?\n"
        )
        linhas.append(
            "- [ ] A decisão de responder/abster-se foi adequada?\n"
        )
        linhas.append(
            "- [ ] Eu manteria a avaliação do Qwen? "
            "Sim / Não / Parcialmente\n"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "".join(linhas),
        encoding="utf-8",
    )

    print("=" * 80)
    print("AUDITORIA DO PILOTO")
    print("=" * 80)

    print(
        f"\nAvaliações OK: {len(casos)}"
    )
    print(
        f"Casos selecionados: {len(selecionados)}"
    )

    print("\nSelecionados:")
    for c in selecionados:
        a = c["avaliacao"]
        print(
            f"- query_id={c['query_id']} | "
            f"{c['arquitetura']} | "
            f"R={a.get('relevance')} "
            f"C={a.get('completeness')} "
            f"Corr={a.get('correctness')} | "
            f"unsupported={a.get('unsupported_claims')}"
        )

    print(
        f"\nRelatório salvo em:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
