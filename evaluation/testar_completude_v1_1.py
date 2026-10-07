import json
import time

from avaliar_respostas import (
    MODELO_JUIZ,
    TEMPO_ENTRE_CHAMADAS,
    avaliar_resposta,
)


PERGUNTA = """
Explique as duas situações relativas ao falecimento do responsável:
(1) quando a citação permanece válida; e
(2) quando uma citação realizada após o falecimento é inválida.
"""


CONTEXTO = """
[DOC_A]
Quando o falecimento do responsável ocorre após o término do prazo
para encaminhamento da defesa, a citação é considerada válida,
independentemente de a defesa ter sido apresentada ou não.
Nessa situação, consideram-se satisfeitos os princípios do
contraditório e da ampla defesa.

[DOC_B]
A citação realizada após o falecimento do responsável não é válida.
Nessa hipótese, o procedimento deve ser refeito mediante citação do
espólio, representado pelo inventariante, ou dos herdeiros e legítimos
sucessores, conforme o caso.
"""


CASOS = [
    {
        "id": "resposta_completa",
        "descricao": "Resposta correta e completa",
        "resposta": """
A citação permanece válida quando o falecimento do responsável ocorre
após o término do prazo para apresentação da defesa, independentemente
de ela ter sido apresentada ou não [DOC_A].

Por outro lado, se a própria citação é realizada somente após o
falecimento do responsável, ela é inválida. Nesse caso, o procedimento
deve ser refeito com a citação do espólio, por meio do inventariante,
ou dos herdeiros ou legítimos sucessores [DOC_B].
""",
    },
    {
        "id": "resposta_incompleta",
        "descricao": "Resposta correta, fiel, mas propositalmente incompleta",
        "resposta": """
A citação permanece válida quando o falecimento do responsável ocorre
após o término do prazo para apresentação da defesa, independentemente
de ela ter sido apresentada ou não [DOC_A].
""",
    },
]


def verificar(caso_id, avaliacao):
    if avaliacao is None:
        return False, ["O juiz não retornou uma avaliação válida."]

    problemas = []

    if caso_id == "resposta_completa":
        if avaliacao["completeness"] != 5:
            problemas.append(
                "A resposta completa deveria receber completeness=5."
            )

        if avaliacao["correctness"] != 5:
            problemas.append(
                "A resposta completa deveria receber correctness=5."
            )

        if avaliacao["unsupported_claims"] != 0:
            problemas.append(
                "A resposta completa não deveria ter claims não suportadas."
            )

    elif caso_id == "resposta_incompleta":
        if avaliacao["completeness"] is None:
            problemas.append(
                "Completeness não deveria ser null neste caso."
            )
        elif avaliacao["completeness"] >= 5:
            problemas.append(
                "A resposta propositalmente incompleta deveria receber "
                "completeness < 5."
            )

        if avaliacao["correctness"] != 5:
            problemas.append(
                "A resposta incompleta é factual e sustentada; "
                "esperava-se correctness=5."
            )

        if avaliacao["unsupported_claims"] != 0:
            problemas.append(
                "A resposta incompleta não inventa informação; "
                "esperava-se unsupported_claims=0."
            )

        if avaliacao["faithfulness"] != 1.0:
            problemas.append(
                "Como todas as claims são sustentadas, "
                "esperava-se faithfulness=1.0."
            )

    return len(problemas) == 0, problemas


def main():
    print("=" * 80)
    print("TESTE CONTROLADO — COMPLETUDE DA RUBRICA v1.1")
    print("=" * 80)

    print(f"\nModelo juiz: {MODELO_JUIZ}")
    print(f"Número de casos: {len(CASOS)}")

    resultados = []

    for i, caso in enumerate(CASOS, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(CASOS)}] {caso['descricao']}")
        print("=" * 80)

        avaliacao = avaliar_resposta(
            pergunta=PERGUNTA,
            contexto=CONTEXTO,
            resposta=caso["resposta"],
        )

        print("\nRESULTADO:")
        print(
            json.dumps(
                avaliacao,
                indent=2,
                ensure_ascii=False,
            )
        )

        passou, problemas = verificar(
            caso["id"],
            avaliacao,
        )

        resultados.append(
            {
                "id": caso["id"],
                "passou": passou,
                "problemas": problemas,
            }
        )

        print("\nVERIFICAÇÃO AUTOMÁTICA:")
        if passou:
            print("PASSOU ✅")
        else:
            print("NÃO PASSOU ❌")
            for problema in problemas:
                print(f"- {problema}")

        if i < len(CASOS):
            print(
                f"\nAguardando {TEMPO_ENTRE_CHAMADAS} segundos "
                "para respeitar o limite observado da Groq..."
            )
            time.sleep(TEMPO_ENTRE_CHAMADAS)

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)

    for resultado in resultados:
        status = "PASSOU ✅" if resultado["passou"] else "NÃO PASSOU ❌"
        print(f"{resultado['id']}: {status}")

    todos_passaram = all(
        resultado["passou"]
        for resultado in resultados
    )

    print("\nRESULTADO FINAL:")
    if todos_passaram:
        print(
            "RUBRICA v1.1 PASSOU NO TESTE CONTROLADO DE COMPLETUDE ✅"
        )
        print(
            "A resposta completa recebeu tratamento de resposta completa "
            "e a resposta parcial foi penalizada apenas na completude."
        )
    else:
        print(
            "A RUBRICA v1.1 AINDA PRECISA DE REVISÃO ❌"
        )
        print(
            "Analise os resultados acima antes de congelar a rubrica."
        )


if __name__ == "__main__":
    main()
