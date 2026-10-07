import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompt_judge.txt"

MODELO_JUIZ = "qwen/qwen3.8-27b"

# Mantido de acordo com o limite observado no seu teste.
MAX_COMPLETION_TOKENS = 450

# Pausa entre chamadas dos testes controlados.
TEMPO_ENTRE_CHAMADAS = 35

# Número máximo de tentativas em caso de rate limit.
MAX_TENTATIVAS = 3

# Tempo de espera após um erro 429.
ESPERA_RATE_LIMIT = 65


# ==========================================================
# CARREGAR VARIÁVEIS DE AMBIENTE
# ==========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY não encontrada no arquivo .env."
    )

client = Groq(api_key=GROQ_API_KEY)


# ==========================================================
# CARREGAR PROMPT
# ==========================================================

def carregar_prompt():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de prompt não encontrado: {PROMPT_PATH}"
        )

    with open(PROMPT_PATH, "r", encoding="utf-8") as arquivo:
        return arquivo.read()


PROMPT_JUIZ = carregar_prompt()


# ==========================================================
# LIMPEZA DO JSON
# ==========================================================

def limpar_json(texto):
    """
    Remove cercas Markdown caso o modelo retorne ```json ... ```.
    """

    texto = texto.strip()

    if texto.startswith("```json"):
        texto = texto[7:]
    elif texto.startswith("```"):
        texto = texto[3:]

    if texto.endswith("```"):
        texto = texto[:-3]

    return texto.strip()


# ==========================================================
# VALIDAÇÃO DA RESPOSTA DO JUIZ
# ==========================================================

def validar_avaliacao(avaliacao):
    """
    Confere se os campos esperados estão presentes
    e se os valores retornados pelo juiz são consistentes.
    """

    campos_obrigatorios = [
        "total_claims",
        "supported_claims",
        "unsupported_claims",
        "relevance",
        "completeness",
        "correctness",
        "context_sufficient",
        "response_abstained",
        "justification",
    ]

    faltando = [
        campo
        for campo in campos_obrigatorios
        if campo not in avaliacao
    ]

    if faltando:
        raise ValueError(
            f"Campos ausentes na avaliação: {faltando}"
        )

    # ======================================================
    # VALIDAR BOOLEANOS
    # ======================================================

    if not isinstance(avaliacao["context_sufficient"], bool):
        raise ValueError(
            "context_sufficient deve ser true ou false."
        )

    if not isinstance(avaliacao["response_abstained"], bool):
        raise ValueError(
            "response_abstained deve ser true ou false."
        )

    response_abstained = avaliacao["response_abstained"]

    # ======================================================
    # VALIDAR CLAIMS
    # ======================================================

    total = avaliacao["total_claims"]
    suportadas = avaliacao["supported_claims"]
    nao_suportadas = avaliacao["unsupported_claims"]

    for nome, valor in [
        ("total_claims", total),
        ("supported_claims", suportadas),
        ("unsupported_claims", nao_suportadas),
    ]:
        if type(valor) is not int:
            raise ValueError(
                f"{nome} deve ser um inteiro. Valor recebido: {valor}"
            )

        if valor < 0:
            raise ValueError(
                f"{nome} não pode ser negativo. Valor recebido: {valor}"
            )

    if suportadas + nao_suportadas != total:
        raise ValueError(
            "Inconsistência nas claims: "
            "supported_claims + unsupported_claims "
            "deve ser igual a total_claims."
        )

    # Se a resposta apenas se absteve, não deve haver claims substantivas.
    if response_abstained and total != 0:
        raise ValueError(
            "Quando response_abstained=true em uma resposta puramente "
            "abstensiva, total_claims deve ser 0."
        )

    # ======================================================
    # VALIDAR RELEVÂNCIA, COMPLETUDE E CORREÇÃO
    # ======================================================

    campos_notas = [
        "relevance",
        "completeness",
        "correctness",
    ]

    if response_abstained:
        # Em respostas abstensivas, essas métricas são não aplicáveis.
        for campo in campos_notas:
            if avaliacao[campo] is not None:
                raise ValueError(
                    f"Quando response_abstained=true, "
                    f"{campo} deve ser null."
                )

    else:
        # Em respostas substantivas, todas devem ser inteiros de 1 a 5.
        for campo in campos_notas:
            nota = avaliacao[campo]

            if type(nota) is not int or nota < 1 or nota > 5:
                raise ValueError(
                    f"{campo} deve ser um inteiro entre 1 e 5 "
                    f"quando response_abstained=false. "
                    f"Valor recebido: {nota}"
                )

    # ======================================================
    # VALIDAR JUSTIFICATIVAS
    # ======================================================

    justificativa = avaliacao["justification"]

    if not isinstance(justificativa, dict):
        raise ValueError(
            "justification deve ser um objeto JSON."
        )

    campos_justificativa = [
        "relevance",
        "completeness",
        "correctness",
        "unsupported_claims",
    ]

    faltando_justificativa = [
        campo
        for campo in campos_justificativa
        if campo not in justificativa
    ]

    if faltando_justificativa:
        raise ValueError(
            "Campos ausentes em justification: "
            f"{faltando_justificativa}"
        )

    if not isinstance(
        justificativa["unsupported_claims"],
        list
    ):
        raise ValueError(
            "justification.unsupported_claims deve ser uma lista."
        )

    # Confere se o número de claims não suportadas textualizadas
    # é compatível com a contagem numérica, sem exigir decomposição perfeita.
    if (
        nao_suportadas == 0
        and len(justificativa["unsupported_claims"]) != 0
    ):
        raise ValueError(
            "unsupported_claims=0, mas justification.unsupported_claims "
            "não está vazia."
        )


# ==========================================================
# CÁLCULO DA ABSTENÇÃO ADEQUADA
# ==========================================================

def calcular_abstencao_adequada(avaliacao):
    """
    Calcula deterministicamente se a decisão de responder
    ou se abster foi adequada.

    Contexto suficiente + respondeu -> adequado.
    Contexto insuficiente + absteve-se -> adequado.
    Qualquer outra combinação -> inadequada.
    """

    contexto_suficiente = avaliacao["context_sufficient"]
    houve_abstencao = avaliacao["response_abstained"]

    appropriate_abstention = (
        (not contexto_suficiente and houve_abstencao)
        or
        (contexto_suficiente and not houve_abstencao)
    )

    avaliacao["appropriate_abstention"] = appropriate_abstention

    return avaliacao


# ==========================================================
# CÁLCULO DAS MÉTRICAS DE CLAIMS
# ==========================================================

def calcular_metricas_claims(avaliacao):
    """
    Calcula Faithfulness e Unsupported Claim Rate
    a partir das contagens fornecidas pelo juiz.
    """

    total = avaliacao["total_claims"]
    suportadas = avaliacao["supported_claims"]
    nao_suportadas = avaliacao["unsupported_claims"]

    if total == 0:
        faithfulness = None
        unsupported_claim_rate = None
    else:
        faithfulness = suportadas / total
        unsupported_claim_rate = nao_suportadas / total

    avaliacao["faithfulness"] = faithfulness
    avaliacao["unsupported_claim_rate"] = unsupported_claim_rate

    return avaliacao


# ==========================================================
# CHAMADA À API GROQ
# ==========================================================

def chamar_juiz(entrada):
    """
    Faz a chamada ao Qwen via Groq.

    Caso a API retorne erro 429, aguarda e tenta novamente.
    """

    for tentativa in range(1, MAX_TENTATIVAS + 1):

        try:
            resultado = client.chat.completions.create(
                model=MODELO_JUIZ,
                messages=[
                    {
                        "role": "user",
                        "content": entrada,
                    }
                ],
                temperature=0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            return resultado

        except RateLimitError as erro:

            print(
                f"\nRate limit da Groq atingido "
                f"(tentativa {tentativa}/{MAX_TENTATIVAS})."
            )

            if tentativa == MAX_TENTATIVAS:
                print(
                    "\nNúmero máximo de tentativas atingido."
                )
                raise erro

            print(
                f"Aguardando {ESPERA_RATE_LIMIT} segundos "
                "antes de tentar novamente..."
            )

            time.sleep(ESPERA_RATE_LIMIT)

    raise RuntimeError(
        "A chamada ao juiz terminou sem retornar resultado."
    )


# ==========================================================
# AVALIAÇÃO
# ==========================================================

def avaliar_resposta(pergunta, contexto, resposta):

    entrada = f"""
{PROMPT_JUIZ}

==================================================
CASO A SER AVALIADO
==================================================

PERGUNTA:
{pergunta}

CONTEXTO:
{contexto}

RESPOSTA:
{resposta}
"""

    resultado = chamar_juiz(entrada)

    texto = resultado.choices[0].message.content

    texto = limpar_json(texto)

    try:
        avaliacao = json.loads(texto)

    except json.JSONDecodeError as erro:
        print("\nERRO: o juiz não retornou JSON válido.")

        print("\nResposta recebida:")
        print(texto)

        print("\nDetalhes do erro:")
        print(erro)

        return None

    try:
        validar_avaliacao(avaliacao)

    except ValueError as erro:
        print(
            "\nERRO: avaliação retornada pelo juiz é inconsistente."
        )

        print(erro)

        print("\nJSON recebido:")

        print(
            json.dumps(
                avaliacao,
                indent=2,
                ensure_ascii=False,
            )
        )

        return None

    # Calculado pelo Python, não pelo LLM.
    avaliacao = calcular_abstencao_adequada(avaliacao)

    # Calculado pelo Python, não pelo LLM.
    avaliacao = calcular_metricas_claims(avaliacao)

    return avaliacao


# ==========================================================
# TESTES CONTROLADOS — ABSTENÇÃO
# ==========================================================

if __name__ == "__main__":

    pergunta = """
Em quais situações pode ocorrer inexigibilidade de licitação?
"""

    contexto_suficiente = """
A inexigibilidade de licitação ocorre quando há inviabilidade
de competição. Essa condição deve estar adequadamente demonstrada
no processo administrativo.
"""

    contexto_insuficiente = """
O documento trata de procedimentos administrativos relativos
à contratação pública, mas não apresenta informações sobre
as hipóteses de inexigibilidade de licitação.
"""

    casos = [
        {
            "id": "caso_04_abstencao_correta",
            "contexto": contexto_insuficiente,
            "resposta": """
O contexto fornecido não apresenta informações suficientes para
responder em quais situações ocorre a inexigibilidade de licitação.
""",
        },
        {
            "id": "caso_05_abstencao_inadequada",
            "contexto": contexto_suficiente,
            "resposta": """
Não é possível responder à pergunta com base no contexto fornecido.
""",
        },
    ]

    print("\n")
    print("=" * 70)
    print("TESTE DA RUBRICA DO LLM-AS-A-JUDGE")
    print("=" * 70)

    print(f"\nModelo juiz: {MODELO_JUIZ}")
    print(f"Máximo de tokens de saída: {MAX_COMPLETION_TOKENS}")
    print(f"Número de casos: {len(casos)}")

    for i, caso in enumerate(casos, start=1):

        print("\n")
        print("=" * 70)
        print(
            f"CASO {i}/{len(casos)} - {caso['id']}"
        )
        print("=" * 70)

        avaliacao = avaliar_resposta(
            pergunta=pergunta,
            contexto=caso["contexto"],
            resposta=caso["resposta"],
        )

        if avaliacao is not None:

            print("\nRESULTADO:\n")

            print(
                json.dumps(
                    avaliacao,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        else:
            print(
                "\nO caso não pôde ser avaliado corretamente."
            )

        if i < len(casos):

            print(
                f"\nAguardando {TEMPO_ENTRE_CHAMADAS} segundos "
                "para respeitar o limite observado da API..."
            )

            time.sleep(TEMPO_ENTRE_CHAMADAS)

    print("\n")
    print("=" * 70)
    print("TESTES FINALIZADOS")
    print("=" * 70)