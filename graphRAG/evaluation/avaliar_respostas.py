import os
import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompt_judge.txt"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY não encontrada no arquivo .env."
    )

client = Groq(api_key=GROQ_API_KEY)


# IMPORTANTE:
# coloque aqui EXATAMENTE o mesmo modelo que funcionou
# no seu teste_groq.py.
MODELO_JUIZ = "qwen/qwen3.8-27b"


# ==========================================================
# CARREGAR PROMPT
# ==========================================================

def carregar_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


PROMPT_JUIZ = carregar_prompt()


# ==========================================================
# LIMPEZA DO JSON
# ==========================================================

def limpar_json(texto):
    texto = texto.strip()

    if texto.startswith("```json"):
        texto = texto[7:]

    elif texto.startswith("```"):
        texto = texto[3:]

    if texto.endswith("```"):
        texto = texto[:-3]

    return texto.strip()


# ==========================================================
# CÁLCULO DAS MÉTRICAS
# ==========================================================

def calcular_metricas_claims(avaliacao):
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

    resultado = client.chat.completions.create(
        model=MODELO_JUIZ,
        messages=[
            {
                "role": "user",
                "content": entrada
            }
        ],
        temperature=0
    )

    texto = resultado.choices[0].message.content

    texto = limpar_json(texto)

    try:
        avaliacao = json.loads(texto)

    except json.JSONDecodeError:
        print("\nERRO: o juiz não retornou JSON válido.")
        print("\nResposta recebida:")
        print(texto)
        return None

    avaliacao = calcular_metricas_claims(avaliacao)

    return avaliacao


# ==========================================================
# TESTE
# ==========================================================

if __name__ == "__main__":

    pergunta = """
Em quais situações pode ocorrer inexigibilidade de licitação?
"""

    contexto = """
A inexigibilidade de licitação ocorre quando há inviabilidade
de competição, conforme as hipóteses previstas na legislação
aplicável. A caracterização deve estar adequadamente demonstrada
no processo administrativo.
"""

    resposta = """
A inexigibilidade pode ocorrer quando existe inviabilidade de
competição. A Administração deve demonstrar essa condição no
processo administrativo.
"""

    avaliacao = avaliar_resposta(
        pergunta=pergunta,
        contexto=contexto,
        resposta=resposta
    )

    if avaliacao is not None:

        print("\n==============================")
        print("AVALIAÇÃO")
        print("==============================\n")

        print(
            json.dumps(
                avaliacao,
                indent=2,
                ensure_ascii=False
            )
        )