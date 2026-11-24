
import os
import time
import traceback

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "A variável GEMINI_API_KEY não está configurada."
    )

MODELO = "gemini-3.6-flash"

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=30000,  # 30 segundos
    ),
)


# ============================================================
# TESTE
# ============================================================

pergunta = "O que é inexigibilidade de licitação?"

print("=" * 80, flush=True)
print("TESTE GEMINI", flush=True)
print("=" * 80, flush=True)

print(f"\nModelo: {MODELO}", flush=True)
print(f"Pergunta: {pergunta}", flush=True)

inicio = time.perf_counter()

try:

    print(
        "\n[1] Enviando requisição...",
        flush=True,
    )

    resposta = client.models.generate_content(
    model=MODELO,
    contents=pergunta,
    config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=1024,
    ),
)

    latencia = time.perf_counter() - inicio

    print(
        f"\n[2] Requisição finalizada em {latencia:.2f}s",
        flush=True,
    )

    texto = resposta.text

    if texto and texto.strip():

        print("\n[3] RESPOSTA:", flush=True)
        print(texto, flush=True)

    else:

        print(
            "\n[3] A API respondeu, mas não retornou texto.",
            flush=True,
        )

        print(
            "Resposta bruta:",
            resposta,
            flush=True,
        )

except Exception as erro:

    latencia = time.perf_counter() - inicio

    print("\n" + "=" * 80)
    print("ERRO NA API GEMINI")
    print("=" * 80)

    print(
        f"\nTempo até o erro: {latencia:.2f}s"
    )

    print(
        "Tipo:",
        type(erro).__name__,
    )

    print(
        "Mensagem:",
        str(erro),
    )

    mensagem = str(erro).lower()

    if (
        "429" in mensagem
        or "resource_exhausted" in mensagem
        or "rate limit" in mensagem
    ):

        print(
            "\nDIAGNÓSTICO: limite de requisições "
            "ou cota da API atingido."
        )

    elif (
        "404" in mensagem
        or "not found" in mensagem
    ):

        print(
            "\nDIAGNÓSTICO: modelo não encontrado "
            "ou indisponível nesta API."
        )

    elif (
        "401" in mensagem
        or "403" in mensagem
        or "permission" in mensagem
    ):

        print(
            "\nDIAGNÓSTICO: possível problema "
            "de autenticação ou permissão."
        )

    elif (
        "timeout" in mensagem
        or "timed out" in mensagem
    ):

        print(
            "\nDIAGNÓSTICO: tempo limite "
            "da requisição excedido."
        )

    print("\nTRACEBACK:")
    traceback.print_exc()

finally:

    print("\n" + "=" * 80)
    print("TESTE FINALIZADO")
    print("=" * 80)
