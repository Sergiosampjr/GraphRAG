import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "A variável GEMINI_API_KEY não está configurada."
    )

client = genai.Client()

MODELO = "gemini-3.6-flash"


# ============================================================
# TESTE
# ============================================================

pergunta = "O que é inexigibilidade de licitação?"


print("=" * 80)
print("TESTE GEMINI")
print("=" * 80)

print(f"\nModelo: {MODELO}")

print("\nPergunta:")
print(pergunta)

print("\nEnviando requisição...")


interaction = client.interactions.create(
    model=MODELO,
    input=pergunta,
)


# ============================================================
# RESULTADO
# ============================================================

print("\nResposta:")
print(interaction.output_text)

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO")
print("=" * 80)