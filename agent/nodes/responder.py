# agent/nodes/responder.py

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

# Inicializa el cliente de Claude usando tu API Key
client = Anthropic(
    api_key=os.getenv("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")  # Asegúrate de tener esta variable en tu entorno
)

def generate_response(question: str, result: str) -> str:
    """
    Genera una respuesta en lenguaje natural basada en la pregunta del usuario y el resultado del análisis.

    Args:
        question (str): Pregunta original del usuario.
        result (str): Resultado devuelto por el modelo (predicción, métricas o matriz de confusión).

    Returns:
        str: Respuesta redactada por el LLM.
    """
    prompt = (
        f"{HUMAN_PROMPT} A continuación tienes una pregunta y su resultado:\n\n"
        f"Pregunta: {question}\n"
        f"Resultado: {result}\n\n"
        "Redacta una respuesta útil, clara y humana para el usuario final."
        f"{AI_PROMPT}"
    )

    response = client.completions.create(
        model="claude-3-sonnet-20240229",
        max_tokens_to_sample=300,
        prompt=prompt
    )

    return response.completion.strip()
