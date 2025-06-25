import os
import re
from enum import Enum
from anthropic import Anthropic
from langchain.output_parsers import EnumOutputParser

# utils/helpers.py


def detect_intent(question: str):
    q = question.lower()
    if "sentimiento" in q or "sentiment" in q or "tweet" in q:
        return "predict"
    elif "precisión" in q or "accuracy" in q or "métrica" in q:
        return "metrics"
    elif "confusión" in q or "confusion matrix" in q:
        return "conf_matrix"
    else:
        return "unknown"


class IntentEnum(str, Enum):
    predict = "predict"
    metrics = "metrics"
    conf_matrix = "conf_matrix"
    unknown = "unknown"


def detect_intent_claude(question: str) -> str:
    """Detect intent using Claude and return one of the allowed intents."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        "Clasifica la intención de la siguiente pregunta. "
        "Las opciones posibles son solo: predict, metrics, conf_matrix, unknown. "
        "Responde únicamente con una de esas palabras en minúsculas.\n\n"
        f"Pregunta: {question}"
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": prompt}],
    )

    parser = EnumOutputParser(enum=IntentEnum)
    text = response.content[0].text.strip()
    return parser.parse(text)


def extract_text_from_question(question: str):
    """
    Extrae el texto de un tweet desde comillas simples o dobles.
    Ej: ¿Qué sentimiento tiene el tweet "I love Mondays"? -> I love Mondays
    """
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    return match.group(1) or match.group(2) if match else question


def extract_text_from_question_claude(question: str) -> str:
    """Extrae el texto de la pregunta usando un modelo de Anthropic."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        "A partir de la siguiente pregunta, devuelve solo el texto del tweet referido, si existe. "
        "Si no se identifica ningún tweet, devuelve la pregunta tal cual.\n\n"
        f"Pregunta: {question}"
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()
