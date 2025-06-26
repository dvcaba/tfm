import os
import re
from enum import Enum
from anthropic import Anthropic
from langchain.output_parsers import EnumOutputParser
from langchain_core.exceptions import OutputParserException

# Si usas .env para cargar tu clave de Anthropic, descomenta:
# from dotenv import load_dotenv
# load_dotenv()

# Enum de intenciones válidas
class IntentEnum(str, Enum):
    predict = "predict"
    metrics = "metrics"
    conf_matrix = "conf_matrix"
    unknown = "unknown"

# Heurística simple local como respaldo
def detect_intent(question: str) -> str:
    q = question.lower()
    if "sentimiento" in q or "sentiment" in q or "tweet" in q:
        return "predict"
    elif "precisión" in q or "accuracy" in q or "métrica" in q:
        return "metrics"
    elif "confusión" in q or "confusion matrix" in q:
        return "conf_matrix"
    else:
        return "unknown"

# Detección de intención usando Claude, con control robusto de errores
def detect_intent_claude(question: str) -> str:
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        prompt = (
            "Clasifica la intención de la siguiente pregunta.\n"
            "Las únicas respuestas posibles son exactamente: predict, metrics, conf_matrix, unknown.\n"
            "Responde solo con una palabra, sin explicaciones.\n\n"
            f"Pregunta: {question}"
        )

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip().lower()

        # Corrección de respuestas comunes mal formadas
        intent_map = {
            "conf": "conf_matrix",
            "confusion": "conf_matrix",
            "prec": "metrics",
            "acc": "metrics",
        }
        text = intent_map.get(text, text)

        parser = EnumOutputParser(enum=IntentEnum)

        try:
            return parser.parse(text)
        except OutputParserException:
            print(f"Claude devolvió intención inválida: '{text}', usando heurística local")
            return detect_intent(question)

    except Exception as e:
        print(f"Error al usar Claude para detectar intención: {e}")
        return "unknown"

# Regex para extraer texto entre comillas
def extract_text_from_question(question: str):
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    return match.group(1) or match.group(2) if match else question

# Claude extrae el texto del tweet desde la pregunta
def extract_text_from_question_claude(question: str) -> str:
    try:
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
    except Exception as e:
        print(f"Error al extraer texto con Claude: {e}")
        return question
