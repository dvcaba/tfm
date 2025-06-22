import re
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


def extract_text_from_question(question: str):
    """
    Extrae el texto de un tweet desde comillas simples o dobles.
    Ej: ¿Qué sentimiento tiene el tweet "I love Mondays"? -> I love Mondays
    """
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    return match.group(1) or match.group(2) if match else question