# agent/nodes/responder.py

import os
from anthropic import Anthropic

# Inicializa cliente Claude con tu API key
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_response(question: str, result: str) -> str:
    """
    Genera la respuesta final usando Claude Sonnet.
    """
    prompt = (
        f"A continuación tienes una pregunta y su resultado:\n\n"
        f"Pregunta: {question}\n"
        f"Resultado: {result}\n\n"
        "Redacta una respuesta útil, clara y humana para el usuario final en español."
    )
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Modelo actual disponible
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        # Re-raise la excepción para que sea manejada en responder_node
        raise e

def responder_node(state):
    """
    Nodo final del grafo:
    - Si intent es 'unknown', devolvemos un mensaje fijo.
    - Si no, tratar de generar respuesta con Claude y capturar errores.
    """
    if state.get("intent") == "unknown":
        state["response"] = (
            "Lo siento, no entiendo esa pregunta; ¿puedes reformularla?"
            "\n\n¿Tienes otra pregunta?"
        )
        return state

    question = state["question"]
    result   = state["result"]

    try:
        response = generate_response(question, str(result))
    except Exception as e:
        err = str(e)
        if "not_found_error" in err:
            response = (
                "No tienes acceso al modelo Claude solicitado. "
                "Estoy usando 'claude-3-5-sonnet-20241022'; revisa tu suscripción o el nombre del modelo."
            )
        elif "credit balance" in err or "insufficient_quota" in err:
            response = (
                "Lo siento, tu saldo en Anthropic es insuficiente para generar la respuesta. "
                "Por favor revisa tu plan o tu clave de Anthropic."
            )
        elif "authentication" in err or "invalid_api_key" in err:
            response = (
                "Error de autenticación con Anthropic: revisa tu variable de entorno "
                "`ANTHROPIC_API_KEY`."
            )
        else:
            response = f"Error al generar la respuesta con Claude: {err}"
    
    state["response"] = response + "\n\n¿Tienes otra pregunta?"
    return state