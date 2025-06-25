# agent/graph.py

from langgraph.graph import StateGraph, END
from agent.nodes.predictor import predict_sentiment
from agent.nodes.evaluator import get_model_metrics
from agent.nodes.visualizer import get_confusion_matrix
from agent.nodes.responder import generate_response
from agent.utils.helpers import detect_intent_claude, extract_text_from_question_claude

# Nodo que decide la intención
def agent_node_router(state):
    question = state["question"]
    intent = detect_intent_claude(question)
    state["intent"] = intent
    return state

# Nodo de predicción
def prediction_node(state):
    text = extract_text_from_question_claude(state["question"])
    result = predict_sentiment(text)
    state["result"] = result
    return state

# Nodo de métricas
def metrics_node(state):
    result = get_model_metrics()
    state["result"] = result
    return state

# Nodo de matriz de confusión
def conf_matrix_node(state):
    result = get_confusion_matrix()
    state["result"] = result
    return state

# Nodo de respuesta con LLM o mensaje fijo para intent unknown
def responder_node(state):
    if state.get("intent") == "unknown":
        state["response"] = "Lo siento, no entiendo esa pregunta; ¿puedes reformularla?"
        return state

    question = state["question"]
    result = state["result"]
    response = generate_response(question, str(result))
    state["response"] = response
    return state

# Construcción del grafo
def build_agent_graph():
    graph = StateGraph(dict)

    graph.add_node("router", agent_node_router)
    graph.add_node("predict", prediction_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("conf_matrix", conf_matrix_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("intent"),
        {
            "predict": "predict",
            "metrics": "metrics",
            "conf_matrix": "conf_matrix",
            "unknown": "responder",
        }
    )

    # Tras cada nodo funcional, va el nodo de respuesta
    for node in ["predict", "metrics", "conf_matrix"]:
        graph.add_edge(node, "responder")

    graph.add_edge("responder", END)

    return graph.compile()

# Instancia del grafo para uso externo
agent_graph = build_agent_graph()

def process_question(question: str) -> str:
    """Función pública que recibe una pregunta y devuelve la respuesta final."""
    result = agent_graph.invoke({"question": question})
    return result.get("response")
