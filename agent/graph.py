from langgraph.graph import StateGraph, END
from agent.nodes.predictor import predict_sentiment
from agent.nodes.evaluator import get_model_metrics
from agent.nodes.visualizer import get_confusion_matrix
from agent.utils.helpers import detect_intent, extract_text_from_question

# Definimos el estado del agente
def agent_node_router(state):
    question = state["question"]
    intent = detect_intent(question)
    state["intent"] = intent
    return intent

def prediction_node(state):
    text = extract_text_from_question(state["question"])
    result = predict_sentiment(text)
    state["result"] = result
    return state

def metrics_node(state):
    result = get_model_metrics()
    state["result"] = result
    return state

def conf_matrix_node(state):
    result = get_confusion_matrix()
    state["result"] = result
    return state

# Construimos el grafo
def build_agent_graph():
    graph = StateGraph(dict)  

    graph.add_node("router", agent_node_router)
    graph.add_node("predict", prediction_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("conf_matrix", conf_matrix_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", lambda x: x["intent"], {
        "predict": "predict",
        "metrics": "metrics",
        "conf_matrix": "conf_matrix"
    })

    for node in ["predict", "metrics", "conf_matrix"]:
        graph.add_edge(node, END)

    return graph.compile()

# Interfaz externa
agent_graph = build_agent_graph()

def process_question(question: str):
    result = agent_graph.invoke({"question": question})
    return result["result"]
