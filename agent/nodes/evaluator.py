# nodes/evaluator.py
from agent.utils.loader import load_metrics

def get_model_metrics():
    df = load_metrics()
    df = df.loc[["accuracy", "macro avg", "weighted avg"]]
    # Convertir DataFrame a estructura dict completamente compatible
    return {row: df.loc[row].to_dict() for row in df.index}
