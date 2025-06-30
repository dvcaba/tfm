# nodes/evaluator.py
from agent.utils.loader import load_metrics

def get_model_metrics():
    df = load_metrics()
    # Keep only the aggregate rows that contain the true model metrics
    df = df.loc[["macro avg", "weighted avg"]]
    # Convertir DataFrame a estructura dict completamente compatible
    # Rename keys replacing spaces with underscores (e.g. "macro avg" -> "macro_avg")
    return {row.replace(" ", "_"): df.loc[row].to_dict() for row in df.index}
