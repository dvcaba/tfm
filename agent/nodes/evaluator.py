# nodes/evaluator.py
from agent.utils.loader import load_metrics

def get_model_metrics():
    df = load_metrics()
    return df.loc[["accuracy", "macro avg", "weighted avg"]].to_dict()
