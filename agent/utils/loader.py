# agent/utils/loader.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from pathlib import Path

# Detecta si hay GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model(model_path: str = "cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return model, tokenizer

def load_metrics(filename: str = "classification_report.csv") -> pd.DataFrame:
    """
    Carga el CSV de métricas desde <raíz-del-proyecto>/results/classification_report.csv.
    """
    # Subimos dos niveles para llegar a la raíz de proyecto
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "results" / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe métricas en {csv_path}")
    return pd.read_csv(csv_path, index_col=0)

def get_conf_matrix_path(filename: str = "conf_matrix.png") -> str:
    """
    Devuelve la ruta absoluta de la imagen de la matriz de confusión
    en <raíz-del-proyecto>/results/conf_matrix.png.
    """
    root = Path(__file__).resolve().parents[2]
    img_path = root / "results" / filename
    if not img_path.exists():
        raise FileNotFoundError(f"No existe imagen en {img_path}")
    return str(img_path)
