# utils/loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model(model_path="cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return model, tokenizer

def load_metrics(path: str = "results/classification_report_finetuned.csv") -> pd.DataFrame:
    """
    Carga el archivo de métricas del modelo fine-tuned desde un CSV y devuelve un DataFrame limpio.
    """
    df = pd.read_csv(path, index_col=0)
    # Asegura que las claves no son defaultdicts al retornar dicts en evaluador
    return df

def get_conf_matrix_path() -> str:
    return "results/conf_matrix_finetuned.png"
