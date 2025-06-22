# utils/loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model(model_path="cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return model, tokenizer

def load_metrics():
    return pd.read_csv("results/classification_report_baseline.csv", index_col=0)

def get_conf_matrix_path():
    return "results/conf_matrix_finetuned.png"
