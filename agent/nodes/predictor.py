# nodes/predictor.py
from agent.utils.loader import load_base_model  
import torch

model, tokenizer = load_base_model()
labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        pred = torch.argmax(probs).item()
    return {
        "label": labels[pred],
        "confidence": round(probs[pred].item(), 4),
        "probabilities": {l: round(p, 4) for l, p in zip(labels, probs.tolist())}
    }
