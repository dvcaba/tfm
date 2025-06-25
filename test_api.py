import requests

URL = "http://127.0.0.1:8000/ask"

questions = [
    "¿Qué sentimiento tiene el siguiente texto? 'Me encanta este producto, es increíble.'",
    "¿Cuáles son las métricas del modelo finetuned?",
    "Muéstrame la matriz de confusión del modelo fine-tuned.",
    "Hola, ¿cómo estás?",  # Desconocido
    "salir"
]

for q in questions:
    print(f"\n--- Pregunta: {q} ---")
    response = requests.post(URL, json={"question": q})
    if response.ok:
        print("Respuesta:", response.json())
    else:
        print("Error", response.status_code)
        print(response.text)
