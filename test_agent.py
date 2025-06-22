import requests

URL = "http://127.0.0.1:8000/ask"

def test_query(prompt):
    response = requests.post(URL, json={"question": prompt})
    print(f"\n--- Test: {prompt} ---")
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error {response.status_code}:")
        print(response.text)

if __name__ == "__main__":
    test_query("¿Qué sentimiento tiene el siguiente texto? 'Me encanta este producto, es increíble.'")
    test_query("¿Cuáles son las métricas del modelo finetuned?")
    test_query("Muéstrame la matriz de confusión del modelo fine-tuned.")
    test_query("Hola, ¿cómo estás?")
