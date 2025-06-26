# interactive_chat.py

import asyncio
import requests
import os

URL = "http://127.0.0.1:8000/ask"
IMAGE_URL = "http://127.0.0.1:8000/confusion-matrix"

def show_image_with_matplotlib(image_path):
    """
    Muestra la imagen usando matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title("Matriz de Confusión del Modelo Fine-tuned")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"Error mostrando imagen: {e}")
        return False

def test_query(prompt):
    response = requests.post(URL, json={"question": prompt})
    print(f"\nPregunta: {prompt}")
    
    if response.status_code == 200:
        result = response.json()
        print("Respuesta:", result.get("response", "[sin respuesta]"))

        # Mostrar imagen si se trata de matriz de confusión
        if "confusión" in prompt.lower() or "confusion" in prompt.lower():
            print("\nDescargando matriz de confusión...")
            try:
                img_response = requests.get(IMAGE_URL)
                if img_response.status_code == 200:
                    temp_path = "temp_confusion_matrix.png"
                    with open(temp_path, "wb") as f:
                        f.write(img_response.content)
                    show_image_with_matplotlib(temp_path)
                    os.remove(temp_path)
                else:
                    print(f"Error al descargar imagen: {img_response.status_code}")
            except Exception as e:
                print(f"Error con la imagen: {e}")
    else:
        print(f"Error {response.status_code}:")
        print(response.text)

async def get_input_with_timeout(prompt, timeout=20):
    print(prompt, end="", flush=True)
    try:
        return await asyncio.wait_for(asyncio.to_thread(input), timeout)
    except asyncio.TimeoutError:
        print("\nTiempo agotado: no se recibió respuesta en 20 segundos.")
        return None

async def interactive_chat():
    print("\nBienvenido al asistente conversacional.")
    print("Escribe tu pregunta o 'salir' para finalizar.\n")

    while True:
        question = await get_input_with_timeout("Tú: ", timeout=20)

        if question is None or question.strip().lower() in {"no", "salir"}:
            print("Conversación finalizada. ¡Hasta luego!\n")
            break

        test_query(question)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
