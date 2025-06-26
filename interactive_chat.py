# interactive_chat.py

import asyncio
import os
from agent.graph import ConversationalSession, conversational_process_question
from agent.utils.loader import get_conf_matrix_path

CONF_MATRIX_PATH = get_conf_matrix_path()


def show_image_with_matplotlib(image_path):
    """
    Muestra la imagen desde un archivo local con matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        print("Mostrando imagen con matplotlib...")

        img = mpimg.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title("Matriz de Confusión del Modelo Fine-tuned")
        plt.axis('off')
        plt.tight_layout()
        plt.show()  # BLOQUEA hasta que el usuario cierre la imagen
        return True
    except Exception as e:
        print(f"Error mostrando imagen: {e}")
        return False


def handle_confusion_matrix_request(prompt: str):
    """
    Si la pregunta es sobre matriz de confusión, muestra la imagen localmente.
    """
    if "confusión" in prompt.lower() or "confusion" in prompt.lower():
        print("\nMostrando matriz de confusión...")
        try:
            if os.path.exists(CONF_MATRIX_PATH):
                show_image_with_matplotlib(CONF_MATRIX_PATH)
            else:
                print(f"No se encontró la imagen en {CONF_MATRIX_PATH}")
        except Exception as e:
            print(f"Error al mostrar la imagen: {e}")


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

    session = ConversationalSession()

    while True:
        question = await get_input_with_timeout("Tú: ", timeout=20)

        if question is None:
            print("Conversación finalizada por inactividad.\n")
            break

        result = conversational_process_question(question, session)
        print("\n" + result["response"])

        handle_confusion_matrix_request(question)

        if not result["session_active"]:
            break


if __name__ == "__main__":
    asyncio.run(interactive_chat())
