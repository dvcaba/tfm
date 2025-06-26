from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent.graph import process_question
from agent.utils.loader import get_conf_matrix_path
import os

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(question: Question):
    try:
        response = process_question(question.question)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/confusion-matrix")
def get_confusion_matrix_image():
    """
    Endpoint para servir la imagen de la matriz de confusión
    """
    try:
        img_path = get_conf_matrix_path()
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Imagen de matriz de confusión no encontrada")
        
        return FileResponse(
            img_path,
            media_type="image/png",
            filename="confusion_matrix.png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar la imagen: {str(e)}")

@app.get("/")
def root():
    return {"message": "Tweet Sentiment Analysis Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
