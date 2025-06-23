from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import process_question

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
