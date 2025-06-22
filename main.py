# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import process_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_model(query: Query):
    response = process_question(query.question)
    return response
