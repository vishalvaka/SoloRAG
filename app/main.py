# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from .retrieval import get_answer   # we'll create this in the next step

app = FastAPI(
    title="SoloRAG – Stripe FAQ Assistant",
    version="0.1.0",
    description="Retrieval-Augmented Generation over Stripe Support docs",
)

class Query(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty.")
        return v

@app.post("/query")
async def query(q: Query):
    """
    Retrieve relevant FAQ snippets, pass them to the LLM,
    and return the markdown answer plus source snippets.
    """
    answer, sources = await get_answer(q.question)
    return {"answer": answer, "sources": sources}

@app.get("/healthz")
async def health():
    return {"status": "ok"}
