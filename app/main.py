# app/main.py
from fastapi import FastAPI, Response
from pydantic import BaseModel, field_validator

# Local
from .retrieval import get_answer

# Prometheus metrics
from app.middleware import MetricsMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="SoloRAG – Stripe FAQ Assistant",
    version="0.1.0",
    description="Retrieval-Augmented Generation over Stripe Support docs",
)

# Register middleware early so it wraps all routes
app.add_middleware(MetricsMiddleware)

class Query(BaseModel):
    question: str

    # Provide an example for the Swagger /docs page
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How do I issue a refund on Stripe?"
                }
            ]
        }
    }

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

# --------------------------- metrics endpoint ---------------------------

# Expose metrics in Prometheus text format at /metrics
@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
