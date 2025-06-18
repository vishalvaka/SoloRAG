# app/main.py
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

# Local
from .retrieval import get_answer
from .retrieval import stream_answer
from .logger import logger

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

class Health(BaseModel):
    status: str = "ok"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok"}
            ]
        }
    }

@app.post("/query")
async def query(q: Query):
    """
    Retrieve relevant FAQ snippets, pass them to the LLM,
    and return the markdown answer plus source snippets.
    """
    logger.info("query_received", question=q.question)
    answer, sources = await get_answer(q.question)
    return {"answer": answer, "sources": sources}

@app.post("/query/stream")
async def query_stream(q: Query):
    """Stream incremental answer tokens as they are produced by the LLM."""

    logger.info("query_stream_received", question=q.question)

    async def token_generator():
        async for chunk in stream_answer(q.question):
            yield chunk
    return StreamingResponse(token_generator(), media_type="text/plain")

@app.get("/healthz", response_model=Health)
async def health() -> Health:
    """Simple liveness probe."""
    return Health()

# --------------------------- metrics endpoint ---------------------------

@app.get(
    "/metrics",
    responses={
        200: {
            "content": {
                "text/plain": {
                    "example": "# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{method=\"post\",code=\"200\"} 1027 1395066363000"
                }
            },
            "description": "Prometheus metrics in text exposition format.",
        }
    },
)
async def metrics() -> Response:
    """Prometheus text exposition endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
