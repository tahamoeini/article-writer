import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Local Academic RAG Engine API")


class ResearchQuery(BaseModel):
    prompt: str = Field(min_length=3, description="Research question or prompt.")


@lru_cache(maxsize=1)
def get_engine():
    try:
        from scripts.query_engine import get_advanced_query_engine
    except ModuleNotFoundError:
        from query_engine import get_advanced_query_engine

    return get_advanced_query_engine()


@app.get("/health")
async def health_check():
    try:
        get_engine()
    except Exception as exc:
        return {
            "status": "degraded",
            "detail": str(exc),
        }

    return {"status": "ok"}


@app.post("/v1/research/query")
async def execute_query(payload: ResearchQuery):
    try:
        engine = get_engine()
        response = engine.query(payload.prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "text": str(response),
        "citations": [
            {
                "title": node.node.metadata.get("title"),
                "author": node.node.metadata.get("author"),
                "year": node.node.metadata.get("year"),
                "page": node.node.metadata.get("page"),
                "paragraph": node.node.metadata.get("paragraph"),
                "score": node.score,
            }
            for node in response.source_nodes
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))