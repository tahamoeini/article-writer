import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scripts.query_engine import get_advanced_query_engine

app = FastAPI(title="Local Academic RAG Engine API")
engine = get_advanced_query_engine()

class ResearchQuery(BaseModel):
    prompt: str

@app.post("/v1/research/query")
async def execute_query(payload: ResearchQuery):
    response = engine.query(payload.prompt)
    return {
        "text": str(response),
        "citations": [
            {
                "title": node.node.metadata.get("title"),
                "author": node.node.metadata.get("author"),
                "year": node.node.metadata.get("year"),
                "page": node.node.metadata.get("page"),
                "score": node.score
            }
            for node in response.source_nodes
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)