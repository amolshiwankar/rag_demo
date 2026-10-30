from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Modular RAG API",
    description="Document ingestion, vector retrieval, and local LLM question answering.",
    version="1.0.0",
)
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Modular RAG API is running", "docs": "/docs"}
