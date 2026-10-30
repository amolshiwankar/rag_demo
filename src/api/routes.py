from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import logging
import requests

from src.utils.helpers import load_config, configure_logging
from src.ingestion.loader import load_file_bytes, load_url
from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import OllamaEmbedder
from src.vectordb.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import OllamaLLM
from src.prompts.prompt_templates import build_rag_prompt

router = APIRouter()
config = load_config()
logger = configure_logging()

ollama_cfg = config["ollama"]
chunk_cfg = config["chunking"]
vector_cfg = config["vector_store"]
retrieval_cfg = config["retrieval"]

embedder = OllamaEmbedder(
    ollama_cfg["base_url"], ollama_cfg["embedding_model"],
    int(ollama_cfg.get("timeout_seconds", 120)),
)
vector_store = ChromaVectorStore(vector_cfg["path"], vector_cfg["collection_name"])
retriever = Retriever(embedder, vector_store)
llm = OllamaLLM(
    ollama_cfg["base_url"], ollama_cfg["chat_model"],
    int(ollama_cfg.get("timeout_seconds", 120)),
)

class TextIngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(default="manual-input", min_length=1)

class URLIngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)

def _ingest_documents(documents: list[dict], source: str):
    chunks = chunk_documents(
        documents,
        int(chunk_cfg.get("chunk_size", 900)),
        int(chunk_cfg.get("chunk_overlap", 150)),
    )
    if not chunks:
        raise ValueError("No text chunks were created.")
    vectors = embedder.embed([item["text"] for item in chunks])
    count = vector_store.replace_source(source, chunks, vectors)
    logger.info("Ingested %s chunks from %s", count, source)
    return count

@router.get("/health")
def health():
    ollama_ok = False
    try:
        response = requests.get(f'{ollama_cfg["base_url"].rstrip("/")}/api/tags', timeout=3)
        ollama_ok = response.ok
    except requests.RequestException:
        pass
    return {
        "status": "ok",
        "ollama_connected": ollama_ok,
        "vector_count": vector_store.count(),
        "collection": vector_cfg["collection_name"],
    }

@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    filename = Path(file.filename or "uploaded.txt").name
    try:
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 20 MB upload limit.")
        documents = load_file_bytes(filename, content)
        count = _ingest_documents(documents, filename)
        return {"message": "File ingested successfully", "source": filename, "chunks": count}
    except HTTPException:
        raise
    except (ValueError, requests.RequestException, RuntimeError) as exc:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected file ingestion error")
        raise HTTPException(status_code=500, detail="File ingestion failed. Check logs for details.") from exc

@router.post("/ingest/text")
def ingest_text(request: TextIngestRequest):
    try:
        count = _ingest_documents([{"text": request.text, "source": request.source}], request.source)
        return {"message": "Text ingested successfully", "source": request.source, "chunks": count}
    except (ValueError, requests.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/ingest/url")
def ingest_web_url(request: URLIngestRequest):
    try:
        documents = load_url(request.url)
        count = _ingest_documents(documents, request.url)
        return {"message": "URL ingested successfully", "source": request.url, "chunks": count}
    except (ValueError, requests.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/query")
def query(request: QueryRequest):
    try:
        top_k = request.top_k or int(retrieval_cfg.get("top_k", 4))
        matches = retriever.retrieve(request.question, top_k=top_k)
        prompt = build_rag_prompt(request.question, matches)
        answer = llm.generate(prompt)
        sources = [
            {
                "source": item["metadata"].get("source", "unknown"),
                "text": item["text"],
                "distance": round(item["distance"], 4),
                "metadata": item["metadata"],
            }
            for item in matches
        ]
        return {"question": request.question, "answer": answer, "sources": sources}
    except requests.RequestException as exc:
        logger.exception("Ollama request failed")
        raise HTTPException(status_code=503, detail=f"Could not reach Ollama: {exc}") from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed. Check logs for details.") from exc
