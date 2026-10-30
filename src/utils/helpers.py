from pathlib import Path
import logging
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config.setdefault("ollama", {})
    config.setdefault("vector_store", {})
    config.setdefault("chunking", {})
    config.setdefault("retrieval", {})

    config["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL", config["ollama"].get("base_url", "http://localhost:11434"))
    config["ollama"]["chat_model"] = os.getenv("OLLAMA_CHAT_MODEL", config["ollama"].get("chat_model", "llama3.2"))
    config["ollama"]["embedding_model"] = os.getenv("OLLAMA_EMBEDDING_MODEL", config["ollama"].get("embedding_model", "nomic-embed-text"))
    config["vector_store"]["path"] = os.getenv("CHROMA_PATH", config["vector_store"].get("path", "./chroma_data"))
    config["vector_store"]["collection_name"] = os.getenv("CHROMA_COLLECTION", config["vector_store"].get("collection_name", "documents"))
    return config

def configure_logging():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("rag-project")
