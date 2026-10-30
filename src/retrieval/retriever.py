from typing import List, Dict
from src.embeddings.embedder import OllamaEmbedder
from src.vectordb.vector_store import ChromaVectorStore

class Retriever:
    def __init__(self, embedder: OllamaEmbedder, vector_store: ChromaVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 4) -> List[Dict]:
        vector = self.embedder.embed([question])[0]
        return self.vector_store.search(vector, top_k=top_k)
