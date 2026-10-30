from pathlib import Path
from typing import Dict, List
import chromadb
import hashlib

class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_source(self, source: str, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """Replace all stored chunks for one source, then insert its current chunks."""
        existing = self.collection.get(where={"source": source}, include=[])
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
        if not chunks:
            return 0

        ids, documents, metadatas = [], [], []
        for index, item in enumerate(chunks):
            metadata = dict(item["metadata"])
            metadata["source"] = source
            raw_id = f"{source}:{metadata.get('page', '')}:{metadata.get('row', '')}:{index}:{item['text']}"
            ids.append(hashlib.sha1(raw_id.encode("utf-8")).hexdigest())
            documents.append(item["text"])
            metadatas.append(metadata)
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 4):
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        for doc, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            matches.append({"text": doc, "metadata": metadata, "distance": float(distance)})
        return matches

    def count(self) -> int:
        return self.collection.count()
