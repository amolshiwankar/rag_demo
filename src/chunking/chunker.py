from typing import Dict, List

def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = text.rfind(" ", start, end)
            if boundary > start + int(chunk_size * 0.6):
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks

def chunk_documents(documents: List[Dict[str, str]], chunk_size: int, chunk_overlap: int):
    """Expand loaded documents into chunks while preserving their metadata."""
    result = []
    for document in documents:
        for index, chunk in enumerate(split_text(document["text"], chunk_size, chunk_overlap)):
            metadata = {k: str(v) for k, v in document.items() if k != "text"}
            metadata["chunk_index"] = str(index)
            result.append({"text": chunk, "metadata": metadata})
    return result
