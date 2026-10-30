import pytest
from src.chunking.chunker import split_text, chunk_documents

def test_split_text_empty():
    assert split_text("   ") == []

def test_split_text_respects_max_size():
    text = " ".join(["word"] * 100)
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)
    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, chunk_overlap=10)

def test_chunk_documents_preserves_metadata():
    docs = [{"text": "hello world " * 10, "source": "unit-test.txt", "page": "1"}]
    chunks = chunk_documents(docs, chunk_size=25, chunk_overlap=5)
    assert chunks
    assert chunks[0]["metadata"]["source"] == "unit-test.txt"
    assert chunks[0]["metadata"]["page"] == "1"
