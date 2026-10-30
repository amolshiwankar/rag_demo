# Modular RAG Project (Python + FastAPI + Ollama + ChromaDB)

A local Retrieval-Augmented Generation (RAG) application that ingests PDF, CSV, TXT, Markdown, and web-page content, splits it into chunks, creates embeddings with Ollama, stores vectors in ChromaDB, retrieves relevant passages, and generates grounded answers with a local Ollama chat model.

## 1. Architecture

```text
Client
  |
  v
FastAPI (src/api/routes.py)
  |
  +---- POST /ingest/file or /ingest/text
  |        |
  |        v
  |     ingestion -> chunking -> embeddings -> ChromaDB
  |
  +---- POST /query
           |
           v
        embedding -> retrieval (similarity search)
           |
           v
        prompt template -> Ollama LLM -> answer + sources
```

## 2. Folder structure

```text
rag-project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt_templates.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   └── test_chunker.py
├── logs/
└── main.py
```

## 3. Prerequisites

- Python 3.10 or newer
- Ollama installed and running: https://ollama.com
- Enough disk/RAM for the models you select

Pull the default models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

The chat and embedding models are configurable in `config.yaml` or through environment variables.

## 4. Installation

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 5. Configuration

Edit `config.yaml`:

```yaml
ollama:
  base_url: "http://localhost:11434"
  chat_model: "llama3.2"
  embedding_model: "nomic-embed-text"
  timeout_seconds: 120

chunking:
  chunk_size: 900
  chunk_overlap: 150

vector_store:
  path: "./chroma_data"
  collection_name: "documents"

retrieval:
  top_k: 4
```

Optional `.env` overrides:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PATH=./chroma_data
CHROMA_COLLECTION=documents
```

## 6. Run the application

From the project root:

```bash
uvicorn main:app --reload
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 7. API usage

### A. Upload a document

`POST /ingest/file` (multipart form upload)

Supported extensions: `.pdf`, `.txt`, `.md`, `.csv`

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/ingest/file" \
  -F "file=@./sample.pdf"
```

PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ingest/file" -F "file=@C:\path\to\sample.pdf"
```

### B. Ingest text directly

`POST /ingest/text`

```json
{
  "text": "The refund period is 30 days from the date of purchase.",
  "source": "refund-policy"
}
```

### C. Ingest a public web page

`POST /ingest/url`

```json
{
  "url": "https://example.com"
}
```

Only ingest pages you are authorized to access. This basic loader does not execute JavaScript or bypass authentication.

### D. Ask a question

`POST /query`

```json
{
  "question": "What is the refund period?",
  "top_k": 4
}
```

Example response:

```json
{
  "question": "What is the refund period?",
  "answer": "The refund period is 30 days from the date of purchase.",
  "sources": [
    {
      "source": "refund-policy",
      "text": "The refund period is 30 days from the date of purchase.",
      "distance": 0.12
    }
  ]
}
```

The exact answer depends on the documents you ingest and the models running locally.

## 8. Test

```bash
pytest -q
```

The included unit test covers chunk splitting and does not require Ollama.

## 9. Important implementation notes

- The first document ingestion may take time while Ollama creates embeddings.
- ChromaDB data is persisted in `./chroma_data`.
- Re-ingesting a source replaces its previous chunks, avoiding duplicate chunks for the same source.
- Retrieval uses vector similarity. The `distance` in the response is a distance score; lower is closer.
- The LLM is instructed to answer from retrieved context and to say when the context does not contain the answer. This is a prompt-level guard, not a guarantee against hallucinations.
- This is a starter implementation. For production, add authentication, upload size limits, malware scanning, rate limiting, stronger URL validation/SSRF protection, observability, backups, and document-level access control.

## 10. Troubleshooting

**Ollama connection refused**
- Start Ollama and check `http://localhost:11434`.
- Confirm the configured `OLLAMA_BASE_URL`.

**Model not found**
```bash
ollama list
ollama pull llama3.2
ollama pull nomic-embed-text
```

**No answer / no sources**
- Ingest documents first.
- Ask a question that matches the ingested content.
- Try increasing `top_k`.

**ChromaDB or dependency installation errors**
- Use Python 3.10–3.12 in a clean virtual environment.
- Upgrade pip and reinstall requirements.

## 11. Extend the project

Good next additions include hybrid keyword + vector search, reranking, conversation memory, metadata filters, OCR for scanned PDFs, background ingestion jobs, and a Docker Compose deployment.


##
What happens in different scenarios?
Operation	              Source	          Result
Upload PDF	          company.pdf	       Stores PDF chunks
Ingest text	          refund-policy	     Adds text chunks; PDF remains
Upload another PDF	  policy.pdf	       Adds PDF chunks; previous PDF remains
Re-upload same PDF	  company.pdf	       Replaces old chunks from that PDF
Ingest text again    	refund-policy	     Replaces previous chunks from that source

Example
Initially, ChromaDB contains:
Existing ChromaDB data
company.pdf

Stored

shipping.pdf

Stored


Now ingest text
refund-policy

Added


Final ChromaDB data

All three sources remain stored.

Important: Your current code deletes and replaces chunks only when the exact same source is used. Different source names do not delete one another.

So, you can safely use PDF upload and direct text ingestion together.