from pathlib import Path
from typing import Dict, List
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

def load_file_bytes(filename: str, content: bytes) -> List[Dict[str, str]]:
    """Parse an uploaded file into one or more text documents with source metadata."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    docs: List[Dict[str, str]] = []
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append({"text": text, "source": filename, "page": str(page_num)})
    elif suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig", errors="replace").strip()
        if text:
            docs.append({"text": text, "source": filename})
    elif suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(content)).fillna("")
        for index, row in frame.iterrows():
            row_text = " | ".join(f"{column}: {value}" for column, value in row.items())
            if row_text.strip(" |"):
                docs.append({"text": row_text, "source": filename, "row": str(index + 1)})
    if not docs:
        raise ValueError("No readable text was found in the uploaded file.")
    return docs

def load_url(url: str) -> List[Dict[str, str]]:
    """Fetch and extract visible text from a public web page."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    response = requests.get(url, timeout=20, headers={"User-Agent": "ModularRAG/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if not text:
        raise ValueError("No readable text found at this URL.")
    return [{"text": text, "source": url}]
