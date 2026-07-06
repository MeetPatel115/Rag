"""SEC 10-K ingestion pipeline: PDF -> chunks -> ChromaDB.

Usage:
    python ingest.py                 # ingest every filing in the registry
    python ingest.py jpm_2025_10k    # ingest specific filing(s) only
"""

import logging
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # src/ -> project root
PDF_DIR = PROJECT_ROOT / "data" / "extract"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"                # keep DB out of data/
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_BATCH_SIZE = 500

# One entry per filing. Key must match the PDF filename stem in PDF_DIR.
FILING_REGISTRY: dict[str, dict] = {
    "apple_2024_10k": {
        "company": "Apple Inc.", "ticker": "AAPL",
        "sector": "Technology", "filing_type": "10-K", "year": 2024,
    },
    "jpm_2025_10k": {
        "company": "JPMorgan Chase & Co.", "ticker": "JPM",
        "sector": "Financials", "filing_type": "10-K", "year": 2025,
    },
    "pfizer_2024_10k": {
        "company": "Pfizer Inc.", "ticker": "PFE",
        "sector": "Healthcare", "filing_type": "10-K", "year": 2024,
    },
    "pfizer_2025_10k": {
        "company": "Pfizer Inc.", "ticker": "PFE",
        "sector": "Healthcare", "filing_type": "10-K", "year": 2025,
    },
    "wmt_2025_10k": {
        "company": "Walmart Inc.", "ticker": "WMT",
        "sector": "Consumer Staples", "filing_type": "10-K", "year": 2025,
    },
    "xom_2025_10k": {
        "company": "Exxon Mobil Corporation", "ticker": "XOM",
        "sector": "Energy", "filing_type": "10-K", "year": 2025,
    },
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> str:
    """Extract full text from a PDF."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, metadata: dict) -> list[dict]:
    """Split text into overlapping chunks, each carrying filing metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        {
            "id": f"{metadata['ticker']}_{metadata['year']}_{i:05d}",
            "text": piece,
            "metadata": {**metadata, "chunk_index": i},
        }
        for i, piece in enumerate(splitter.split_text(text))
    ]


def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent shared collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[dict], collection: chromadb.Collection) -> None:
    """Batched, idempotent insert."""
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = chunks[start:start + CHROMA_BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def ingest_filing(name: str, collection: chromadb.Collection) -> int:
    """End-to-end ingestion for one filing. Returns chunk count."""
    if name not in FILING_REGISTRY:
        raise KeyError(f"'{name}' not in FILING_REGISTRY")

    pdf_path = PDF_DIR / f"{name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    metadata = {**FILING_REGISTRY[name], "source_file": pdf_path.name}

    t0 = time.time()
    text = extract_pdf(pdf_path)
    chunks = chunk_text(text, metadata)
    upsert_chunks(chunks, collection)

    log.info("%-18s %5d chunks  %6.1fs", name, len(chunks), time.time() - t0)
    return len(chunks)


def main(names: list[str] | None = None) -> None:
    """Ingest the given filings, or the entire registry if none given."""
    targets = names or list(FILING_REGISTRY)
    collection = get_collection()

    total = sum(ingest_filing(name, collection) for name in targets)

    log.info("Ingested %d chunks across %d filings", total, len(targets))
    log.info("Collection '%s' now holds %d chunks",
             COLLECTION_NAME, collection.count())


if __name__ == "__main__":
    main(sys.argv[1:] or None)