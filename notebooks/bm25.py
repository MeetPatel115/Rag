"""Simple BM25 pipeline: PDF -> chunks -> BM25 index -> pickle -> search.

Reuses extract_pdf / chunk_text / FILING_REGISTRY from chroma_ingest.py,
but skips Chroma entirely — chunks go straight into an in-memory BM25 index,
which is then pickled so query scripts never pay the PDF-extraction cost.

Workflow:
    python bm25_simple.py                      # BUILD: slow, run after any re-ingest
    (other scripts)  from bm25_simple import load_index   # QUERY: instant

Usage:
    python bm25_simple.py                      # build + pickle + demo queries
    python bm25_simple.py "your query here"    # build + pickle + your query
"""

import logging
import pickle
import string
import sys
import time

from rank_bm25 import BM25Okapi

from chroma_ingest import FILING_REGISTRY, PDF_DIR, PROJECT_ROOT, chunk_text, extract_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BM25_PATH = PROJECT_ROOT / "bm25_index.pkl"

_PUNCT = string.punctuation


def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip edge punctuation."""
    return [t for raw in text.lower().split() if (t := raw.strip(_PUNCT))]


def build_chunks() -> list[dict]:
    """PDF -> chunks for every filing in the registry (no Chroma involved)."""
    all_chunks = []
    for name, meta in FILING_REGISTRY.items():
        pdf_path = PDF_DIR / f"{name}.pdf"
        if not pdf_path.exists():
            log.warning("Skipping %s (PDF not found)", name)
            continue

        t0 = time.time()
        text = extract_pdf(pdf_path)
        chunks = chunk_text(text, {**meta, "source_file": pdf_path.name})
        all_chunks.extend(chunks)
        log.info("%-18s %5d chunks  %5.1fs", name, len(chunks), time.time() - t0)

    if not all_chunks:
        raise RuntimeError("No chunks produced — check PDF_DIR and filenames.")
    return all_chunks


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Tokenize chunk texts and build the BM25 index over them."""
    t0 = time.time()
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    log.info("Built BM25 over %d chunks in %.1fs", len(chunks), time.time() - t0)
    return bm25


def save_index(bm25: BM25Okapi, chunks: list[dict]) -> None:
    """Pickle the index and chunks TOGETHER — position i in bm25 is chunks[i].

    Saving them as one bundle makes misalignment structurally impossible.
    """
    bundle = {"bm25": bm25, "chunks": chunks}
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved index to %s (%.1f MB)",
             BM25_PATH, BM25_PATH.stat().st_size / 1e6)


def load_index() -> dict:
    """Load the pickled bundle: {"bm25": BM25Okapi, "chunks": list[dict]}.

    Query scripts import and call this instead of rebuilding from PDFs.
    """
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"{BM25_PATH} not found — run `python bm25_simple.py` first to build it."
        )
    with open(BM25_PATH, "rb") as f:
        bundle = pickle.load(f)
    log.info("Loaded BM25 index: %d chunks", len(bundle["chunks"]))
    return bundle


def search(query: str, bm25: BM25Okapi, chunks: list[dict], k: int = 5) -> list[tuple]:
    """Score all chunks for the query, return top-k as (chunk, score)."""
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in top_idx]


def print_results(query: str, results: list[tuple]) -> None:
    print(f"\nQuery: {query!r}")
    print("-" * 70)
    for chunk, score in results:
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"{score:7.2f}  {chunk['id']:<18}  {preview}...")


def main() -> None:
    """Builder entry point: PDFs -> chunks -> BM25 -> pickle, then demo queries."""
    chunks = build_chunks()
    bm25 = build_bm25(chunks)
    save_index(bm25, chunks)

    queries = sys.argv[1:] or [
        "Item 7A quantitative and qualitative disclosures",
        "Pfizer litigation exposure",
        "how profitable is the iPhone business",
    ]
    for q in queries:
        print_results(q, search(q, bm25, chunks))


if __name__ == "__main__":
    main()