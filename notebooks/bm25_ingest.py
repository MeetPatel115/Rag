"""Simple BM25 pipeline: PDF -> chunks -> BM25 index -> search.

Reuses extract_pdf / chunk_text / FILING_REGISTRY from chroma_ingest.py,
but skips Chroma entirely — chunks go straight into an in-memory BM25 index.

Usage:
    python bm25_simple.py                      # build index, run demo queries
    python bm25_simple.py "your query here"    # build index, run your query
"""

import logging
import string
import sys
import time

from rank_bm25 import BM25Okapi

from chroma_ingest import FILING_REGISTRY, PDF_DIR, chunk_text, extract_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

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
    chunks = build_chunks()
    bm25 = build_bm25(chunks)

    queries = sys.argv[1:] or [
        "Item 7A quantitative and qualitative disclosures",
        "Pfizer litigation exposure",
        "how profitable is the iPhone business",
    ]
    for q in queries:
        print_results(q, search(q, bm25, chunks))


if __name__ == "__main__":
    main()