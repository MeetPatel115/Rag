"""Two-stage retrieval: hybrid (dense + BM25 + RRF) -> BGE cross-encoder rerank.

Stage 1 casts a wide net (retrieve_k=25) optimizing recall.
Stage 2 re-judges each candidate WITH the query (cross-encoder) for precision.
Flags let you A/B test every configuration for the Exercise 6 matrix.

Usage:
    python retriever.py                      # demo: before/after rerank on test queries
    python retriever.py "your query here"    # your own query, before/after
"""

import logging
import sys
import time
from sentence_transformers import CrossEncoder

from chroma_ingest import get_collection
from bm25 import load_index, tokenize          # <- your notebooks/bm25.py
from rrf import (
    dense_query,
    sparse_query,
    reciprocal_rank_fusion,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time setup (module level — never inside functions)
# ---------------------------------------------------------------------------
log.info("Loading indexes...")
_collection = get_collection()
_bundle = load_index()
_bm25 = _bundle["bm25"]
_chunks = _bundle["chunks"]
_ids = [c["id"] for c in _chunks]
_chunk_by_id = {c["id"]: c for c in _chunks}

log.info("Loading BGE reranker (first run downloads ~2.3 GB)...")
_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
log.info("Ready.")


# ---------------------------------------------------------------------------
# Stage 1 — retrieval (recall)
# ---------------------------------------------------------------------------
def hybrid_query(text: str, k: int = 25) -> list[dict]:
    """Dense + sparse in parallel, fused with RRF. Returns full chunk dicts."""
    dense_ids = dense_query(_collection, text, k=k)
    sparse_ids = sparse_query(_bm25, _ids, text, k=k)
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    return [_chunk_by_id[cid] for cid, _ in fused[:k]]


def dense_only_query(text: str, k: int = 25) -> list[dict]:
    """Dense retrieval alone. Returns full chunk dicts."""
    return [_chunk_by_id[cid] for cid in dense_query(_collection, text, k=k)]


# ---------------------------------------------------------------------------
# Stage 2 — reranking (precision)
# ---------------------------------------------------------------------------
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder re-judges every (query, chunk) pair from scratch.

    Input order (the RRF ranking) is deliberately discarded — stage 1's job
    was getting the right chunks into the candidate set, not ordering them.
    """
    pairs = [[query, c["text"]] for c in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# The full pipeline with A/B flags (Exercise 6 configs = flag combinations)
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    k: int = 5,
    retrieve_k: int = 25,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> list[dict]:
    if use_hybrid:
        candidates = hybrid_query(query, k=retrieve_k)
    else:
        candidates = dense_only_query(query, k=retrieve_k)

    if use_rerank:
        return rerank(query, candidates, top_k=k)
    return candidates[:k]


# ---------------------------------------------------------------------------
# Demo: before/after rerank, side by side, with latency
# ---------------------------------------------------------------------------
def _show(label: str, results: list[dict], ms: float) -> None:
    print(f"\n--- {label}  ({ms:.0f} ms) " + "-" * max(0, 50 - len(label)))
    for i, c in enumerate(results, 1):
        preview = c["text"][:100].replace("\n", " ")
        print(f"{i}. {c['id']:<20} {preview}...")


def main() -> None:
    queries = sys.argv[1:] or [
        "Apple Item 7A market risk",
        "Pfizer opioid litigation exposure",
        "how profitable is the iPhone business",
    ]
    for q in queries:
        print(f"\n{'=' * 74}")
        print(f"QUERY: {q}")

        t0 = time.perf_counter()
        before = retrieve(q, use_hybrid=True, use_rerank=False)
        t_before = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        after = retrieve(q, use_hybrid=True, use_rerank=True)
        t_after = (time.perf_counter() - t0) * 1000

        _show("HYBRID only (no rerank)", before, t_before)
        _show("HYBRID + BGE RERANK", after, t_after)


if __name__ == "__main__":
    main()