"""Hybrid retrieval comparison: dense (Chroma) vs sparse (BM25) vs RRF fusion.

Runs the same query through both retrievers over the SAME chunks,
prints the rankings side by side, and fuses them with Reciprocal Rank Fusion.

Usage:
    python hybrid_compare.py                     # run the built-in 6 test queries
    python hybrid_compare.py "your query here"   # run your own query
"""

import sys
from collections import defaultdict

from chroma_ingest import get_collection
from bm25 import load_index, tokenize

TOP_K = 10


# ---------------------------------------------------------------------------
# The two retrievers — identical shape: query text in, ranked ID list out
# ---------------------------------------------------------------------------
def dense_query(collection, text: str, k: int = TOP_K) -> list[str]:
    """Chroma semantic search. Note: .query() nests results one level deep."""
    results = collection.query(query_texts=[text], n_results=k)
    return results["ids"][0]


def sparse_query(bm25, ids: list[str], text: str, k: int = TOP_K) -> list[str]:
    """BM25 keyword search. Positions mapped back to chunk IDs via ids[i]."""
    scores = bm25.get_scores(tokenize(text))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [ids[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Fusion — Exercise 3
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[tuple]:
    """Fuse N ranked ID lists. rank+1 converts 0-based position to true rank."""
    scores = defaultdict(float)
    for results in result_lists:
        for rank, chunk_id in enumerate(results):
            scores[chunk_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Comparison display
# ---------------------------------------------------------------------------
def compare(query: str, collection, bm25, ids: list[str]) -> None:
    dense_ids = dense_query(collection, query)
    sparse_ids = sparse_query(bm25, ids, query)
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:TOP_K]

    overlap = set(dense_ids) & set(sparse_ids)

    print(f"\n{'=' * 78}")
    print(f"Query: {query!r}")
    print(f"Overlap in top-{TOP_K}: {len(overlap)} chunks  {sorted(overlap)}")
    print(f"{'-' * 78}")
    print(f"{'#':<3} {'DENSE (Chroma)':<24} {'SPARSE (BM25)':<24} {'FUSED (RRF)':<24}")
    print(f"{'-' * 78}")
    for i in range(TOP_K):
        d = dense_ids[i] if i < len(dense_ids) else ""
        s = sparse_ids[i] if i < len(sparse_ids) else ""
        f = fused[i][0] if i < len(fused) else ""
        both = "*" if f in overlap else " "
        print(f"{i + 1:<3} {d:<24} {s:<24} {f:<22} {both}")
    print(f"{'-' * 78}")
    print("* = fused chunk that appeared in BOTH retrievers' top-10 (consensus)")


def main() -> None:
    collection = get_collection()

    bundle = load_index()                     # instant — reads the pickle
    bm25 = bundle["bm25"]
    chunks = bundle["chunks"]
    ids = [c["id"] for c in chunks]

    if collection.count() != len(chunks):
        print(f"WARNING: Chroma holds {collection.count()} chunks but the BM25 "
              f"pickle has {len(chunks)} — re-run your BM25 build script.")

    queries = sys.argv[1:] or [
        # lexical — expect BM25 to win
        "Item 7A quantitative and qualitative disclosures about market risk",
        "ASC 842 lease obligations",
        # semantic — expect dense to win
        "how profitable is the iPhone business",
        "what threatens Walmart's supply chain",
        # mixed — expect hybrid to shine
        "Pfizer opioid litigation exposure",
        "Exxon climate-related risk disclosures",
    ]
    for q in queries:
        compare(q, collection, bm25, ids)


if __name__ == "__main__":
    main()