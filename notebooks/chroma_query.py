"""Query the SEC filings collection.

Usage:
    python query.py "research and development spending"

Or import:
    from query import retrieve
    hits = retrieve("risk factors", where={"sector": "Healthcare"})
"""

import sys

from chroma_ingest import get_collection

_collection = None


def _get_cached_collection():
    """Load the collection (and embedding model) once, reuse across calls."""
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def retrieve(query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Semantic search with optional metadata filtering.

    Filter examples:
        where={"ticker": "AAPL"}
        where={"$and": [{"sector": "Healthcare"}, {"year": 2025}]}
    """
    results = _get_cached_collection().query(
        query_texts=[query], n_results=n_results, where=where
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def print_hits(query: str, where: dict | None = None, n_results: int = 3) -> None:
    """Human-readable output for quick inspection."""
    print(f"\nquery: '{query}'  |  filter: {where}")
    print("-" * 70)
    hits = retrieve(query, n_results=n_results, where=where)
    if not hits:
        print("-> 0 results")
        return
    for rank, hit in enumerate(hits, 1):
        m = hit["metadata"]
        print(f"#{rank}  dist={hit['distance']:.4f}  "
              f"{m['ticker']} {m['year']} {m.get('filing_type', '')}  "
              f"chunk {m['chunk_index']}")
        print(f"    {hit['text'][:180].replace(chr(10), ' ')} ...\n")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "research and development spending"
    print_hits(q)                                        # no filter
    print_hits(q, where={"ticker": "AAPL"})              # company
    print_hits(q, where={"$and": [{"ticker": "PFE"},     # company + year
                                  {"year": 2025}]})