"""Dense vs Sparse comparison: same query through Chroma and BM25, side by side.

No fusion, no scoring tricks — just the two rankings next to each other
so you can manually judge which retriever understood the query better.

Usage:
    python compare.py                     # run the built-in 6 test queries
    python compare.py "your query here"   # run your own query
"""

import sys

from chroma_ingest import get_collection
from bm25 import load_index, tokenize

TOP_K = 5


def dense_query(collection, text: str, k: int = TOP_K) -> list[str]:
    """Chroma semantic search. .query() nests results one level deep."""
    results = collection.query(query_texts=[text], n_results=k)
    return results["ids"][0]


def sparse_query(bm25, ids: list[str], text: str, k: int = TOP_K) -> list[str]:
    """BM25 keyword search. Positions mapped back to chunk IDs via ids[i]."""
    scores = bm25.get_scores(tokenize(text))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [ids[i] for i in top_idx]


def compare(query: str, collection, bm25, ids: list[str], text_by_id: dict) -> None:
    dense_ids = dense_query(collection, query)
    sparse_ids = sparse_query(bm25, ids, query)
    overlap = set(dense_ids) & set(sparse_ids)

    print(f"\n{'=' * 74}")
    print(f"QUERY: {query}")
    print(f"Overlap in top-{TOP_K}: {len(overlap)}  {sorted(overlap)}")

    print(f"\n--- DENSE (Chroma) top-{TOP_K} " + "-" * 44)
    for i, cid in enumerate(dense_ids, 1):
        preview = text_by_id[cid][:100].replace("\n", " ")
        mark = "*" if cid in overlap else " "
        print(f"{i}. {mark} {cid:<20} {preview}...")

    print(f"\n--- SPARSE (BM25) top-{TOP_K} " + "-" * 45)
    for i, cid in enumerate(sparse_ids, 1):
        preview = text_by_id[cid][:100].replace("\n", " ")
        mark = "*" if cid in overlap else " "
        print(f"{i}. {mark} {cid:<20} {preview}...")

    print("\n(* = chunk found by BOTH retrievers)")


def main() -> None:
    collection = get_collection()

    bundle = load_index()                     # instant — no PDF extraction
    bm25 = bundle["bm25"]
    chunks = bundle["chunks"]
    ids = [c["id"] for c in chunks]
    text_by_id = {c["id"]: c["text"] for c in chunks}

    # Drift alarm: the pickle is a snapshot — warn if Chroma has moved on.
    if collection.count() != len(chunks):
        print(f"WARNING: Chroma holds {collection.count()} chunks but the BM25 "
              f"pickle has {len(chunks)} — re-run `python bm25_simple.py`.")

    queries = sys.argv[1:] or [
        # lexical — predict BM25 wins
        "Item 7A quantitative and qualitative disclosures about market risk",
        "ASC 842 lease obligations",
        # semantic — predict dense wins
        "how profitable is the iPhone business",
        "what threatens Walmart's supply chain",
        # mixed — predict neither is complete alone
        "Pfizer opioid litigation exposure",
        "Exxon climate-related risk disclosures",
    ]
    for q in queries:
        compare(q, collection, bm25, ids, text_by_id)


if __name__ == "__main__":
    main()