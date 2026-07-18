import sys
from chroma_ingest import get_collection
from apikey import key_api
from anthropic import Anthropic


key = key_api()
client = Anthropic(api_key=key)
_collection = None
def _get_cached_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection

def generate_query_variants(query: str, n: int = 4) -> list[str]:
    """Generate n different phrasings of the query for retrieval."""
    prompt = f"""Generate {n} different phrasings of this question for search.
    Each variant must use different vocabulary and different sentence structure — no near-duplicates.
    Vary formality and specificity. Include one that uses SEC filing / financial jargon.
    Return only the questions, one per line, no numbering, no extra text.

    Question: {query}"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap, fast
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = response.content[0].text.strip().split("\n")
    return [line.strip() for line in lines if line.strip()][:n]

def reciprocal_rank_fusion(result_lists, rrf_k: int = 60):
    """
    Merge multiple ranked chunk lists into one.
    Each chunk earns 1/(rrf_k + rank) points per list it appears in.
    Chunks found by many query variants rank highest.
    result_lists: list of chroma query results (one per variant)
    Returns: (ranked_ids, docs_by_id, meta_by_id, scores_by_id)
    """
    scores = {}
    docs = {}
    metas = {}
    for results in result_lists:
        ids = results["ids"][0]
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
            docs[chunk_id] = results["documents"][0][rank - 1]
            metas[chunk_id] = results["metadatas"][0][rank - 1]
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return ranked_ids, docs, metas, scores

def retrieve_multi_query(query, collection, k=5, n_variants=4):
    variants = generate_query_variants(query, n=n_variants)
    all_queries = [query] + variants
    all_results = [collection.query(query_texts=[q], n_results=k) for q in all_queries]
    ranked_ids, docs, metas, scores = reciprocal_rank_fusion(all_results)
    top_ids = ranked_ids[:k]
    context = "\n\n".join(docs[cid] for cid in top_ids)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    fused_info = {
        "variants": variants,
        "top_ids": top_ids,
        "metas": [metas[cid] for cid in top_ids],
        "scores": [scores[cid] for cid in top_ids],
    }
    return answer, fused_info

def naive_rag(question, collection, k=5):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n\n".join(results["documents"][0])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    return answer, results

QUESTIONS = [
    "Is Apple worried about depending too much on China?"
]

if __name__ == "__main__":
    collection = _get_cached_collection()
    for i, q in enumerate(QUESTIONS, 1):

        answer_multi, fused_info = retrieve_multi_query(q, collection, k=5)
        answer, results = naive_rag(q, collection, k=5)
        print(f"\n{'='*70}\nQ{i}: {q}")
        print(f"\nVariants generated:")
        for v in fused_info["variants"]:
            print(f"  - {v}")
        sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
        dists = [f"{d:.3f}" for d in results["distances"][0]]
        fused_sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in fused_info["metas"]]
        rrf_scores = [f"{s:.4f}" for s in fused_info["scores"]]
        print(f"\nRetrieved (direct):      {sources}")
        print(f"Distances (direct):      {dists}")
        print(f"Retrieved (multi-query): {fused_sources}")
        print(f"RRF scores:              {rrf_scores}")
        print(f"\nANSWER for direct retrival :\n{answer}")
        print(f"\nANSWER after multi-query + RRF :\n{answer_multi}")