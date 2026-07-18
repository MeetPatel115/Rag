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

def decompose(query: str, max_subqs: int = 4) -> list[str]:
    """Break a complex question into simpler sub-questions, each retrievable on its own."""
    prompt = f"""Break the complex question below into {max_subqs} or fewer simple sub-questions.
    Each sub-question must be self-contained and answerable from a single section of an SEC 10-K filing.
    Name the specific company in every sub-question (do not use "it" or "the company").
    If the question compares companies, create separate sub-questions for each company.
    Return only the sub-questions, one per line, no numbering, no extra text.

    Complex question: {query}
    Sub-questions:"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap, fast
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = response.content[0].text.strip().split("\n")
    return [line.strip() for line in lines if line.strip()][:max_subqs]

# ---------------------------------------------------------------
# Retrieval + merge + generation — COMMENTED OUT for now.

# ---------------------------------------------------------------

def retrieve_with_decomposition(query, collection, k=3):
    # NOTE: k=3 per sub-question (not 5) — with up to 4 sub-questions,
    # merged context can reach 12 chunks; k=5 would risk 20.
    sub_questions = decompose(query)

    # Retrieve for EACH sub-question
    all_results = [collection.query(query_texts=[sq], n_results=k) for sq in sub_questions]

    # Merge chunks, deduplicate by chunk_id (in sub-question order)
    seen_ids = set()
    merged_docs = []
    merged_metas = []
    for results in all_results:
        for i, chunk_id in enumerate(results["ids"][0]):
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_docs.append(results["documents"][0][i])
                merged_metas.append(results["metadatas"][0][i])

    context = "\n\n".join(merged_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    return answer, sub_questions, merged_metas

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
    "Compare the supply chain risks of Apple and Walmart.",
    "How do Pfizer's R&D spending and patent expiration risks affect its future revenue outlook?",
    "Which company is more exposed to interest rate changes, JPMorgan or ExxonMobil, and why?",
    "What are Apple's main revenue segments and how did each perform?",
    "How do Walmart and ExxonMobil differ in how they discuss climate-related risks?",
]

if __name__ == "__main__":
    collection = _get_cached_collection()   # not needed for transformation-only test
    for i, q in enumerate(QUESTIONS, 1):
        sub_questions = decompose(q)
        print(f"\n{'='*70}\nQ{i} (complex): {q}")
        print(f"Sub-questions:")
        for sq in sub_questions:
            print(f"  - {sq}")

  
        answer_dc, sub_questions, merged_metas = retrieve_with_decomposition(q, collection, k=3)
        answer, results = naive_rag(q, collection, k=5)
        sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
        merged_sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in merged_metas]
        print(f"Retrieved (direct)   : {sources}")
        print(f"Retrieved (merged)   : {merged_sources}  ({len(merged_sources)} chunks after dedup)")
        print(f"\nANSWER for direct retrival :\n{answer}")
        print(f"\nANSWER with decomposition :\n{answer_dc}")