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

def step_back(query: str) -> str:
    """Generate a more general, higher-level question for background context."""
    prompt = f"""Given the specific question below, generate a more general, higher-level question that provides useful background context.
    The step-back question should be answerable from a broader section of a 10-K filing.
    Return only the step-back question, nothing else.

    Specific question: {query}
    Step-back question:"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap, fast
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ---------------------------------------------------------------
# Retrieval + merge + generation — COMMENTED OUT for now.
#---------------------------------------------------

def retrieve_with_step_back(query, collection, k=5):
    step_back_q = step_back(query)

    # Retrieve for BOTH questions
    results_original = collection.query(query_texts=[query], n_results=k)
    results_stepback = collection.query(query_texts=[step_back_q], n_results=k)

    # Merge chunks, deduplicate by chunk_id (original first, then step-back)
    seen_ids = set()
    merged_docs = []
    merged_metas = []
    for results in [results_original, results_stepback]:
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
    return answer, step_back_q, merged_metas

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
    "Is Apple worried about depending too much on China?",
    "How does JPMorgan make sure it has enough cash if markets freeze up?",
    "What did Apple spend on research and development?",
    "What are Pfizer's main patent expiration risks?",
    "How is ExxonMobil thinking about the energy transition?",
]

if __name__ == "__main__":
    collection = _get_cached_collection()   # not needed for transformation-only test
    for i, q in enumerate(QUESTIONS, 1):
        step_back_q = step_back(q)
        print(f"\n{'='*70}\nQ{i} (specific) : {q}")
        print(f"Q{i} (step-back): {step_back_q}")


        answer_sb, step_back_q, merged_metas = retrieve_with_step_back(q, collection, k=5)
        answer, results = naive_rag(q, collection, k=5)
        sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
        merged_sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in merged_metas]
        print(f"Retrieved (direct)   : {sources}")
        print(f"Retrieved (merged)   : {merged_sources}  ({len(merged_sources)} chunks after dedup)")
        print(f"\nANSWER for direct retrival :\n{answer}")
        print(f"\nANSWER with step-back context :\n{answer_sb}")