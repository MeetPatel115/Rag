import sys 
from chroma_ingest import get_collection
from apikey import key_api
from anthropic import Anthropic


key=key_api()
client = Anthropic(api_key=key) 
_collection = None
def _get_cached_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection

def hyde(query: str) -> str:
    """Generate a hypothetical answer/passage to use as the embedding target."""
    prompt = f"""Write a hypothetical passage from an SEC 10-K filing that would answer this question. 
    Write it as if it's actually from a filing — use professional financial language, be specific with numeric placeholders like $X billion, and match the tone of an annual report.

    Question: {query}

    Passage:"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap, fast
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
def retrieve_with_hyde(query, collection, k=5):
    hypothetical = hyde(query)
    results = collection.query(query_texts=[hypothetical], n_results=k)
    context = "\n\n".join(results["documents"][0])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    return answer

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
    "How does JPMorgan make sure it has enough cash if markets freeze up?"
]

if __name__ == "__main__":
    collection = _get_cached_collection()
    for i, q in enumerate(QUESTIONS, 1):
        
        answer_hyde = retrieve_with_hyde(q, collection, k=5)
        answer, results = naive_rag(q, collection, k=5)
        print(f"\n{'='*70}\nQ{i}: {q}")
        sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
        dists = [f"{d:.3f}" for d in results["distances"][0]]
        print(f"Retrieved: {sources}")
        print(f"Distances: {dists}")
        print(f"ANSWER for direct retrival :\n{answer}")
        print(f"ANSWER after query transformation :\n{answer_hyde}")
        