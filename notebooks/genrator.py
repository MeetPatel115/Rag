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

def naive_rag(question, collection, k=5):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n\n".join(results["documents"][0])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    return answer, results          

QUESTIONS = [
    "What was Apple's total net sales in fiscal year 2024?",
    "How much did Pfizer spend on research and development in 2023?",
    "What risks does Walmart disclose about supply chain disruptions?",
    "What's Apple's cafeteria menu?",
    "What was Tesla's revenue in 2023?",
    "What did Apple's CEO say about AI in the Q3 earnings call?",
    "Compare their strategies.",
    "How is the company performing?",
    "Which company has the highest R&D spending as a percentage of revenue?",
    "How did total revenue change for each company between their two most recent fiscal years?",
]

if __name__ == "__main__":
    collection = _get_cached_collection()
    for i, q in enumerate(QUESTIONS, 1):
        answer, results = naive_rag(q, collection, k=5)
        print(f"\n{'='*70}\nQ{i}: {q}")
        sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
        dists = [f"{d:.3f}" for d in results["distances"][0]]
        print(f"Retrieved: {sources}")
        print(f"Distances: {dists}")
        print(f"ANSWER:\n{answer}")