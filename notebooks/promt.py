import logging
from chroma_ingest import get_collection
from apikey import key_api
from anthropic import Anthropic

logging.getLogger("httpx").setLevel(logging.WARNING)

client = Anthropic(api_key=key_api())

_collection = None

def _get_cached_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


# ---------- NEW: prompt versions ----------

SYSTEM_V1 = """You are a financial analyst assistant. Answer questions about SEC 10-K filings using ONLY the provided context.

Rules:
- Base your answer strictly on the context. Never use outside knowledge.
- If the context doesn't contain the answer, say exactly: "The provided filings don't contain that information."
- Prefer exact figures over vague summaries."""

SYSTEM_V3 = SYSTEM_V1 + """
- Cite every factual claim using the chunk IDs, like [chunk_id: AAPL_2024_0142]
- If the context only partially answers the question, answer what you can and state what is missing."""

SYSTEMS = {"v1": SYSTEM_V1, "v2": SYSTEM_V1, "v3": SYSTEM_V3}


def format_context(results, version):
    docs = results["documents"][0]
    if version == "v1":
        return "\n\n".join(docs)                      # plain, like naive
    # v2/v3: XML chunks with IDs
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    return "\n\n".join(
        f'<chunk id="{cid}" source="{m.get("ticker","?")} {m.get("fiscal_year","?")} 10-K">\n{doc}\n</chunk>'
        for doc, cid, m in zip(docs, ids, metas)
    )
# ---------- END NEW ----------


def naive_rag(question, collection, k=5, version="v1"):          # CHANGED: version param
    results = collection.query(query_texts=[question], n_results=k)
    context = format_context(results, version)                    # CHANGED
    prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEMS[version],                                   # CHANGED: system prompt
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, results


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
    for version in ["v1", "v2", "v3"]:
        print(f"\n{'#'*70}\n# PROMPT VERSION {version}\n{'#'*70}")
        for i, q in enumerate(QUESTIONS, 1):
            answer, results = naive_rag(q, collection, k=5, version=version)
            sources = [f"{m.get('ticker')}-{m.get('fiscal_year')}" for m in results["metadatas"][0]]
            print(f"\nQ{i} [{version}]: {q}")
            print(f"Retrieved: {sources}")
            print(f"ANSWER:\n{answer}")