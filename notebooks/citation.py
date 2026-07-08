import re
import logging
from chroma_ingest import get_collection
from apikey import key_api
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

logging.getLogger("httpx").setLevel(logging.WARNING)

client = Anthropic(api_key=key_api())

_collection = None
_embedder = None

def _get_cached_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedder


# ---------- prompts (Exercise 2) ----------

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
        return "\n\n".join(docs)
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    return "\n\n".join(
        f'<chunk id="{cid}" source="{m.get("ticker","?")} {m.get("fiscal_year","?")} 10-K">\n{doc}\n</chunk>'
        for doc, cid, m in zip(docs, ids, metas)
    )


def naive_rag(question, collection, k=5, version="v3"):
    results = collection.query(query_texts=[question], n_results=k)
    context = format_context(results, version)
    prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}\n\nAnswer:"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEMS[version],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, results


# ---------- citation verification (Exercise 3) ----------

CITATION_RE = re.compile(r"\[chunk_id:\s*([^\]]+)\]")


def extract_citations(answer):
    """Return (claim_sentence, cited_id) pairs; claim = sentence minus the tag."""
    pairs = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    for sent in sentences:
        for cid in CITATION_RE.findall(sent):
            claim = CITATION_RE.sub("", sent).strip()
            pairs.append((claim, cid.strip()))
    return pairs


def verify_answer(answer, results, sim_threshold=0.6):
    """Check every citation: does the ID exist, and does the chunk support the claim?"""
    retrieved = dict(zip(results["ids"][0], results["documents"][0]))
    citations = extract_citations(answer)
    if not citations:
        return []
    embedder = _get_embedder()
    report = []
    for claim, cid in citations:
        id_exists = cid in retrieved
        similarity = None
        if id_exists and claim:
            embs = embedder.encode([claim, retrieved[cid]], normalize_embeddings=True)
            similarity = float(embs[0] @ embs[1])
        report.append({
            "claim": claim,
            "cited_id": cid,
            "id_exists": id_exists,
            "similarity": similarity,
            "citation_valid": bool(id_exists and similarity is not None
                                   and similarity >= sim_threshold),
        })
    return report


def summarize(report):
    total = len(report)
    if total == 0:
        return {"total": 0, "invented": 0, "unsupported": 0, "valid": 0, "invalid_rate": None}
    invented = sum(1 for r in report if not r["id_exists"])
    unsupported = sum(1 for r in report if r["id_exists"] and not r["citation_valid"])
    valid = sum(1 for r in report if r["citation_valid"])
    return {"total": total, "invented": invented, "unsupported": unsupported,
            "valid": valid, "invalid_rate": round((total - valid) / total, 3)}


# ---------- test loop ----------

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
    grand_total, grand_valid = 0, 0

    for i, q in enumerate(QUESTIONS, 1):
        answer, results = naive_rag(q, collection, k=5, version="v3")
        report = verify_answer(answer, results)
        stats = summarize(report)

        print(f"\n{'='*70}\nQ{i}: {q}")
        print(f"ANSWER:\n{answer}\n")
        for r in report:
            sim = round(r["similarity"], 3) if r["similarity"] is not None else None
            print(f"  [{'OK ' if r['citation_valid'] else 'BAD'}] {r['cited_id']}  "
                  f"exists={r['id_exists']}  sim={sim}")
        print(f"  Stats: {stats}")

        grand_total += stats["total"]
        grand_valid += stats["valid"]

    if grand_total:
        rate = round(1 - grand_valid / grand_total, 3)
        print(f"\n{'='*70}\nOVERALL invalid citation rate: {rate} "
              f"({grand_total - grand_valid}/{grand_total} citations, sim>=0.6)")
    else:
        print("\nNo citations produced across all questions.")