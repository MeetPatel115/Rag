# SEC 10-K RAG Lab — Benchmarking Retrieval Techniques on Financial Filings

A controlled comparison of RAG techniques on a corpus of SEC 10-K annual reports. Instead of building one pipeline and calling it done, this project implements each retrieval technique in isolation, evaluates all of them against the same golden dataset with RAGAS, and reports which ones actually move the needle on dense, footnote-heavy financial text.

The question the project answers: **when does added retrieval complexity pay for itself, and when is it just latency?**

---

## Corpus

10-K annual filings for five companies spanning distinct sectors, chosen so retrieval has to handle genuinely different vocabulary and disclosure styles:

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple | Technology |
| JPM | JPMorgan Chase | Financial services |
| PFE | Pfizer | Pharmaceuticals |
| WMT | Walmart | Retail |
| XOM | ExxonMobil | Energy |

10-Ks are a deliberately hard target: hundreds of pages, heavy cross-referencing, near-identical boilerplate across companies and years, and numbers that only mean anything alongside the section they sit in. Retrieval that looks fine on clean documentation falls apart here.

---

## Architecture

```
        10-K PDFs
            │
            ▼
     pypdf extraction
            │
            ▼
   section-aware chunking          ← deterministic chunk IDs: TICKER_YEAR_INDEX
            │
            ▼
   BGE-base-en-v1.5 embeddings
            │
            ▼
        ChromaDB  (persistent)
            │
            ▼
  ┌─────────────────────────────────────────┐
  │           RETRIEVAL LAYER               │
  │                                         │
  │  query transformation                   │
  │    HyDE · multi-query · step-back ·     │
  │    decomposition                        │
  │              │                          │
  │              ▼                          │
  │  hybrid search: dense + BM25 → RRF      │
  │              │                          │
  │              ▼                          │
  │  cross-encoder reranking                │
  └─────────────────────────────────────────┘
            │
            ▼
   Claude generation + citation verification
            │
            ▼
   RAGAS evaluation vs. 150-question golden set
```

---

## Techniques implemented

Each technique lives in its own single-file implementation so it can be run, measured, and reasoned about independently.

| # | Technique | What it does | Hypothesis being tested |
|---|---|---|---|
| 1 | **Naive RAG** | Dense retrieval → stuff context → generate | Baseline everything else is measured against |
| 2 | **Prompt engineering** | Iterated system/user prompts for grounding and refusal behavior | How much of "bad RAG" is actually bad prompting |
| 3 | **Citation verification** | Answers must cite chunk IDs; citations are checked back against retrieved text | Catches confident answers built on context that was never retrieved |
| 4 | **BM25 hybrid + RRF** | Lexical BM25 fused with dense results via Reciprocal Rank Fusion | Exact tickers, line items, and defined terms need lexical matching |
| 5 | **Cross-encoder reranking** | Retrieve wide, rerank the candidate set, keep top-k | Better precision at k without shrinking the recall net |
| 6 | **HyDE** | Generate a hypothetical answer, embed *that*, retrieve against it | Bridges the vocabulary gap between casual questions and filing language |
| 7 | **Multi-query** | Fan one question into several paraphrases, union the results | Reduces sensitivity to how the question happens to be phrased |
| 8 | **Step-back prompting** | Ask a broader question first, then use it to ground the specific one | Helps when the answer needs surrounding context, not just the matching sentence |
| 9 | **Query decomposition** | Split multi-hop questions into sub-questions, retrieve per part | Comparison and multi-entity questions ("how does X differ across Y") |

---

## Evaluation

- **Golden dataset:** 150 hand-built question/answer pairs across the five filings, covering single-fact lookup, multi-hop comparison, section-scoped questions, and unanswerable questions (to test refusal rather than fabrication).
- **Framework:** RAGAS — faithfulness, answer relevancy, context precision, context recall.
- **Protocol:** every technique runs against the identical question set, the same chunk store, and the same generation model, so differences are attributable to retrieval and not to a changed variable somewhere else.

### Results

| Technique | Faithfulness | Answer relevancy | Context precision | Context recall | Latency |
|---|---|---|---|---|---|
| Naive RAG (baseline) | | | | | |
| + prompt engineering | | | | | |
| + citation verification | | | | | |
| BM25 hybrid (RRF) | | | | | |
| + cross-encoder rerank | | | | | |
| HyDE | | | | | |
| Multi-query | | | | | |
| Step-back | | | | | |
| Query decomposition | | | | | |

> Fill in from your RAGAS runs. This table is the most valuable thing in the repo — a reader who scrolls to nothing else should be able to read the ranking off it in five seconds. Bold the winner per column.

---

## Tech stack

- **Vector store:** ChromaDB (persistent client)
- **Embeddings:** BAAI/bge-base-en-v1.5
- **Generation:** Anthropic Claude
- **Lexical search:** BM25 with Reciprocal Rank Fusion
- **Reranking:** cross-encoder
- **Ingestion:** pypdf, custom section-aware chunker
- **Evaluation:** RAGAS
- **Language:** Python

---

## Ingestion details

The corpus is built from scratch rather than pulled from a prepared dataset:

- **PDF extraction** with pypdf, page by page.
- **Section-aware chunking** — chunks respect 10-K item boundaries (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements) instead of cutting blindly at a fixed token count. A risk factor split across two chunks retrieves badly; a risk factor kept whole retrieves well.
- **Deterministic chunk IDs** in `TICKER_YEAR_INDEX` format, so citations are stable across rebuilds, verification can resolve a citation to an exact chunk, and re-ingesting doesn't scramble evaluation results.
- **Persistent storage kept outside cloud-synced folders** — SQLite and file sync do not get along, and ChromaDB uses SQLite.

---

## Project structure

```
├── ingestion/
│   ├── extract.py            # pypdf → raw text per filing
│   ├── chunk.py              # section-aware chunking, deterministic IDs
│   └── embed_index.py        # BGE embeddings → ChromaDB
├── techniques/               # one self-contained file per technique
│   ├── 01_naive_rag.py
│   ├── 02_prompt_engineering.py
│   ├── 03_citation_verification.py
│   ├── 04_hybrid_bm25_rrf.py
│   ├── 05_cross_encoder_rerank.py
│   ├── 06_hyde.py
│   ├── 07_multi_query.py
│   ├── 08_step_back.py
│   └── 09_query_decomposition.py
├── evaluation/
│   ├── golden_dataset.json   # 150 questions
│   ├── run_ragas.py
│   └── results/
├── data/
│   ├── raw/                  # source 10-K PDFs
│   └── chroma/               # persistent vector store
└── README.md
```

Single-file implementations are a deliberate choice. A shared abstraction layer across nine techniques would hide exactly the differences the project exists to measure — and would make it impossible to read one technique end to end without jumping through four files.

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/MeetPatel115/<repo-name>.git
cd <repo-name>

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env          # add ANTHROPIC_API_KEY

# 4. Build the index
python ingestion/extract.py
python ingestion/chunk.py
python ingestion/embed_index.py

# 5. Run a single technique
python techniques/06_hyde.py --question "How does Pfizer describe its patent expiration risk?"

# 6. Evaluate everything
python evaluation/run_ragas.py --all
```

---

## Key takeaways

- **Complexity is not free.** Query transformation techniques multiply LLM calls per question. A technique that gains two points of faithfulness while tripling latency is a real trade-off, not an upgrade — the evaluation table makes that visible instead of assumed.
- **Lexical search still matters.** Dense retrieval alone struggles with exact identifiers and defined financial terms; the BM25 half of the hybrid is doing work that embeddings can't.
- **Chunking beats clever retrieval.** Section-aware boundaries improved results more than several downstream techniques did. Retrieval can only rank what ingestion produced.
- **Citation verification changes the failure mode.** Without it, a wrong answer looks identical to a right one. With it, unsupported claims are detectable programmatically rather than by reading every output.
- **Evaluate before optimizing.** The golden dataset was built before most techniques were implemented, which is why the comparisons mean anything.
