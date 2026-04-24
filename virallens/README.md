# Virallens AI Engineer Assignment

> **Q1** — Multi-Agent RAG  (LangChain + LangGraph + ChromaDB)
> **Q2** — Custom Domain Embedder  (Triplet loss + Clustering)

---

## Project Structure

```
virallens/
├── run.py                      ← UNIFIED LAUNCHER — start here
├── .env                        ← add GROQ_API_KEY here
├── .env.example                ← template
├── data/                       ← place your 5 PDF research papers here
│   ├── paper1.pdf
│   └── ...
│
├── q1_multiagent_rag/
│   ├── agents/
│   │   ├── retriever_agent.py  # Hybrid ChromaDB+BM25 retrieval + metadata filter
│   │   ├── grader_agent.py     # LLM relevance scoring (Groq llama3-8b)
│   │   ├── generator_agent.py  # Cited answer synthesis (Groq llama3-70b)
│   │   └── validator_agent.py  # Hallucination detection (NLI-style)
│   ├── graph/
│   │   ├── workflow.py         # LangGraph StateGraph + conditional retry loops
│   │   └── state.py            # Typed AgentState with Annotated reducers
│   ├── utils/
│   │   ├── vector_store.py     # ChromaDB hybrid store (dense+BM25+RRF)
│   │   ├── chunker.py          # Semantic + recursive hybrid chunker
│   │   └── rate_limiter.py     # Token-bucket + exponential backoff
│   ├── tests/
│   │   └── test_pipeline.py    # 28 unit + integration tests
│   ├── config.py               # All settings (paths resolve to project root)
│   ├── main.py                 # Q1 standalone entry point
│   ├── requirements.txt
│   └── Dockerfile
│
└── q2_custom_embedder/
    ├── models/custom_embedder.py      # TripletLoss fine-tuned SentenceTransformer
    ├── utils/preprocessor.py          # PDF cleaning, sentence segmentation
    ├── utils/triplet_generator.py     # BM25 hard-negative mining
    ├── train_embedder.py
    ├── evaluate_embedder.py           # 5 metrics + 6 visualizations
    ├── cluster_analysis.py            # KMeans + DBSCAN + Agglomerative
    ├── main.py
    └── FINDINGS.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
cd virallens/q1_multiagent_rag
pip install -r requirements.txt

cd ../q2_custom_embedder
pip install -r requirements.txt
```

### 2 — Add your API key

```bash
# Create .env in the project root (same folder as run.py)
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```
Get a free key: https://console.groq.com/keys

### 3 — Add your PDFs

Drop your 5 research paper PDFs into `data/`.

### 4 — Run

```bash
# From project root (virallens/)

python run.py q1                          # RAG demo queries
python run.py q1 --query "What methods?"  # Single query
python run.py q1 --interactive            # Interactive REPL
python run.py q1 --rebuild-index          # Force re-index

python run.py q2                          # Train embedder + full eval
python run.py q2 --skip-training          # Eval only
python run.py q2 --epochs 5              # Custom epochs

python run.py both                        # Q2 then Q1
```

### Docker

```bash
# Build from project root
docker build -f q1_multiagent_rag/Dockerfile -t virallens .
docker run --env-file .env -v $(pwd)/data:/app/data virallens q1
```

---

## Q1 — Agent Roles

| Agent | Model | Role |
|---|---|---|
| **RetrieverAgent** | — | ChromaDB hybrid search: dense cosine + BM25 + RRF + metadata filter |
| **GraderAgent** | llama3-8b | Scores each chunk 0–1 for query relevance. Filters below threshold |
| **GeneratorAgent** | llama3-70b | Synthesizes grounded answer with inline citations |
| **ValidatorAgent** | llama3-8b | NLI-style hallucination check. Triggers regeneration loop if needed |

## Q1 — Retrieval Architecture (ChromaDB)

```
                ┌──────────────────────────────────────────┐
  Query ───────►│  ChromaDB  where= filter  (metadata)      │
                │  source / page_min / page_max  BEFORE ANN  │
                └──────────────────────┬───────────────────┘
                   ┌───────────────────┼──────────────────────┐
                   ▼                                          ▼
      ┌────────────────────────┐            ┌──────────────────────────┐
      │  Dense (Chroma cosine) │            │  Sparse BM25             │
      │  HNSWlib HNSW index    │            │  exact term matching      │
      └──────────┬─────────────┘            └──────────┬───────────────┘
                 └──────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  RRF: score = Σ 1/(60 + rank) │
                    └───────────────────────────────┘
```

**Why ChromaDB beats FAISS for this task:**
- Metadata filtering runs **inside** the DB (not post-retrieval) — you get exactly top-K results that satisfy source/page constraints, not top-K×N then filtered down
- Native `$eq $in $gte $lte $and $or` operators on every metadata field
- Persistent by default — no manual pickle/FAISS file management
- HNSW index (faster than FAISS FlatIP at scale)

## Q1 — LangGraph Flow

```
[Query] → [RetrieverAgent] → [GraderAgent]
                                   │
              ┌────────────────────┤
              │ no relevant chunks │ has chunks
              ▼                    ▼
       [QueryRewriter]      [GeneratorAgent]
              │                    │
              └──→ [Retriever]  [ValidatorAgent]
                                   │
              ┌────────────────────┤
              │ hallucination      │ grounded
              ▼                    ▼
       [GeneratorAgent]          [END]
```

## Q2 — Custom Embedder

Fine-tunes `all-MiniLM-L6-v2` with `TripletLoss` on domain-specific triplets mined via BM25 hard-negative selection. Evaluates against baseline on 5 metrics and 6 visualizations. See `q2_custom_embedder/FINDINGS.md` for detailed analysis.
