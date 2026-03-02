# RAG Done Right in Production

> A complete stack of production RAG techniques — hybrid retrieval, reranking, mmr, adaptive-k, grounded generation, graph-rag with page-rank and light-rag — in a single self-contained Colab notebook. No API keys or GPU required.

This notebook is the companion to my talk **"RAG Done Right in Production"**. It implements every layer of a real production RAG stack on a live corpus of **1,342 Zerodha support FAQ articles** — then tears each layer apart to show exactly why naive RAG fails and what it takes to fix it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indranildchandra/rag-done-right-in-production/blob/main/rag_done_right_production.ipynb)

---

## Why This Repo

Most RAG tutorials give you a `dense_search → LLM` pipeline and call it done. That pipeline fails in production. This one doesn't paper over the cracks — it shows you exactly where naive RAG breaks and implements each fix layer by layer, with live output at every stage.

The complete production stack:

```text
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  HYBRID RETRIEVAL  (BM25 + Dense + RRF) │  ← exact-term + semantic, neither alone
└──────────────────────┬──────────────────┘
                       │  ~30 chunks
                       ▼
┌─────────────────────────────────────────┐
│  MMR DIVERSITY FILTER  (λ = 0.6)        │  ← kill redundant chunks before reranking
└──────────────────────┬──────────────────┘
                       │  8 diverse candidates
                       ▼
┌─────────────────────────────────────────┐
│  CROSS-ENCODER RERANK  (ms-marco)       │  ← precision pass: query ✕ doc jointly
└──────────────────────┬──────────────────┘
                       │  15 CE-scored chunks
                       ▼
┌─────────────────────────────────────────┐
│  ADAPTIVE-k  (score-gap cliff detect)   │  ← don't pad the LLM with noise
└──────────────────────┬──────────────────┘
                       │  2–8 grounded chunks
                       ▼
┌─────────────────────────────────────────┐
│  GROUNDED GENERATION  (Flan-T5-large)   │  ← cited answer, faithfulness-scored
└─────────────────────────────────────────┘
```

Then, past the main pipeline, the notebook walks through two modern retrieval techniques:

- **GraphRAG** — PageRank-augmented retrieval over a knowledge graph
- **LightRAG** — Dual-path (local entity-first + global relationship-first) graph retrieval

---

## Table of Contents

1. [Quickstart](#1-quickstart)
2. [Architecture Deep-Dive](#2-architecture-deep-dive)
   - [Corpus Ingestion](#21-corpus-ingestion)
   - [Chunking Strategy](#22-chunking-strategy)
   - [Embedding & Indexing](#23-embedding--indexing)
   - [Hybrid Search](#24-hybrid-search)
   - [MMR Diversity Filter](#25-mmr-diversity-filter)
   - [Cross-Encoder Reranking](#26-cross-encoder-reranking)
   - [Adaptive-k Boundary Detection](#27-adaptive-k-boundary-detection)
   - [Grounded Generation](#28-grounded-generation)
3. [Evaluation Framework](#3-evaluation-framework)
4. [GraphRAG & LightRAG](#4-graphrag--lightrag)
5. [Key Constants Reference](#5-key-constants-reference)
6. [Adapting to Your Use Case](#6-adapting-to-your-use-case)
7. [Engineering Insights — The Non-Obvious Stuff](#7-engineering-insights--the-non-obvious-stuff)
8. [Dependencies](#8-dependencies)
9. [Corpus Reference](#9-corpus-reference)

---

## 1. Quickstart

### Google Colab (recommended)

Click the badge above. Runtime → Run all. No GPU needed, no API keys.

| Run type | Time | Bottleneck |
|----------|------|------------|
| First run (fresh scrape + embed + eval) | **~30 min** | fetch-articles (8m) + embedding (8m) + eval (12m) |
| Cache-hit run (corpus + embeddings cached) | **~14 min** | eval loop dominates (12m); scraping + embedding skipped |

The first run scrapes 1,342 articles and computes 6,215 embeddings — both are cached after that. Set `REDOWNLOAD_DATA = False` and `REGENERATE_EMBEDDINGS = False` (the defaults) to skip them on every subsequent run.

### Local

```bash
git clone https://github.com/indranildchandra/rag-done-right-in-production
cd rag-done-right-in-production

pip install qdrant-client==1.17.0 sentence-transformers==5.2.3 rank-bm25==0.2.2 \
            "beautifulsoup4==4.13.5" requests==2.32.4 transformers==5.0.0 \
            openai==2.23.0 tiktoken==0.12.0 tqdm==4.67.3 colorama==0.4.6 \
            networkx==3.6.1 numpy==2.0.2

jupyter notebook rag_done_right_production.ipynb
```

### Force re-scrape / re-embed

```python
# In the scraper-config cell:
REDOWNLOAD_DATA       = True   # re-crawl Zerodha support site
REGENERATE_EMBEDDINGS = True   # recompute all 6,215 chunk embeddings
```

Leave both `False` for all subsequent runs — caches load in seconds.

### Interactive query (after full run)

```python
ask("What is the process to close my Zerodha account?")
# ─────────────────────────────────────────────────────────
# Q: What is the process to close my Zerodha account?
# A: You can close your Zerodha account online or offline, depending on your account type. The closure process takes 2 working days once you submit your request.
#   Pipeline: Hybrid Search → CE Rerank → Adaptive-k=2
#   Latency:  12555ms
#   Sources:
#     [1] How to close my Zerodha account?
#          https://support.zerodha.com/category/your-zerodha-account/your-profile/general-profile-questions/articles/how-do-i-close-my-zerodha-account
#     [2] What is the process to close a non individual Zerodha account?
#          https://support.zerodha.com/category/account-opening/company-partnership-and-huf-account-opening/company/articles/process-to-close-a-non-individual-zerodha-account
```

---

## 2. Architecture Deep-Dive

### 2.1 Corpus Ingestion

The scraper discovers and fetches the entire Zerodha support knowledge base automatically.

**3-phase discovery pipeline:**

```text
Phase 1 — Section Discovery
  GET /category/account-opening        ──► parse sub-category hrefs
  GET /category/trading-and-markets    ──► parse sub-category hrefs
  GET /category/funds                  ──► ...
  GET /category/console
  GET /category/mutual-funds
  GET /category/your-zerodha-account
       │
       ▼  (~32 sub-categories discovered)

Phase 2 — Article Link Extraction
  GET /category/account-opening/resident-individual  ──► 94 article hrefs
  GET /category/trading-and-markets/margins          ──► 57 article hrefs
  ...  (one request per sub-category)
       │
       ▼  (~3,400 unique article hrefs)

Phase 3 — Parallel Article Fetch
  ThreadPoolExecutor(max_workers=2)
  Each article: 3 retries, 0.5s polite delay
  Parse: <h1> title + article body (strip HTML)
       │
       ▼
  zerodha_faqs.json  (4.3 MB, 1,342 articles)
```

**Fallback safety:** If any section page is JS-rendered and returns no sub-categories, `FALLBACK_SUBCATEGORIES` — a hardcoded list of all ~30 known sub-category URLs — is used automatically. The scraper never silently drops a section.

**Output corpus stats:**

| Metric | Value |
|--------|-------|
| Articles | 1,342 |
| Sub-categories | ~32 (auto-discovered) |
| Cache file | `zerodha_faqs.json` (4.3 MB) |
| Avg article body | ~3,300 chars |

---

### 2.2 Chunking Strategy

Two strategies are implemented side by side so you can see the difference directly:

```text
Naive (fixed-size):                  Sentence-aware (production):
────────────────────────────────     ────────────────────────────────
"You can withdraw funds         "    "You can withdraw funds to your
 to your linked bank account.        linked bank account. The process
 The proce"                          takes 1–3 business days."
         ↑ split mid-sentence                  ↑ split at sentence boundary
```

**Production chunker config:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `max_chars` | 800 | ~200 tokens — fits inside `all-MiniLM-L6-v2`'s 256-token soft limit with headroom |
| `overlap_chars` | 120 | Preserves cross-sentence context at boundaries |
| Sentence boundary | `(?<=[.!?])\s+` | Respects semantic units |

**Critical design detail — title-prefixed chunks:**

Every chunk is stored as `"{article.title}. {body_chunk}"`. This applies to both what gets embedded and what BM25 indexes. Removing the title prefix measurably degrades both dense and sparse retrieval quality because the embedding model and BM25 need the article's semantic anchor to correctly represent a mid-article chunk.

**Output:**

| Metric | Value |
|--------|-------|
| Total chunks | 6,215 |
| Avg chunks/article | 4.6 |
| Avg chunk length | ~718 chars |
| Embeddings cache | `zerodha_faqs_embeddings.npz` (~18 MB, float32) |

---

### 2.3 Embedding & Indexing

**Dense index (Qdrant):**

```python
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM        = 384
COLLECTION       = "zerodha_faqs"

qdrant = QdrantClient(':memory:')          # swap to cloud with one line
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)
```

`all-MiniLM-L6-v2` is chosen for this demo because it:

- Runs on CPU at ~80ms per 100-chunk batch
- Fits inside free Colab RAM (~100 MB model size)
- Produces normalised embeddings (L2-normalised), so dot product = cosine similarity

**Sparse index (BM25):**

```python
tokenizer = lambda text: re.findall(r"[a-z0-9]+", text.lower())
bm25      = BM25Okapi([tokenizer(c.text) for c in all_chunks])
```

Simple lowercase + alphanumeric tokenisation. Deliberately minimal — BM25's strength is exact token matching for product codes and acronyms (`TPIN`, `MTF`, `GTT`, `CDSL`), not NLP sophistication.

**Embedding cache:** First run computes and saves `zerodha_faqs_embeddings.npz`. Every subsequent run loads from cache in <5 seconds, bypassing the 8-minute embedding step entirely.

---

### 2.4 Hybrid Search

The core retrieval stage combines both indexes via Reciprocal Rank Fusion (RRF).

**Why you can't choose one:**

| Query type | Dense (semantic) | BM25 (exact) |
|------------|-----------------|--------------|
| "how to add a bank account" | ✓ semantic match | ✓ token match |
| "TPIN CDSL sell order" | ~ may generalise to auth/PIN articles | ✓ exact token match on "TPIN" |
| "forgot to square off" | ✓ maps to "auto squareoff" concept | ✗ no token overlap with "auto squareoff" |
| "eDIS for demat transfer" | ~ | ✓ exact token match |

Neither index alone handles all query types. RRF fusion ensures neither blind spot causes a relevant article to miss the top-k.

**Reciprocal Rank Fusion:**

```python
def reciprocal_rank_fusion(dense_hits, bm25_hits, k=60):
    scores = defaultdict(float)
    for rank, (idx, _) in enumerate(dense_hits, 1):
        scores[idx] += 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(bm25_hits, 1):
        scores[idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

`k=60` is the smoothing constant from the original RRF paper (Cormack et al., 2009). It dampens rank differences so that a rank-1 result in one system and rank-15 result in the other still fuses well. `k=1` would almost exclusively favour rank-1 results from each system; `k=60` is more egalitarian.

**Demo output** (run live in the notebook):

```text
TPIN query — BM25 excels at exact regulatory codes
─────────────────────────────────────────────────────────────────────────────
BM25-only                  Dense-only                 Hybrid (RRF)
─────────────────────────────────────────────────────────────────────────────
[1] TPIN for selling...    [1] How to authorise...    [1] TPIN for selling...
[2] CDSL eDIS guide        [2] Two-factor auth        [2] CDSL eDIS guide
[3] Selling shares CDSL    [3] TPIN for selling...    [3] Selling shares CDSL

Insight: BM25 gets TPIN at rank 1; Dense generalises to auth/2FA articles.
Hybrid promotes BM25's signal without losing Dense's semantic coverage.

Intraday square-off query — Dense excels at paraphrase bridging
─────────────────────────────────────────────────────────────────────────────
BM25-only                  Dense-only                 Hybrid (RRF)
─────────────────────────────────────────────────────────────────────────────
[1] Square off rules       [1] Auto square off        [1] Auto square off
[2] MIS order types        [2] Intraday cutoff times  [2] MIS order types
[3] Intraday margin        [3] Square off rules       [3] Auto square off rules

Insight: "forget to square off" has zero token overlap with "auto squareoff".
Dense bridges the paraphrase; BM25 cannot. Hybrid keeps both.
```

---

### 2.5 MMR Diversity Filter

Before the expensive cross-encoder pass, Maximal Marginal Relevance removes redundant candidates.

**The problem MMR solves:** A query like "why was my order rejected?" returns 5 chunks from the same "order rejection" article (same boilerplate, different snippets). The cross-encoder then spends its entire quota scoring variations of the same content.

**Algorithm:**

```text
Given: candidates C, query q, already-selected S, λ = 0.6

Greedily select d* = argmax over C \ S of:
    λ · sim(d, q)  −  (1 − λ) · max_{s ∈ S} sim(d, s)
    ─────────────   ──────────────────────────────────
    relevance             redundancy penalty
```

**λ = 0.6** — slightly relevance-biased. This ensures:

- The single most relevant chunk is always selected first (λ > 0.5 guarantees this)
- Subsequent selections are penalised for redundancy, not ignored
- Pure diversity (λ → 0) would surface irrelevant but maximally different chunks

**Why MMR before cross-encoder, not after?**
The cross-encoder is the most expensive operation (~O(n) full-attention inference per candidate). Passing it 30 redundant chunks wastes quota. MMR prunes to few diverse candidates first (~8-10), so the cross-encoder spends its budget on genuinely distinct content.

---

### 2.6 Cross-Encoder Reranking

The bi-encoder (all-MiniLM) and cross-encoder serve fundamentally different roles:

```text
Bi-encoder (fast, pre-computed):         Cross-encoder (slow, on-demand):
──────────────────────────────────        ─────────────────────────────────
Query   ──encode──► q_vec                 (Query, Document) ──encode jointly──►
Document ──encode──► d_vec                     relevance score
cosine(q_vec, d_vec) = similarity
                                          Full attention across both inputs.
Encodes each independently.               Token-level query-document interaction.
Can't model "how well does doc            Models "how well does this exact doc
answer this exact query?"                 answer this exact query?"
```

The gap matters. A bi-encoder ranks "What is GTT?" highly for a query about "how to place a GTT order" because both mention GTT. A cross-encoder reads the full (query, doc) pair and can see that "What is GTT?" explains the concept but doesn't answer *how to place* one — and ranks it lower.

**Config:**

```python
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Trained on MS MARCO passage ranking. Works well out-of-the-box for FAQ corpora.
# final_k = 15  (feed this many candidates to adaptive-k)
```

**Two-stage retrieval is the standard production pattern:**

1. Bi-encoder: retrieve many fast (O(1), pre-computed vectors)
2. Cross-encoder: rerank few accurately (O(n), on-demand joint encoding)

Running the cross-encoder on all 6,215 chunks would take minutes. Running it on 15 candidates takes ~200ms.

---

### 2.7 Adaptive-k Boundary Detection

Fixed-k=5 fails in two directions simultaneously:

- **Too few for broad queries:** "What are all the charges I pay for trading?" has 8+ genuinely distinct relevant articles. Fixed k=5 truncates real information.
- **Too many for narrow queries:** "How to place a GTT order?" has 2–3 highly relevant articles, then a sharp drop to generic content. Fixed k=5 pads the LLM context with noise.

**Adaptive-k finds the cliff automatically:**

```text
CE scores for "how to place a GTT order":
Rank 1: GTT order placement guide         CE = 9.98  ┐
Rank 2: GTT OCO order variant             CE = 9.17  ┘ plateau
         ←── gap = 1.85 (largest gap) ───►
Rank 3: What is a GTT order?              CE = 7.32  ┐
Rank 4: GTT cancellation                  CE = 7.10  │ drop
Rank 5: Generic Kite order guide          CE = 6.95  ┘

Adaptive-k = 2   (cliff at rank 2→3)
Fixed-k    = 5   (sends ranks 3–5 to LLM — concept articles, not how-to)
```

```text
CE scores for "what are all the charges for trading?":
Rank 1: Equity delivery charges           CE = 9.21  ┐
Rank 2: Intraday brokerage                CE = 8.95  │
Rank 3: F&O charges breakdown             CE = 8.71  │ plateau
Rank 4: STT and exchange charges          CE = 8.44  │
Rank 5: GST on brokerage                  CE = 8.19  ┘
         ←── largest gap is only 0.26 ───►

Adaptive-k = 5   (no significant cliff — send all)
```

**Algorithm:**

```python
def adaptive_k(ranked, min_k=2, max_k=8):
    scores = [score for _, score in ranked[:max_k]]
    gaps   = [scores[i] - scores[i+1] for i in range(len(scores) - 1)]
    cliff  = gaps.index(max(gaps)) + 1        # +1: cliff is after rank cliff
    k      = max(min_k, min(cliff, max_k))
    return ranked[:k]
```

**On a corpus of 1,342 Zerodha articles:** adaptive-k sends an average of **3.2 chunks** to the LLM vs naive fixed k=5. That's 36% fewer tokens with no hit-rate loss.

---

### 2.8 Grounded Generation

**Model:** `google/flan-t5-large` — 770M param instruction-tuned seq2seq. Runs on CPU with no API key, produces coherent citations-grounded answers.

**Prompt design has three non-obvious constraints for seq2seq models:**

#### Constraint 1: Sources first, question last

```text
Source 1: How to withdraw funds. To withdraw funds from your Zerodha account,
          go to Console > Funds > Withdraw...

Source 2: Fund withdrawal timeline. Withdrawals are processed on the same day
          if submitted before 11 AM...

Question: How long does a fund withdrawal take?
Answer:
```

Flan-T5 encodes left-to-right. If context overflows the 512-token encoder limit, the truncation cuts the **end**. Sources-first means the question is always fully seen even when context is long — the model always knows what it's being asked.

Reversing this (question first, then sources) causes context to be truncated → model answers from memory rather than retrieved sources → hallucination.

#### Constraint 2: No citation instruction

Adding "Answer based on Source 1, cite it" causes the model to echo `"Source 1"` as the full answer — a known seq2seq training artefact. Citations are shown separately via `RAGResponse.sources`.

#### Constraint 3: Full article body, not chunk fragment

```python
# In rag_query() Stage 4:
body = _article_bodies.get(chunk.doc_id, chunk.text)    # full article body
text = f"{chunk.title}. {body}"[:400]                    # per-source budget
```

A chunk fragment mid-article is often an incomplete sentence. Using the full article body (via `_article_bodies` lookup, built once at pipeline assembly) ensures the model sees coherent source text. The 400-char per-source budget caps total context at ~300 tokens for 5 sources.

**`RAGResponse` dataclass:**

```python
@dataclass
class RAGResponse:
    query:      str
    answer:     str
    sources:    List[Chunk]     # ordered, deduplicated by doc_id
    latency_ms: float
```

The response type is **LLM-agnostic** — swap Flan-T5 for GPT-4.1 or Claude Sonnet and `RAGResponse` doesn't change. See [Adapting to Your Use Case](#6-adapting-to-your-use-case).

---

## 3. Evaluation Framework

Production RAG evaluation requires two orthogonal metrics. Neither alone is sufficient.

### Hit Rate @ k (Retrieval quality)

```python
def hit_at_k(retrieved_chunks, expected_title_fragments, k=5):
    return any(
        frag.lower() in chunk.title.lower()
        for frag in expected_title_fragments
        for chunk in retrieved_chunks[:k]
    )
```

**Measures:** Did the right article make it into the top-k?
**Blind spot:** The article was retrieved but the answer was hallucinated.

### Faithfulness Score (Generation grounding)

```python
def faithfulness_score(answer, source_chunks, n=3):
    answer_trigrams = set(zip(*[answer.split()[i:] for i in range(n)]))
    source_text     = " ".join(c.text for c in source_chunks)
    source_trigrams = set(zip(*[source_text.split()[i:] for i in range(n)]))
    return len(answer_trigrams & source_trigrams) / max(len(answer_trigrams), 1)
```

**Measures:** What fraction of the answer's 3-word phrases appear verbatim in sources?
**Range:** 0.0 (fully hallucinated) → 1.0 (fully grounded)
**Why trigrams?** Single words match trivially (common words). Full sentence matching is too strict (reasonable paraphrase fails). Trigrams correlate well with human faithfulness judgments.
**Blind spot:** Verbatim copy from sources scores 1.0 even if it doesn't answer the question.

### Golden QA Set

53 hand-crafted query–answer pairs covering all 30 sub-categories (1–5 queries per sub-category). Includes edge cases: regulatory acronyms, multi-step answers, NRI-specific queries, corporate action questions.

**Known limitation:** Synthetic QA written by someone who knows the corpus is optimistically biased. Real user queries are messier, more ambiguous, and multi-intent. Treat evaluation scores as upper bounds, not ground truth.

### Production vs Naive comparison

| Metric | Naive (dense-only, k=5) | Production Pipeline |
|--------|------------------------|---------------------|
| Hit Rate @ 5 | ~75% | ~92% |
| Faithfulness | ~0.55 | ~0.68 |
| Avg chunks to LLM | 5 (fixed) | 3.2 (adaptive) |
| Latency | ~900ms | ~1,400ms |

The ~500ms latency cost of the production pipeline (cross-encoder + adaptive-k) buys +17pp retrieval accuracy and +13pp generation grounding.

---

## 4. GraphRAG & LightRAG

### 4.1 GraphRAG — PageRank-Augmented Retrieval

Standard RAG ranks chunks by semantic similarity to the query. This fails for queries that require knowing *which entities matter* — not just which chunks are semantically similar.

**The problem:** A query about "BSE-listed companies and their headquarters" semantically matches a chunk that says "The city hosts the headquarters of major Indian conglomerates..." — but that chunk uses a pronoun ("The city") rather than naming Mumbai. A flat-RAG pipeline hands the LLM context it can't extract an answer from.

**GraphRAG solution:**

```text
Knowledge Graph (Indian cities domain)
──────────────────────────────────────
  Mumbai ──HOSTS──► BSE
  Mumbai ──HOSTS──► NSE
  Mumbai ──HOSTS──► Reserve Bank of India
  Reliance Industries ──LISTED_ON──► BSE
  Reliance Industries ──HEADQUARTERS_OF──► Mumbai
  Tata Group ──LISTED_ON──► BSE
  Tata Group ──HEADQUARTERS_OF──► Mumbai
  ...

PageRank centrality:
  Bombay Stock Exchange (BSE):  PR = 0.159  ← high: many edges point here
  Mumbai:                       PR = 0.053  ← medium
  Reliance Industries:          PR = 0.059  ← medium
  Kolkata:                      PR = 0.031  ← low (fewer connections to finance graph)
```

**Re-ranking formula:**

```text
final_score = α · semantic_similarity + (1 − α) · max_pagerank_in_chunk

α = 0.6  (semantic weight)
```

High-PR (page-ranked) entities appear in sentences that name them explicitly ("Mumbai is the financial capital of India and home to the Bombay Stock Exchange..."). GraphRAG promotes these sentences — the LLM now sees named entities rather than pronouns.

**Graph traversal (deterministic, no LLM):**

```python
# Direct structured answer — zero hallucination:
bse_companies = [u for u, v, d in G.edges(data=True)
                 if v == "BSE" and d["rel"] == "LISTED_ON"]
for company in bse_companies:
    hq = [u for u, v, d in G.edges(data=True)
          if v == company and d["rel"] == "HEADQUARTERS_OF"]
    print(f"{company} → HQ: {hq[0]}")
# Reliance Industries → HQ: Mumbai
# Tata Group → HQ: Mumbai
```

For structured questions about entity relationships, graph traversal gives exact, non-hallucinatory answers that no amount of prompt engineering can guarantee from a flat LLM.

---

### 4.2 LightRAG — Dual-Path Graph Retrieval

LightRAG separates retrieval into two orthogonal paths and fuses them:

```text
                           ┌─ LOCAL PATH (entity-first) ───────────────────┐
                           │  Query embedding → cosine search on entity    │
                           │  vectors → top-3 entities → 1-hop expansion   │
                           │  Best for: concrete factual questions         │
Query ──embed──► q_vec ──► │                                               │──► HYBRID
                           │  Query embedding → cosine search on           │    (union + dedup)
                           │  relationship type vectors → thematic         │     
                           │  cluster retrieval                            │
                           │  Best for: broad conceptual questions         │
                           └─ GLOBAL PATH (relationship-first) ────────────┘
```

**When each path wins:**

| Query | Winner | Why |
|-------|--------|-----|
| "BSE listed companies and their headquarters" | **Local** | Concrete entities — BSE, Reliance, Tata all in entity DB; 1-hop expansion gets LISTED_ON + HEADQUARTERS_OF directly |
| "cultural and economic relationships between Indian cities" | **Global** | Abstract theme — relationship clusters (CONNECTED_TO, HOUSES) span all three cities; local entities are too granular |
| "How does financial dominance connect Mumbai to listed companies" | **Hybrid** | Needs entity precision (Mumbai, BSE) AND relationship breadth (HOSTS, LISTED_ON) |

**Live demo output (Query 1 — Local path):**

```text
── LOCAL  (entity-first, 1-hop expansion) ──
  [Company] Reliance Industries: The city hosts the headquarters of major Indian...
    ↳ LISTED_ON → [Institution] BSE: Mumbai is the financial capital of India...
    ↳ HEADQUARTERS_OF → [City] Mumbai: Mumbai is the financial capital of India...
  [Company] Tata Group: The city hosts the headquarters of major Indian...
  [Institution] Reserve Bank of India: The city hosts the headquarters of major...
```

The notebook implements the full LightRAG pipeline manually using `embed_model` (already loaded) and `G` (the NetworkX graph from the GraphRAG section) — no Ollama, no LLM server, no new dependencies.

---

## 5. Key Constants Reference

All production-relevant constants are in the `scraper-config` and individual demo cells.

| Constant | Value | Cell | Change to... |
|----------|-------|------|--------------|
| `REDOWNLOAD_DATA` | `False` | `scraper-config` | `True` to re-crawl corpus |
| `REGENERATE_EMBEDDINGS` | `False` | `scraper-config` | `True` to re-embed |
| `max_chars` | `800` | `chunker` | Increase if you use a larger embedding model |
| `overlap_chars` | `120` | `chunker` | Increase for longer, denser documents |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | `embedding-model` | Any sentence-transformer |
| `EMBED_DIM` | `384` | `embedding-model` | Must match model's output dim |
| BM25 tokeniser | `r"[a-z0-9]+"` | `bm25-setup` | Add stemming for noisy corpora |
| Dense `top_k` | `30` | `hybrid-search-fn` | Increase for larger corpora |
| BM25 `top_k` | `30` | `hybrid-search-fn` | Keep in sync with dense `top_k` |
| RRF `k` | `60` | `hybrid-search-fn` | Rarely needs tuning; 60 is empirically robust |
| MMR `final_k` | `8` | MMR cell | Increase for broader queries |
| MMR `λ` | `0.6` | MMR cell | `[0.5, 0.7]` — tune on your golden QA set |
| `CE_MODEL_NAME` | `ms-marco-MiniLM-L-6-v2` | `cross-encoder` | Larger: `ms-marco-MiniLM-L-12-v2` |
| CE `final_k` | `15` | `cross-encoder` | Increase if adaptive-k too often hits `max_k` |
| Adaptive `min_k` | `2` | `adaptive-k` | Set to `1` for high-precision pipelines |
| Adaptive `max_k` | `8` | `adaptive-k` | Set to `10–12` for document-heavy corpora |
| `GEN_MODEL` | `google/flan-t5-large` | `load-generator` | Any seq2seq or autoregressive LLM |
| Per-source budget | `400` chars | `rag-pipeline` | Scale up when swapping to larger-context LLM |
| Golden QA set size | `53` queries | `golden-qa` | Add more; cover all sub-categories |
| GraphRAG `ALPHA` | `0.6` | GraphRAG cell | Tune: higher = more semantic, lower = more graph |

---

## 6. Adapting to Your Use Case

### Swap the vector database (1 line)

```python
# Colab / local dev:
qdrant = QdrantClient(':memory:')

# Production (Qdrant Cloud):
qdrant = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your_api_key"
)
# The client API is identical — no other code changes.
```

### Swap the LLM (1 function)

The `generator()` function is a thin wrapper. Replace the body:

```python
# Current (local Flan-T5-large, no API key):
def generator(prompt: str, max_new_tokens: int = 256, **kwargs) -> list:
    inputs  = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text    = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [{"generated_text": text}]

# OpenAI GPT-4.1-mini:
def generator(prompt: str, max_new_tokens: int = 256, **kwargs) -> list:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens
    )
    return [{"generated_text": response.choices[0].message.content}]

# Anthropic Claude:
def generator(prompt: str, max_new_tokens: int = 256, **kwargs) -> list:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_new_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return [{"generated_text": response.content[0].text}]
```

`build_prompt()` and `RAGResponse` are completely LLM-agnostic. Everything upstream (retrieval, reranking, adaptive-k) stays identical.

When swapping to a larger-context LLM, also increase the per-source budget in `build_prompt()`:

```python
text = f"{chunk.title}. {body}"[:400]   # Flan-T5: 400 chars (~100 tokens)
text = f"{chunk.title}. {body}"[:2000]  # GPT-4: 2000 chars (~500 tokens)
```

### Swap the embedding model

```python
EMBED_MODEL_NAME = "text-embedding-3-small"  # OpenAI, 1536-dim
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # open-source, 1024-dim, top MTEB scores
EMBED_MODEL_NAME = "intfloat/e5-large-v2"    # open-source, 1024-dim
```

When changing the embedding model:

1. Update `EMBED_DIM` to match
2. Update `max_chars` in chunker — larger models handle longer inputs (`max_chars=1500` for 512-token models)
3. Delete `zerodha_faqs_embeddings.npz` and set `REGENERATE_EMBEDDINGS = True`

### Use your own corpus

Replace the scraper section (cells 6–10) with your own data loading:

```python
# Option 1: Load from JSON
articles = [Article(url=d["url"], title=d["title"], body=d["body"])
            for d in json.load(open("my_corpus.json"))]

# Option 2: Load from database
articles = [Article(url=row.url, title=row.title, body=row.body)
            for row in db.query("SELECT url, title, body FROM articles")]

# Option 3: Load from files
articles = []
for path in Path("docs/").glob("**/*.md"):
    articles.append(Article(
        url=str(path),
        title=path.stem.replace("-", " ").title(),
        body=path.read_text()
    ))
```

Everything from the chunker cell onwards works on `articles: List[Article]` — no changes needed downstream.

### Production evaluation (beyond trigram faithfulness)

For real production systems, replace or supplement the built-in metrics:

```python
# RAGAS (LLM-as-judge, more nuanced):
from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, faithfulness
result = evaluate(dataset, metrics=[answer_correctness, faithfulness, context_precision])

# TruLens (dashboard + continuous monitoring):
from trulens_eval import TruChain, Tru
tru = Tru()
tru_recorder = TruChain(pipeline, app_id="rag-done-right-in-production-v1")

# DeepEval (unit tests for LLM outputs):
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
metric = FaithfulnessMetric(threshold=0.7)
test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=sources)
```

---

## 7. Engineering Insights — The Non-Obvious Stuff

### Chunk-level search, document-level dedup (don't conflate them)

`dense_search()` and `bm25_search()` intentionally return **multiple chunks per article**. This is required for correct RRF fusion — the ranking signal from multiple relevant chunks of the same article correctly boosts that article in the fused ranking. Adding dedup inside the search functions would destroy this signal.

Dedup happens in two later, correct places:

- **Display time:** `_dedup_chunks()` by article title — one result per article in demos
- **Generation time:** `rag_query()` Stage 4 by `doc_id` — one source slot per article in the LLM prompt

### Why `doc_id` dedup fails on this corpus (and how to fix it)

`doc_id` is computed as `hash(article.url)`. The same Zerodha FAQ article is sometimes discoverable from two different URL paths (sub-category URL vs. search URL). Same content, different `doc_id`. Deduplication by `doc_id` silently fails — the same article appears twice in results.

**Fix:** Dedup by `chunk.title` instead. Article titles are unique; URL paths aren't guaranteed to be.

This applies anywhere dedup is used: `_dedup_chunks()` in the hybrid demo, `_dedup_by_title()` in the adaptive-k demo.

### The Flan-T5-large "unanswerable" failure mode

Flan-T5-large returns "unanswerable" when:

1. The context doesn't contain the answer in a format it recognises as extractable
2. Greedy decoding assigns higher probability to "unanswerable" than the actual answer

Greedy decoding (no beam search) is the cause — it greedily picks "unanswerable" when uncertain rather than exploring alternatives. For complex questions or long context, use beam search: `gen_model.generate(**inputs, max_new_tokens=256, num_beams=4)`.

The `generator()` function in this notebook uses greedy decoding by design (faster, deterministic for demos). Swap to beam search for higher answer quality in production.

### The "Source 1 echo" bug

Adding any citation instruction to the Flan-T5 prompt — "Answer using Source 1", "Cite your sources", "Reference the article" — causes the model to output the instruction phrase literally as its answer. This is a seq2seq fine-tuning artefact: the model learned "Source 1" as a high-probability token after citation-style prompts.

Solution: Never add citation instructions to the generation prompt. Show sources via `RAGResponse.sources` separately.

### Why the golden QA set has an optimism bias

53 queries written by someone who knows the Zerodha corpus will always over-represent "clean" queries — well-formed, unambiguous, matching the exact terminology in the articles. Real support queries are messier: typos, mixed languages, multi-intent ("how do I add a bank account and also change my nominee?"), implicit context ("why isn't it working?").

Production eval scores will be 10–20pp lower than golden set scores. Budget for this when setting SLOs.

### BM25 tokenisation matters more than you think

The demo uses `r"[a-z0-9]+"` — simple lowercase alphanumeric. For Zerodha FAQs this works because the important exact-match tokens are acronyms (`TPIN`, `MTF`, `BTST`) and product codes that survive this tokenisation.

For corpora with morphologically rich language (Hindi, Marathi, Tamil) or domain jargon with common prefixes/suffixes, add stemming or use a subword tokeniser. `rank-bm25` accepts any tokenisation function.

### Adaptive-k min_k=2 prevents empty context

`min_k=2` ensures the LLM always receives at least 2 source chunks, even when the top-1 CE score has a massive cliff to rank 2. Without this floor, highly specific queries ("What is the ISIN for SGB?" — only one article in the corpus) would return 1 chunk, increasing hallucination risk for multi-part answers.

---

## 8. Dependencies

| Package | Pinned Version | What it's used for |
|---------|---------------|---------------------|
| `qdrant-client` | `1.17.0` | In-memory (or cloud) vector database |
| `sentence-transformers` | `5.2.3` | `all-MiniLM-L6-v2` embeddings + cross-encoder |
| `rank-bm25` | `0.2.2` | BM25Okapi sparse retrieval |
| `beautifulsoup4` | `4.13.5` | HTML parsing for corpus scraping |
| `requests` | `2.32.4` | HTTP client |
| `transformers` | `5.0.0` | Flan-T5-large tokenizer + model |
| `openai` | `2.23.0` | Optional — for LLM swap |
| `tiktoken` | `0.12.0` | Optional — OpenAI tokenizer |
| `tqdm` | `4.67.3` | Progress bars |
| `colorama` | `0.4.6` | Coloured terminal output |
| `networkx` | `3.6.1` | Knowledge graph construction + PageRank for GraphRAG/LightRAG |
| `numpy` | `2.0.2` | Array ops for embeddings, cosine similarity, score arithmetic |

Versions are pinned for reproducibility on Python 3.10+ and Google Colab (as of the last tested run).

---

## 9. Corpus Reference

### Source

[Zerodha Support](https://support.zerodha.com) — publicly available FAQ knowledge base for India's largest retail brokerage. All data is scraped from public URLs with polite rate-limiting.

### Size

| Metric | Value |
|--------|-------|
| Articles | 1,342 |
| Sub-categories | ~32 |
| Chunks | 6,215 |
| Avg chunk length | ~718 chars |
| Corpus cache | `zerodha_faqs.json` (4.3 MB) |
| Embeddings cache | `zerodha_faqs_embeddings.npz` (~18 MB) |

### Cache Files

| File | Size | Contents | Regenerate |
|------|------|----------|------------|
| `zerodha_faqs.json` | ~4.3 MB | 1,342 articles — `url`, `title`, `body`, `category` per entry | `REDOWNLOAD_DATA = True` in `scraper-config` |
| `zerodha_faqs_embeddings.npz` | ~18 MB | Float32 array of 6,215 chunk embeddings (384-dim), aligned to chunked corpus | `REGENERATE_EMBEDDINGS = True` in `scraper-config` |

**`zerodha_faqs.json` entry shape:**

```json
{
  "url":      "https://support.zerodha.com/category/.../articles/...",
  "title":    "How to withdraw funds from Zerodha",
  "body":     "To withdraw funds, navigate to Console > Funds...",
  "category": "Fund Withdrawal"
}
```

---

### Categories covered

`account-opening` (resident, NRI, minor, corporate) · `your-zerodha-account` (profile, bank details, nomination, share transfer) · `trading-and-markets` (FAQs, margins, charts & orders, general Kite, charges, IPO, alerts) · `funds` (adding funds, withdrawal, bank accounts, mandates) · `console` (portfolio, corporate actions, ledger, reports, segments) · `mutual-funds` (understanding MF, payments & orders, NPS, fixed deposits, Coin features)

### Swap the corpus

```python
# zerodha_faqs.json schema — adapt your loader to produce this shape:
[
  {
    "url":      "https://support.zerodha.com/category/.../articles/...",
    "title":    "How to withdraw funds from Zerodha",
    "body":     "To withdraw funds, navigate to Console > Funds...",
    "category": "Fund Withdrawal"
  },
  ...
]
```

Any corpus that fits this schema (internal wiki, product docs, support tickets, research papers) will work with the rest of the pipeline unchanged.

---

## Production Adaptation Checklist

Before taking this pipeline to production, tick off:

**Infrastructure**

- [ ] Qdrant Cloud or self-hosted (swap `':memory:'` for cloud URL)
- [ ] Embedding model fine-tuned or validated on your domain
- [ ] LLM with sufficient context window for your source budgets
- [ ] Async request handling (FastAPI + background tasks, or a queue)

**Data**

- [ ] Golden QA set derived from real user queries (not synthetic)
- [ ] Corpus versioning — know what's in your vector DB at any point in time
- [ ] Document update pipeline — handle edits/deletions in source content

**Evaluation**

- [ ] Hit@k + faithfulness as baselines
- [ ] LLM-as-judge for answer quality (RAGAS, DeepEval, or custom)
- [ ] Latency SLOs defined and measured per pipeline stage
- [ ] Evaluation runs automatically on every model/pipeline change

**Observability**

- [ ] Query logging (query, retrieved chunks, generated answer, latency, user signal)
- [ ] Retrieval failure detection (low max CE score → flag for review)
- [ ] Hallucination monitoring (faithfulness below threshold → alert)

**Safety**

- [ ] Input validation (length, injection attempts)
- [ ] Output filtering (PII, sensitive content)
- [ ] Rate limiting on generation endpoint

---

## Slide Deck

<a href="https://docs.google.com/presentation/d/1NolhGaGtHTzUVNeSyr1afO0YmF9nkr0D/mobilepresent?fbclid=PAVERFWAQKok1leHRuA2FlbQIxMABzcnRjBmFwcF9pZA8xMjQwMjQ1NzQyODc0MTQAAacGrvYDVDLUi2LpsLU7o4gakb12GVDYIGnYnIaV7P2qWbIen9Eu78vqzKIRRg_aem_87H23qUBfrhNmAcuHb30yQ&slide=id.p1"><img src="slide-deck-qr-code.png" alt="Scan to open slides" width="180"/></a>

---

*Built for the talk "RAG Done Right in Production" by [Indranil Chandra](https://indranilchandra.com).*
