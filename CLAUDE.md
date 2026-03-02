# CLAUDE.md

> **Read `README.md` before asking questions.** Full architecture, all constants, engineering gotchas, adaptation guides. Still unclear? Ask the user — don't guess.

---

## Project Overview

Self-contained production RAG demo notebook for the talk "RAG Done Right in Production". Runs entirely in Google Colab — no external API keys required. Corpus: Zerodha public support FAQs (~32 sub-categories, **1,342 articles, 6,215 chunks**). Cached to `zerodha_faqs.json` (corpus) and `zerodha_faqs_embeddings.npz` (embeddings).

---

## Reference Docs

| File | Role |
|------|------|
| `README.md` | Architecture deep-dive, constants, gotchas, adaptation guide — **read before touching the notebook** |
| `PLAN.md` | Pre-change planning doc — read before implementing, update before any non-trivial change |
| `CHANGELOG.md` | Per-cell change log keyed to cell IDs — update after every non-trivial change |

---

## Change Workflow

Every non-trivial change follows three steps **in order**. Skipping any step causes PLAN.md, the code, and CHANGELOG.md to drift out of sync.

### Step 1 — Consult and update PLAN.md (before writing any code)

Read PLAN.md. If it already covers your task, confirm it still applies. If your change isn't in the plan, add it before proceeding. State: the problem, the cell IDs affected, and the exact change. Do not start implementing until PLAN.md reflects intent. Maintain clear sections in the PLAN.md that distinguishes `CURRENT PLAN` from `PLAN HISTORY`.

### Step 2 — Implement using the correct tools

See **Tool Selection** below. Always read the target cell before editing it.

### Step 3 — Update CHANGELOG.md (immediately after, never batched)

Append under the current date heading:

- Cell ID(s) affected
- What changed and why (one sentence is enough)
- `### Fixed`, `### Added`, or `### Changed` heading as appropriate along with timestamp in `YYYY-MM-DD hh:mm:ss` format

---

## Trivial vs Non-trivial

**Trivial** (skip PLAN.md / CHANGELOG.md update):

- Typo fix in a markdown cell
- Display label / print string with no downstream effect

**Non-trivial** (full 3-step workflow):

- Any code cell change that affects runtime behaviour
- Adding, removing, or reordering a cell
- Changing a query string, lambda value, threshold, or model name
- Changing demo output — even if the underlying function is correct

**Approval gate — ask the user before proceeding:**

- Adding a new pipeline stage or demo section
- Changing `EMBED_MODEL_NAME` — invalidates the index and changes eval results
- Changing the chunking strategy — same
- Deleting a cell
- Modifying `rag_query()` — it is the production path used by evaluation

---

## Tool Selection for Notebook Edits

**Never use `Edit` or `Write` on the `.ipynb` file directly** — it corrupts cell IDs and metadata.

| Task | Tool |
|------|------|
| Find a cell by content | `Grep("MMR_QUERY", glob="*.ipynb")` |
| Read cell content + IDs | `Read("rag_done_right_production.ipynb")` |
| Edit an existing cell | `NotebookEdit(cell_id=..., new_source=...)` |
| Insert a new cell | `NotebookEdit(edit_mode="insert", cell_id=<cell after which to insert>, ...)` |
| Delete a cell | `NotebookEdit(edit_mode="delete", cell_id=...)` |

The notebook's cell `"id"` fields match the slugs in the **Key Cell IDs** table below. If a cell has no human-readable slug (auto-generated ID), grep for a unique constant inside it to locate it.

---

## Running the Notebook

Primary environment: **Google Colab** (CPU only, no GPU, ~15–20 min full run).

```bash
# Local run
pip install qdrant-client==1.17.0 sentence-transformers==5.2.3 rank-bm25==0.2.2 \
            "beautifulsoup4==4.13.5" requests==2.32.4 transformers==5.0.0 \
            openai==2.23.0 tiktoken==0.12.0 tqdm==4.67.3 colorama==0.4.6 \
            networkx==3.6.1 numpy==2.0.2
jupyter notebook rag_done_right_production.ipynb
```

---

## Notebook Pipeline (execution order)

```text
Corpus Ingestion  — auto-discovery of ~32 sub-categories, cache-aware        (cells 06–10)
  → Chunking      — sentence-aware 800-char + naive comparison                (cells 12–13)
  → Embedding     — all-MiniLM-L6-v2, 384-dim, L2-normalised, Qdrant in-mem  (cells 15–16)
  → Hybrid Search — BM25 + dense + RRF fusion                                 (cells 18–20)
  → MMR           — diversity reranking λ=0.6                                 (cell 22)
  → Cross-Encoder — precision reranking                                       (cell 24)
  → Adaptive-k    — score-cliff boundary detection                            (cell 26)
  → Generation    — Flan-T5-large, sources-first prompt, 512-token limit      (cells 28–30)
  → Evaluation    — hit rate + faithfulness, production vs naive              (cells 32–35)
  → Interactive   — live query interface                                      (cells 37–38)
  → GraphRAG      — PageRank-augmented retrieval over a knowledge graph       (cells 46–50)
  → LightRAG      — local (entity-first) + global (relationship-first) paths  (cells 51–53)
```

---

## Key Cell IDs

| What to change | Cell ID | Notes |
|----------------|---------|-------|
| Re-crawl toggle | `scraper-config` | `REDOWNLOAD_DATA = True/False` |
| Re-embed toggle | `scraper-config` | `REGENERATE_EMBEDDINGS = True/False` |
| Sub-categories scraped | `scraper-config` | Auto-discovered; extend `FALLBACK_SUBCATEGORIES` for new ones |
| Chunk size / overlap | `chunker` | `max_chars` / `overlap_chars` args |
| Embedding model | `embedding-model` | Also update `EMBED_DIM` in `qdrant-setup` |
| Vector DB (swap to cloud) | `qdrant-setup` | One-line swap; client API is identical |
| BM25 tokenizer | `bm25-setup` | |
| Hybrid display dedup | `hybrid-search-demo` | `_dedup_chunks()` — keys on `chunk.title`, not `chunk.doc_id` |
| MMR λ / diversity | MMR demo | No slug — grep for `MMR_QUERY` to locate |
| Cross-encoder model | `cross-encoder` | `CE_MODEL_NAME` constant |
| Swap the LLM | `load-generator` | `build_prompt()` + `RAGResponse` in `rag-pipeline` are LLM-agnostic |
| Prompt context budget | `rag-pipeline` | `build_prompt()` — `[:400]` per-source limit |
| Evaluation queries | `golden-qa` | `GOLDEN_QA` list |
| Interactive queries | `more-queries` | |
| GraphRAG α weight | `graph-rag-demo` | `ALPHA` constant (semantic vs PageRank blend) |
| GraphRAG corpus | `graph-corpus` | `_GRAPH_CORPUS` string |
| LightRAG top-k | LightRAG demo | `top_k` in entity/relationship search |

---

## Global State Map

Use this to check whether a variable exists at a given cell position without reading the notebook.

| After cell | Key globals added |
|-----------|-------------------|
| `fetch-articles` | `articles: List[Article]` |
| `chunker` | `chunks: List[Chunk]`, `sentence_chunker()` |
| `qdrant-setup` | `client`, `COLLECTION_NAME`, `EMBED_DIM` |
| `embedding-model` | `EMBED_MODEL`, `embed()` |
| `index-chunks` | chunks indexed in Qdrant (no new Python var) |
| `bm25-setup` | `BM25_INDEX`, `bm25_search()` |
| `hybrid-search-setup` | `dense_search()`, `hybrid_search()` |
| `mmr-function` | `mmr()` |
| `cross-encoder` | `CE_MODEL`, `rerank()` |
| adaptive-k cell | `adaptive_k()` |
| `load-generator` | `generator`, `gen_tokenizer` |
| `rag-pipeline` | `build_prompt()`, `RAGResponse`, `rag_query()` |
| `golden-qa` | `GOLDEN_QA` |
| `graph-corpus` | `_GRAPH_CORPUS` |

**Safe partial re-run:** From `load-generator` onwards if only the generation layer changes. Any scraping or re-embedding change requires running from cell 06.

---

## Key Design Constraints

**No API keys by design.** Everything runs offline after `pip install` + one-time model downloads.

**Global state is strictly linear.** Cells must run top-to-bottom. After a kernel restart, re-run from `install-deps`.

**`Chunk.text` is always title-prefixed.** Format: `"{article.title}. {body_chunk}"`. This is what gets embedded and BM25-indexed. Removing the prefix degrades both retrieval paths.

**`dense_search()` and `bm25_search()` are intentionally chunk-level.** Multiple chunks per article is required for correct RRF rank fusion. Dedup by `doc_id` happens at display time (`_dedup_chunks()` in `hybrid-search-demo`) and at generation time (`rag_query()` Stage 4). Do not add dedup inside the search functions.

**`_dedup_chunks()` and `_dedup_by_title()` key on `chunk.title`, not `chunk.doc_id`.** The same Zerodha article can be scraped from two URL paths → two different `doc_id` hashes → `doc_id`-based dedup silently fails.

**Flan-T5-large encoder hard limit is 512 tokens.** `build_prompt()` puts **sources first, question last** — truncation cuts trailing sources, never the question. Per-source budget is 400 chars (~100 tokens). Do not raise `max_length` above 512. Do not add citation instructions — this causes Flan-T5 to echo "Source 1" as the answer.

**`graph-rag-demo` must use `generator()`, not `gen_model.generate()`.** Prompt format: bare context text then `Question:`, no `Context:` label prefix. Using `Context: {text}` causes Flan-T5 to return "unanswerable".

**GraphRAG and LightRAG use a self-contained Indian cities corpus (`_GRAPH_CORPUS`).** They do not depend on Zerodha data and are independent of the main RAG pipeline. They require only that `graph-corpus` has run.

**Qdrant in-memory is a one-line swap.** `QdrantClient(':memory:')` → `QdrantClient(url='...', api_key='...')` in `qdrant-setup`. Client API is identical.

**No article cap — all articles scraped.** `run-scraper` discovers all sub-categories automatically. There is no `MAX_ARTICLES` limit. The corpus cache means scraping only runs once unless `REDOWNLOAD_DATA = True`.
