# Hybrid Search + OpenAI Embeddings for claude-context-local

**Date:** 2026-03-15
**Status:** Approved
**Repo:** redacted-org/claude-context-local (fork of FarhanAliRaza/claude-context-local)

## Context

EXP-008 evaluated 13 semantic code search tools to complement codebase-memory-mcp's structural graph. Two tools were tested in depth:

- **claude-context** (zilliztech, 5,639 stars): Hybrid BM25+vector via Milvus/Zilliz Cloud + OpenAI embeddings. Scored 9/10 on a 10-query battery against mcp-servers repo.
- **claude-context-local** (FarhanAliRaza, 200 stars): Pure vector search via FAISS + local MiniLM embeddings. Scored 6/10 on the same battery.

The 3-query gap (S05 "env vars", S07 "logging", S10 "middleware") was caused by two factors: (1) missing BM25 keyword search, and (2) lower-quality local embeddings (384-dim MiniLM vs 1536-dim OpenAI).

We forked claude-context-local because it's Python (matches our stack), fully local, and the changes needed are well-scoped. We can't use claude-context directly because it sends source code to Zilliz Cloud.

## Hypothesis

Adding FTS5 BM25 keyword search + OpenAI embeddings + RRF fusion to claude-context-local will match claude-context's 9/10 query quality while keeping source code off third-party infrastructure.

## Design

### 1. Configurable Embedding Provider

Add an `OpenAIEmbeddingModel` class alongside the existing `SentenceTransformerModel`. Configured via env var:

```
EMBEDDING_PROVIDER=openai    -> OpenAI text-embedding-3-small (1536-dim, API call)
EMBEDDING_PROVIDER=local     -> SentenceTransformerModel all-MiniLM-L6-v2 (384-dim, local)

Default: openai
```

**Files changed:**
- New: `embeddings/openai_embedder.py` - implements `EmbeddingModel` base class, uses `httpx` to call OpenAI `/v1/embeddings`. Batches up to 2048 texts per request.
- Modified: `embeddings/embedding_models_register.py` - add OpenAI and MiniLM to the registry
- Modified: `embeddings/embedder.py` - read `EMBEDDING_PROVIDER` env var, select model
- Modified: `pyproject.toml` - add `httpx` dependency

**Constraint:** Changing embedding provider after indexing requires a full re-index (different vector dimensions). The index should store which provider/model was used and warn on mismatch at startup.

**Research basis:** OpenAI `text-embedding-3-small` is the same model that produced claude-context's 9/10 results. The `3-large` variant (3072-dim) doubles memory for marginal gain on code search benchmarks.

### 2. BM25 Keyword Search via SQLite FTS5

Add a FTS5 virtual table to the existing `metadata.db` SQLite database.

**Schema:**
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id,
    content,
    file_path,
    name,
    tokenize='porter unicode61'
);
```

The `porter unicode61` tokenizer provides Porter stemming ("logging" matches "logged", "validates" matches "validation") and Unicode-aware tokenization. FTS5 splits on punctuation, so `os.environ` becomes tokens `["os", "environ"]` and `logging.getLogger` becomes `["logging", "getlogger"]`.

**Index time:** Insert chunk text + metadata into FTS5 alongside the existing FAISS vector insert and SQLite metadata insert. Same transaction, same lifecycle.

**Search time:** `SELECT chunk_id, rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 50`

**Incremental indexing:** When the Merkle-tree change detector finds modified files, delete from `chunk_fts` and re-insert alongside the existing FAISS delete/re-insert.

**Files changed:**
- Modified: `search/indexer.py` - FTS5 table creation, insert, delete, search methods
- Modified: `search/searcher.py` - add BM25 search call and fusion

**Research basis:** Every major search platform (Elasticsearch, OpenSearch, Azure AI Search, Milvus) uses BM25 via inverted index as the keyword component of hybrid search. FTS5 is SQLite's built-in inverted index with BM25 ranking. Zero new dependencies. Handles 50K+ documents without performance issues (persistent, disk-backed, B-tree indexed).

**Alternative considered:** `rank_bm25` Python library (in-memory). Rejected because it requires rebuilding the full index on every server start - at Corsair scale (50K+ chunks) this adds 5-15s startup latency. FTS5 is persistent and supports incremental updates.

**Alternative considered:** Tantivy (Rust full-text engine via Python bindings). Rejected as overkill - adds ~50MB dependency for code-aware tokenization that vector search already handles. FTS5's generic tokenizer is sufficient for the keyword queries where BM25 needs to help (exact identifiers like `os.environ`, `logging`, `middleware`). Tantivy remains an option if FTS5 tokenization proves insufficient during testing.

### 3. Reciprocal Rank Fusion (RRF)

Fuse BM25 and vector results using RRF rather than weighted score combination.

**Formula:** `RRF(d) = sum(1 / (k + rank_i(d)))` for each list containing document d.

**Implementation:**
```python
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, (chunk_id, ...) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1/(k + rank + 1)
    for rank, (chunk_id, ...) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Parameters:**
- `k=60` (default, configurable via `FUSION_K` env var)
- Retrieve 50 candidates from each source (10x the typical final K of 5)

**Research basis:** RRF is used by Milvus (what claude-context uses), Elasticsearch, OpenSearch, Azure AI Search, Chroma, and Google Vertex AI. It works on rank positions rather than raw scores, avoiding the score normalization problem (BM25 scores are unbounded, cosine similarity is 0-1).

**Competing approach:** Pinecone's research (ACM-published) found convex combination (weighted score sum) outperforms RRF in some benchmarks. However, convex combination requires normalizing BM25 and cosine scores to the same scale, which is fragile and domain-dependent. RRF is zero-shot (no tuning required) and is what claude-context's Milvus backend uses, making it the closest match to our comparison target.

**Future option:** If RRF underperforms, add convex combination as `FUSION_METHOD=weighted` for A/B testing. The fusion step is ~30 lines of code and pluggable.

### 4. Search Flow

```
Query arrives
    |
    v
[1] Embed query via OpenAI (or local model)
[2] FAISS search: top 50 results by cosine similarity
[3] FTS5 search: top 50 results by BM25 rank
    |
    v
[4] RRF fusion: merge both lists by rank position
    |
    v
[5] Return top K results (default 5)
```

**No reranker.** claude-context (our 9/10 baseline) does not use a reranker. Research shows rerankers add +9-17% MRR improvement but cost 400-800ms latency and a ~400MB cross-encoder model. We skip this for now - if hybrid search doesn't match claude-context quality, reranking is the next lever to pull.

**No mode switching on MCP tool.** The `search_code` tool always runs hybrid search internally. A `SEARCH_MODE` env var (`hybrid|semantic|keyword`) is exposed for testing only. The MCP tool signature is unchanged: `search_code(query, path, top_k)`.

### 5. Configuration Summary

| Env Var | Default | Values | Purpose |
|---------|---------|--------|---------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `local` | Embedding model selection |
| `OPENAI_API_KEY` | (required if openai) | API key | OpenAI authentication |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Any OpenAI model | Override OpenAI model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformer | Override local model |
| `SEARCH_MODE` | `hybrid` | `hybrid`, `semantic`, `keyword` | Testing only |
| `FUSION_METHOD` | `rrf` | `rrf`, `weighted` | Fusion algorithm |
| `FUSION_K` | `60` | Integer | RRF smoothing parameter |

### 6. What Does NOT Change

- Chunking (AST-aware multi-language via tree-sitter) - this is the tool's strength
- Merkle-tree incremental indexing
- FAISS index format and storage layout
- MCP tool names and signatures
- Storage directory structure (`~/.claude_code_search/`)
- Test suite (existing tests should pass; new tests added for BM25 and fusion)

## Testing Plan

### Phase 1: Component isolation

Run each component solo against the 10 EXP-008 semantic queries:

| Test | Config | What it proves |
|------|--------|---|
| OpenAI + FAISS only | `SEARCH_MODE=semantic`, `EMBEDDING_PROVIDER=openai` | Does embedding quality alone close the gap? |
| FTS5 BM25 only | `SEARCH_MODE=keyword` | Does FTS5 find the keyword queries (S05, S07)? |
| MiniLM + FAISS only | `SEARCH_MODE=semantic`, `EMBEDDING_PROVIDER=local` | Baseline (same as prior test) |

### Phase 2: Fusion validation

| Test | Config | What it proves |
|------|--------|---|
| OpenAI + FTS5 + RRF | `SEARCH_MODE=hybrid`, `EMBEDDING_PROVIDER=openai` | The full hypothesis |
| MiniLM + FTS5 + RRF | `SEARCH_MODE=hybrid`, `EMBEDDING_PROVIDER=local` | Does hybrid help with local embeddings? |

### Phase 3: Head-to-head

Compare the best configuration against claude-context's recorded results (9/10).

**Scoring:** Human-judged, 3-point scale per result:
- 2 = Good (actual implementation code answering the query)
- 1 = Partial (relevant file but wrong section, or loosely related)
- 0 = Miss (irrelevant, noise, wrong file)

Top-3 results scored per query. Max 6 per query, 60 total.

**Success threshold:** Score Good (2) on top-1 result for at least 8 of 10 queries.

### Decision matrix

| Outcome | Action |
|---------|--------|
| Hybrid + OpenAI >= 8/10 | Deploy. Hypothesis confirmed. |
| Hybrid + OpenAI = 6-7/10 | Investigate tokenization (Tantivy?) or fusion method (convex combination?). |
| Hybrid + OpenAI < 6/10 | Debug: FTS5 tokenization, RRF implementation, index correctness. |
| Pure OpenAI alone >= 8/10 | BM25 unnecessary. Ship embedding swap only, skip BM25. |

## Scale Considerations

Target repo: internal-rust-monorepo (Corsair) - 45K-60K codebase-memory-mcp nodes, estimated 30K-65K AST chunks.

- FAISS: 65K vectors x 1536 dims x 4 bytes = ~400MB index. Fits in memory. IndexIVF available if needed.
- FTS5: 65K rows is trivial for SQLite. Persistent, disk-backed, no startup cost.
- RRF fusion: 100 candidates (50+50), O(n log n) sort. Sub-millisecond.
- OpenAI embedding: 65K chunks x ~500 tokens avg = ~32.5M tokens at index time. At $0.02/1M tokens = ~$0.65 one-time cost. Queries are single embeddings (~$0.00004 each).
