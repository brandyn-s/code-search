# P4 — Content-hash document-embedding cache: DONE (mechanism measured)

**Date**: 2026-06-10
**Plan**: `docs/plans/2026-06-10-retrieval-improvement-roadmap.md` P4
**Verdict**: **DONE** — mechanism implemented, unit-pinned (10 tests), and
measured end-to-end with a real local model. Retrieval quality is untouched
by construction (cache hits return byte-identical vectors; pinned by test).

## What shipped

`CodeEmbedder._embed_documents_cached` fronts the flat document-encode paths
(`embed_chunks` + `embed_chunk`) with a SQLite cache keyed by
**(sha256(content), provider, model, input_mode)** — the PR #224 query-cache
keying discipline. Grouped/contextualized providers (voyage-context) bypass
the cache entirely: their vectors are document-context-dependent. Misses are
deduped within a batch; rows are capped at 200K with oldest-first eviction;
cache failures degrade to a plain encode. Side fix: the single-chunk path now
honors `VOYAGE_INPUT_TYPE` like the batch path (it previously didn't — the
same content could embed differently depending on which path indexed it).

## Measurement (container, 2026-06-10)

Corpus: code-search's own `search/ embeddings/ chunking/ mcp_server/ merkle/`
(56 files → 309 chunks), `EMBEDDING_PROVIDER=local`
(all-MiniLM-L6-v2, CPU). Encode counts via a counting wrapper on
`model.encode`; each run uses a fresh embedder + index-manager instance.

| Run | Wall | Chunks | Texts encoded |
|---|---|---|---|
| Full index, cold | 12.6 s | 309 | 309 |
| Full index, repeat (cache warm) | **0.6 s** | 309 | **0** |
| Incremental after touching 5% of files | 0.4 s | 4 re-added | **2** |

- No-change full reindex: **100% hit rate, ~21× wall-time reduction** (the
  remaining 0.6 s is chunking + FAISS/FTS writes).
- 5%-churn incremental: only the 2 genuinely-novel chunk texts hit the model;
  the other re-added chunks were content-identical and served from cache.
- For API providers the texts-encoded column is the dollar column: a
  no-change full reindex goes from full-corpus token spend to ~zero.

**Caveat (rule 10 scope)**: wall-time numbers above are local-CPU-model
numbers on a 309-chunk corpus. The PSM-scale Voyage-API cost table (the
plan's optional follow-up) would be strictly more favorable — API latency
per skipped call exceeds the local model's — but it has not been measured;
treat per-deployment dollar savings as unmeasured until run locally.

## Tests

`tests/unit/test_doc_embedding_cache.py` — 10 cases: second-run zero-encode,
partial-churn novel-only encode, provider/model isolation, input_mode keying,
intra-batch dedupe, single-path/batch-path cache sharing, grouped-provider
bypass, clear, cache-unavailable degradation, cached-vs-fresh vector
identity.
