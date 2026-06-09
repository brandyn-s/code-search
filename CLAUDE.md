# CLAUDE.md

redacted fork of claude-context-local. Hybrid semantic + keyword code search MCP server.

> **Reference split**: this file is the always-loaded core. The full
> environment-variable table (~30 vars incl. every diagnostic/experimental
> reranker knob), the reranker `reason` vocabulary, the streaming-SDK
> follow-up, and the multi-language embedding eval tables live in
> [`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md). Keep that file's measured
> records intact (ship-discipline) — relocate findings here only as one-liners.

## Ship-Discipline Policy

Before claiming any change "improves code-search" (quality, latency, or
any measurable outcome), apply rules 9 and 10 from `docs/SHIP_DISCIPLINE.md`:

- **Rule 9 (evidence staleness)**: cited eval results must reflect current
  state. Run `git log --since=<eval date> -- <comparison-arms>` before
  citing measurements; if upstream changes touch the comparison, re-run
  or explicitly justify why the old result still applies.
- **Rule 10 (affirmative outcome)**: outcome claims need measurement
  under current conditions. The closing of a goal / PR must explicitly
  use one of: **DONE** (measured + affirmed), **DECIDE** (measured +
  refuted), or **BLOCKED ON MEASUREMENT** (no current-state measurement).
  Defensibility narratives, architectural reasoning, and absence-of-
  contradicting-evidence do not satisfy rule 10.

When in doubt, the correct closing is BLOCKED, not a softened DONE.

See also: `docs/EVAL_RUNBOOK.md` (how to run paired-bootstrap CI).

## Key Commands

```bash
# Run tests
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Run MCP server
.venv/Scripts/python.exe -m mcp_server.server

# Index a repo (from MCP client)
# mcp__code-search__index_directory(directory_path="...")
```

## Architecture

- **Embedding providers**: Voyage AI (`voyage-4-large` default — +0.053 weighted avg MRR over `voyage-context-3` across 4 langs), `voyage-code-3` (available, non-default — wins on TypeScript, regresses on Nix vs voyage-4-large per 2026-05-15 A/B; see `docs/findings/`), `voyage-context-3` legacy, OpenAI, local sentence-transformers
- **Search**: Weighted RRF fusion of FAISS vector + FTS5 BM25. Content mode boosts (code: function/method 1.3x, docs: section 1.3x)
- **Chunking**: Tree-sitter AST for 12+ languages, regex-based for TOML/YAML/HCL/Markdown/Nix. Post-processing merge step (cAST-style) greedily combines small adjacent chunks to 1500 NWS char budget, capturing gap code (imports, constants) between semantic units.
- **Per-project config**: `project_info.json` stores embedding provider, model, content mode. Server creates correct embedder on project switch.
- **Contextual headers**: `# From <path> - <type> <name>` prepended before embedding (controlled by `CONTEXTUAL_HEADERS=on`)

## Testing

```bash
# All unit tests (34+)
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Specific test files
.venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py -v
.venv/Scripts/python.exe -m pytest tests/unit/test_voyage_context_embedder.py -v
.venv/Scripts/python.exe -m pytest tests/unit/test_hybrid_search.py -v
```

## Environment Variables (production defaults)

Only the variables you set in normal operation. The diagnostic and experimental
knobs — reranker pool size, skip / hybrid-prior thresholds, per-call timeouts,
per-cohort prompt/threshold overrides, latency-diag logging, `LLM_CONTEXT_PATH`,
`JINA_TRUNCATE_DIM` — are in [`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md).

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `voyage` (if `VOYAGE_API_KEY` set) | `voyage` (voyage-4-large, recommended), `voyage-code-3` (TS-optimized, regresses on Nix), `voyage-context` (legacy), `openai`, `jina` (local), `local` |
| `VOYAGE_API_KEY` | – | Voyage AI API key |
| `CONTENT_MODE` | `code` | `code` or `docs` — affects search weights and provider auto-select |
| `CONTEXTUAL_HEADERS` | `on` | Prepend `# From <path>` context headers before embedding |
| `QUERY_EXPANSION` | `on` | Expand query terms with domain synonyms |
| `RERANKER` | `sonnet` | `sonnet` (pointwise, default), `listwise`, `cross-encoder` (legacy), `off`. Graceful fallback to hybrid order on any error (reason recorded in `_metadata`). Full knob set + eval history in ENV_REFERENCE. |
| `QUANTIZATION` | `int8` | `int8` (QT_8bit trained, 4x smaller), `float32` (legacy), `binary` (32x smaller, 100K+ chunks). **Gotcha**: must be QT_8bit, NOT QT_8bit_direct (silently returns 0.0 sims — reindex anything built pre-2026-04-05). |
| `VOYAGE_BATCH_API` | `off` | `on` for 33% cheaper async embedding on full reindex (1000+ chunks) |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Storage directory |
| `CODE_SEARCH_DISABLE_AUTO_REINDEX` | unset | `1`/`true`/`yes`/`on` makes `auto_reindex_if_needed` a no-op (large projects); refresh on demand via `index_directory(incremental=false)` |

## Search Response Metadata

Every `search_code` response includes a `_metadata` envelope with structured
observability fields (reranker outcome, index freshness, committed-epoch
manifest state):

```json
{
  "query": "find auth handler",
  "results": [...],
  "_metadata": {
    "reranker": { "applied": true, "reason": "ok", "latency_ms": 1842 },
    "freshness": "fresh",
    "manifest": { "status": "fresh", "epoch_id": "2026-05-06T13-42-09-a1b2c3d4" }
  }
}
```

- **`reranker.reason`** is a stable string vocabulary (`ok`, `api_key_missing`,
  `timeout`, `rate_limit`, `hybrid_prior_fallback`, `skipped_high_confidence`,
  `disabled_by_env`, …) — full table in
  [`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md). A sustained `applied: false`
  rate on production traffic is the canary for a rotated `ANTHROPIC_API_KEY`,
  sustained rate-limiting, or prompt-coverage gaps (high `hybrid_prior_fallback`).
  Without this metadata, silent fallback was indistinguishable from successful
  rerank.
- **`manifest.status`** (Plan-2 E2-6, PR #122) — same strings
  `verify_index_integrity` reports, sourced from
  `search.epoch_manifest.ReadResult.freshness`:

| Status | Meaning |
|--------|---------|
| `fresh` | `manifest/current.json` exists; recorded artifact SHAs match disk |
| `stale_using_prior_epoch` | current.json corrupt; fell back to verified prior.json |
| `missing` | No manifest exists yet (legacy index pre-PR #119) |
| `corrupt` | Both current and prior failed verification |

`epoch_id` is present when a manifest read succeeded. The `manifest` field is
absent entirely if the probe itself raised — search responses must never break
on observability-path failures.

## Embeddings

`voyage-4-large` (MoE, standard `/v1/embeddings`) is the default — +0.053
weighted avg MRR over `voyage-context-3` across 4 languages (per-language eval
table and full provider comparison in
[`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md)). Pipeline: tree-sitter AST
chunks → contextual headers → Voyage embeddings → FAISS (int8) + BM25 →
weighted RRF (50/50) → chunk-type boosts → optional Sonnet rerank. Voyage
rerank-2.5 degrades quality (−30% MRR) and is disabled; the reranking layer is
Sonnet (see `RERANKER`).

## Protected Repo

PR required to merge to main. The repo is `redacted-org/code-search`
(transferred from `redacted-org` in the 2026-04-26 split) — pass
`--repo redacted-org/code-search` to `gh` so a bare `gh` doesn't
resolve to upstream.
