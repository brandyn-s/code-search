# CLAUDE.md

redacted fork of claude-context-local. Hybrid semantic + keyword code search MCP server.

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

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `voyage` (if `VOYAGE_API_KEY` set) | Provider: `voyage` (recommended, uses voyage-4-large), `voyage-code-3` (available non-default; TypeScript-optimized, regresses on Nix — see `docs/findings/2026-05-15-voyage-code-3-ab-finding.md`), `voyage-context` (legacy contextualized), `openai`, `jina` (local, code-optimized), `local` |
| `JINA_TRUNCATE_DIM` | - | Matryoshka dim truncation for jina provider (0.5b: 64-896, 1.5b: 128-1536) |
| `VOYAGE_API_KEY` | - | Voyage AI API key |
| `CONTENT_MODE` | `code` | `code` or `docs` - affects search weights and provider auto-select |
| `CONTEXTUAL_HEADERS` | `on` | Prepend context headers to embeddings |
| `QUERY_EXPANSION` | `on` | Expand query terms with domain synonyms |
| `RERANKER` | `listwise` | Reranker mode. `listwise` (default, 2026-05-23+): single Sonnet 4.6 listwise call ranks top-15 hybrid candidates in one comparative pass. **Quality status per ship-discipline rule 10: BLOCKED ON MEASUREMENT.** The Phase C v2 bootstrap CI (PR #180, 2026-05-16) showed +0.044 MRR CI [+0.003, +0.084] on harvested, +0.047 nDCG@10 CI [+0.004, +0.095] on golden, +0.13-0.22 MRR on adversarial — but that comparison ran against pre-R9 pointwise. R9 (PR #193, 2026-05-23) added a Nix-aware clause to pointwise's `JUDGE_PROMPT`, closing part of the gap. The current-main listwise-vs-pointwise comparison has not been re-run; the cited lift overstates the present-day delta by some unmeasured amount. Latency justification (1 call vs slowest-of-15) is architectural and not measured on this codebase. See `docs/SHIP_DISCIPLINE.md` for the rule-9/10 framework and `docs/EVAL_RUNBOOK.md` for the re-eval procedure. Hard deadline 12s default (`SONNET_LISTWISE_TIMEOUT`); on deadline/parse-fail/missing `ANTHROPIC_API_KEY`/any error → silently returns hybrid order with a `_metadata.reranker.reason` naming the cause. Cost ~$0.005/query, latency +1-2s. See `docs/LISTWISE_CANARY.md` for the p99 monitoring runbook. `sonnet` (fallback, kept for regression observation): the previous default — Sonnet 4.6 pointwise reranker issuing 15 isolated per-candidate calls. Validated +0.087 MRR / +0.137 HR@1 on n=183 multi-target real_session (D4b, PR #93+) but slowest-of-15 dominates p99 wall and `hybrid_prior_fallback` fires on majority of queries. Selectable via `RERANKER=sonnet`. `cross-encoder`: legacy MiniLM cross-encoder (off-by-default since 2026-03-22 A/B showed quality regression). `off`: skip reranking, return RRF+boost order. |
| `SONNET_LISTWISE_TIMEOUT` | `12.0` | Hard deadline (seconds) for the listwise reranker. Per the 2026-05-16 Phase C v2 simulated-deadline analysis: 10s is the smallest deadline where all 4 production fixtures stay favorable on both MRR and nDCG@10; 12s captures more lift (harvested applied 93.4% vs 76.5%, worst Δ nDCG@10 +0.010 vs +0.004) at the cost of 2s additional p99. Default 12s chosen 2026-05-16 for the higher applied rate. Decrease to 10s for tighter p99 SLO with slightly lower applied rate; 8s drops nDCG@10 below hybrid on harvested — don't go there. Only honored when `RERANKER=listwise`. |
| `ANTHROPIC_CONCURRENCY_LIMIT` | unset (unbounded) | Diagnostic cap on concurrent in-flight Sonnet rerank calls. When unset (default), all 15 candidate-scoring calls fire in parallel via `asyncio.gather`. Set to a positive integer (e.g., `5`) to bound parallelism via `asyncio.Semaphore`. Used by `bench/research/anthropic_latency_diag.py` to test whether observed latency degradation is concurrency-related. Each call also emits a `[ANTHROPIC_DIAG] model=... total_ms=... in_flight=... attempt=... outcome=...` log line at INFO to `~/.claude/logs/code-search-mcp.log`. |
| `ANTHROPIC_MAX_RETRIES` | `1` | Anthropic SDK `max_retries` for the reranker client (Plan D1-Pass-2 B.1). Default lowered from SDK default `2` → `1` after the 2026-05-06 latency diagnosis (PR #133) showed retry-exhausted failures eat ~7.5s of wall time per failure (~3 attempts × ~2.5s with backoff). Cohort-level `FAILURE_TOLERANCE=0.3` already provides graceful fallback; stacking SDK retries wastes wall. Set to `0` to disable SDK retries entirely; set higher for debugging transient-error recovery. |
| `ANTHROPIC_PER_CALL_TIMEOUT_S` | `12.0` | Per-SDK-call timeout in seconds (Plan D1-Pass-2 B.1). Set above the documented p99-successful (8.6s on 2026-05-06 baseline) so genuinely-stuck calls are bounded but healthy slow calls are not truncated. The cohort-level `SONNET_RERANKER_TIMEOUT` (default 8s, applied via `asyncio.wait_for` over `asyncio.gather`) is the OUTER bound — `ANTHROPIC_PER_CALL_TIMEOUT_S` adds a per-call inner cap so one stuck call doesn't dominate the cohort budget. |
| `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD` | `6` | Hybrid-prior fallback threshold (added 2026-05-03, tuned 2026-05-04). When the max Sonnet score across the candidate pool is below this value, the reranker is uncertain — score ties get arbitrary tie-breaking that favors keyword-dense chunks over canonical files. Falls back to hybrid order in that case. Set to `0` to disable. Default tuned 7→6 (PR #96): n=183 multi-target eval showed threshold=6 wins MRR (0.838 vs 0.830 at 7) and HR@1 (0.803 vs 0.787 at 7). |
| `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES` | unset (no overrides) | Per-path-prefix override for the hybrid-prior threshold above. JSON object mapping path-prefix → threshold int, e.g. `'{"assetman/": 11, "mithrandir/": 4}'`. The cohort threshold is the MAX of the base threshold and any override matching a candidate path; mixed cohorts inherit the most restrictive (highest) override. Lowering thresholds via overrides is currently a no-op (cohort never goes below base) — the mechanism is conservative-by-design: tighten only. Bootstrap CI on n=183+102 PSM eval (2026-05-09) showed sonnet rerank effect splits sharply by subproject — assetman MRR delta -0.0695 (CI [-0.16, -0.01], excludes zero), mithrandir +0.1733 (CI [+0.02, +0.33], excludes zero), aggregate golden delta within noise. The override env var lets per-domain tuning ship without changing the global default. Default unset = behavior unchanged. Validate any new override mapping with `bench/research/paired_bootstrap_per_subproject.py` against the prior eval before promoting to production. |
| `SONNET_RERANKER_LOG_PER_CANDIDATE_SCORE` | unset (off) | Phase B'''(a) diagnostic logging (2026-05-14). When truthy (any non-empty value), emits one `[SONNET_PER_CANDIDATE_SCORE] query_hash=... file_path=... score=N pool_size=M` log line per successful Sonnet score call. Used by `bench/research/diagnose_pool_size_score_drift.py` to compare per-candidate scores across pool=15 vs pool=5 runs — tests whether Sonnet's score for the SAME (query, candidate) pair differs based on pool composition. Diagnostic only; no production effect. Default unset = no logging overhead. |
| `SONNET_RERANKER_SKIP_THRESHOLD` | unset (off) | Phase B'''(b) opt-in latency knob (2026-05-14). When set to a positive float, Sonnet rerank is skipped entirely if the top-1 hybrid candidate's `similarity_score >= threshold`; the hybrid top-k is returned in its existing order. Motivation: 2026-05-14 Phase B'' labeling identified ~7% of harvested queries where Sonnet at pool=5 corrupts already-perfect hybrid rank-1 results; skipping Sonnet on high-confidence queries preserves rank-1 + saves ~4-5s latency. Tradeoff: queries where rerank WOULD have helped (low-confidence hybrid) skip the rerank too if their score happens to exceed threshold (rare for low-confidence cohort by definition). Threshold is corpus-specific; tune per-deployment by inspecting `similarity_score` distribution via `bench/research/label_pool_size_safety.py`. Default unset = pre-Phase-B'''(b) behavior. `_metadata.reranker.reason="skipped_high_confidence"` when fired. Not yet validated across all fixtures via bootstrap CI — opt-in only. |
| `SONNET_RERANKER_POOL_SIZE` | `0` (unbounded) | Phase A latency lever (Plan 8-Phase Arc, 2026-05-09). When set to a positive integer N, only the top-N candidates (in hybrid order) are scored by Sonnet; the remaining candidates are appended unchanged in hybrid order at the end of the output. When `0` (default), all input candidates are scored — pre-Phase-A behavior preserved. Latency motivation: cohort wall scales with the slowest of N parallel calls; cutting pool from 15 → 5 reduces parallel-call tail-dominator. Tradeoff: candidates beyond the pool can't move up via rerank — only candidates in the pool participate in score-based reordering. PSM golden eval baseline (2026-05-09): with pool=15, top-1 hybrid is in the rerank-winner position 76% of the time, rerank moves something from rank 2-5 in 22%, rank 6-15 in 2%. Pool=5 keeps the 22% gain while cutting parallel-call count 3x. Default unset = behavior unchanged. Validate any new pool size with `bench/research/anthropic_latency_diag.py` (latency) AND PSM eval bootstrap CI (MRR no-regression) before flipping the default. |

### Phase A — Streaming SDK option (out-of-scope follow-up, 2026-05-09)

The Anthropic Python SDK exposes `client.messages.stream()` and
`client.messages.with_streaming_response`, so streaming responses for the
per-candidate scoring calls is technically possible. **Not pursued for
this phase** because:

- Per-call output is ~50-200 tokens of JSON (`{"score": N, "reasoning": "..."}`).
  Total generation time is small relative to request RTT + first-token
  latency. Streaming reduces time-to-first-token but doesn't reduce
  time-to-last-token meaningfully on short responses.
- The latency dominator measured today is the slowest-of-N-parallel-calls
  pattern, NOT per-call generation time. `SONNET_RERANKER_POOL_SIZE` (this
  PR) directly cuts N; streaming would only marginally reduce per-call wall.
- Streaming would require parsing partial JSON during accumulation — added
  complexity without proportional gain for our use case.

A future phase that pivots from "rerank N candidates fully" to "stream
hybrid order immediately, swap in rerank order as scores arrive" (perceived
latency = hybrid baseline) would benefit from streaming. That's a
materially different UX contract; out of scope here.
| `QUANTIZATION` | `int8` | Index type: `int8` (QT_8bit trained, 4x smaller, default), `float32` (legacy), `binary` (32x smaller, opt-in for 100K+ chunks). **Note**: QT_8bit requires a training step (learns value range). Indexes built before 2026-04-05 used QT_8bit_direct which silently returned 0.0 similarities — must reindex. |
| `VOYAGE_BATCH_API` | `off` | `on` to use Batch API for full reindex (33% cheaper, 1000+ chunk threshold) |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Storage directory |
| `CODE_SEARCH_DISABLE_AUTO_REINDEX` | unset | Set to `1`/`true`/`yes`/`on` to make `auto_reindex_if_needed` a no-op. Useful for large projects (10K+ chunks, 2000+ files) where `detect_changes` is multi-minute. Refresh on demand via `index_directory(incremental=false)` instead. Logs `[REINDEX_PROGRESS] auto_reindex_if_needed: SKIPPED` to `~/.claude/logs/code-search-mcp.log` when active. |

## Search Response Metadata

Every `search_code` response includes a `_metadata` envelope with structured
observability fields covering reranker outcome, index freshness (Phase F2),
and committed-epoch manifest state (Phase E2-6).

```json
{
  "query": "find auth handler",
  "results": [...],
  "_metadata": {
    "reranker": {
      "applied": true,
      "reason": "ok",
      "latency_ms": 1842
    },
    "freshness": "fresh",
    "manifest": {
      "status": "fresh",
      "epoch_id": "2026-05-06T13-42-09-a1b2c3d4"
    }
  }
}
```

**`_metadata.manifest.status`** vocabulary (Plan-2 E2-6, PR #122) — same
strings `verify_index_integrity` reports, sourced from
`search.epoch_manifest.ReadResult.freshness`:

| Status | Meaning |
|--------|---------|
| `fresh` | `manifest/current.json` exists; recorded artifact SHAs match disk |
| `stale_using_prior_epoch` | current.json corrupt; fell back to verified prior.json |
| `missing` | No manifest exists yet (legacy index pre-PR #119) |
| `corrupt` | Both current and prior failed verification (or current missing + prior corrupt) |

`epoch_id` is present when a manifest was successfully read (status `fresh` or
`stale_using_prior_epoch`). The `manifest` field is absent entirely if the
manifest probe itself raised — search responses must never break on
observability path failures.

**Reasons** (stable string vocabulary):

| Reason | applied | Meaning |
|--------|---------|---------|
| `ok` | true | Sonnet rerank applied successfully |
| `empty_input` | false | No candidates passed to reranker (rare) |
| `api_key_missing` | false | `ANTHROPIC_API_KEY` not set — fell back to hybrid |
| `package_not_installed` | false | `anthropic` SDK not installed |
| `timeout` | false | Total timeout exceeded OR per-call timeouts dominated |
| `rate_limit` | false | Anthropic rate-limit responses dominated — back off |
| `too_many_failures` | false | >30% per-call failures (HTTP, parse) |
| `hybrid_prior_fallback` | false | Max Sonnet score below threshold; hybrid order used |
| `skipped_high_confidence` | false | Top-1 hybrid score met `SONNET_RERANKER_SKIP_THRESHOLD`; Sonnet not invoked |
| `disabled_by_env` | false | `RERANKER=off` |
| `not_invoked_keyword_mode` | false | `search_mode=keyword` skipped reranking |
| `not_invoked_semantic_mode` | false | `search_mode=semantic` skipped reranking |
| `not_invoked_cross_encoder_mode` | false | `RERANKER=cross-encoder` legacy path |
| `not_invoked_no_candidates` | false | Index returned 0 candidates |
| `not_invoked_insufficient_candidates` | false | Fewer candidates than `k`; rerank skipped |
| `async_context` | false | Reranker called from async context (unsupported) |
| `unexpected_error` | false | Catch-all (logged) |

**Why this matters**: a sustained `applied: false` rate on production traffic
is the canary for a rotated `ANTHROPIC_API_KEY`, sustained rate-limiting, or
prompt-coverage issues (high `hybrid_prior_fallback` rate). Without this
metadata, silent fallback was indistinguishable from successful rerank — see
roundtable Disagreement 1 (META_SYNTHESIS, 2026-05-05).

## Voyage AI Integration

code-search uses [Voyage AI](https://voyageai.com) embedding models to convert code chunks into vectors for semantic similarity search. When you search "where is the firewall config?", your query and every indexed chunk are compared as vectors — chunks whose vectors point in similar directions are returned as results.

**Model**: `voyage-4-large` (MoE architecture, SOTA retrieval). Uses the standard `/v1/embeddings` endpoint. Eval (n=102 queries, 4 languages) showed +0.053 weighted avg MRR over `voyage-context-3`. Wins on Nix (+0.034), Rust service (+0.134), TypeScript (+0.021), ties on Rust lib. The `voyage-context` provider (contextualized embeddings via `/v1/contextualizedembeddings`) is preserved as legacy.

**Search pipeline**:
1. **Indexing**: Tree-sitter parses code into AST chunks → contextual headers prepended → sent to Voyage `/v1/embeddings` → vectors stored in FAISS (int8 quantized, 4x smaller)
2. **Search**: Query embedded via Voyage → FAISS cosine similarity → BM25 keyword search → Weighted RRF fusion (50/50) → chunk-type boosts → results

**Optimizations**:
- **int8 quantization** (default): QT_8bit (trained), 4x smaller FAISS indexes. Must use `QT_8bit`, NOT `QT_8bit_direct` (which silently returns 0.0 similarities on normalized vectors — discovered 2026-04-05).
- **Binary + rescore** (opt-in `QUANTIZATION=binary`): 32x smaller, hamming search → float rescore top-k. For 100K+ chunk repos.
- **Token pre-count**: Batches split by estimated token budget before API calls, preventing 400 errors
- **Batch API** (opt-in `VOYAGE_BATCH_API=on`): 33% cheaper async embedding for full reindexes (1000+ chunks)

**Multi-language eval results** (n=102, 4 language sub-projects, MRR):

| Provider | Model | Nix (n=44) | Rust svc (n=20) | Rust lib (n=18) | TypeScript (n=20) | Avg |
|----------|-------|-----------|----------------|----------------|------------------|-----|
| `voyage` | **voyage-4-large** | **0.826** | **0.917** | 0.861 | **0.683** | **0.828** |
| `voyage-context` | voyage-context-3 | 0.792 | 0.783 | **0.861** | 0.662 | 0.775 |
| `voyage` | voyage-4 | 0.803 | 0.892 | 0.861 | 0.650 | 0.806 |
| `voyage` | voyage-4-lite | 0.785 | 0.892 | 0.880 | 0.650 | 0.798 |
| `jina` (enriched) | jina-code-0.5b | 0.638 | 0.742 | ~0.86 | 0.660 | — |

Key: voyage-4-large wins 3 of 4 languages, +0.053 weighted avg MRR over voyage-context-3. Uses standard `/embeddings` endpoint (simpler than contextualized). Reranking (rerank-2.5) degrades quality (-30% MRR) — disabled.

## Protected Repo

PR required to merge to main. Use `--repo redacted-org/code-search` with `gh` CLI (bare `gh` resolves to upstream).
