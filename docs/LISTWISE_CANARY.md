# Listwise Reranker — Deployment + p99 Monitoring Runbook

**Status (2026-05-23): DEFAULT-GO.** Listwise reranker is the production default (`RERANKER=listwise`) per Phase C v2 bootstrap CI clearing the eval-shipping-discipline binary ship gate: +0.044 MRR CI [+0.003, +0.084] harvested, +0.047 nDCG@10 CI [+0.004, +0.095] golden, +0.13-0.22 MRR adversarial — all gates favorable, CI excludes zero. Pointwise (`RERANKER=sonnet`) stays selectable as a fallback knob for regression observation; cross-encoder and off paths unchanged. This document is the post-graduation operational runbook covering p99 monitoring and rollback.

> Graduation history: the 2026-05-16 canary phase shipped listwise as opt-in (`RERANKER=listwise`) per [GPT-5.5-pro pass 5 — Phase C v2 ship-readiness](https://github.com/redacted-org/claude-knowledge-base/pull/554). The default flipped to listwise on 2026-05-23 once `eval-shipping-discipline.md`'s binary {SHIP, RETIRE} rule replaced the older canary-then-flip pattern: the Phase C v2 bootstrap CI evidence had already cleared the gate, and opt-in defaults are documented non-decisions.

---

## What changed

Pointwise Sonnet reranker (default `RERANKER=sonnet`) issues 15 isolated per-candidate scoring calls to the Anthropic API per search query, then sorts by the returned scores. Two structural issues drove the listwise hypothesis:

1. **Slowest-of-15 latency** — cohort wall scales with the slowest call. Observed p99 = 24.8s on PSM harvested.
2. **Per-domain inconsistency** — `hybrid_prior_fallback` fires on 75% of golden queries and 95% of harvested queries, meaning pointwise mostly returns hybrid order. When it DOES rerank, it has a known regression on assetman (−0.0695 MRR vs hybrid, CI excludes zero unfavorable).

The listwise reranker (`RERANKER=listwise`) replaces those 15 calls with **one comparative call** that returns ranked IDs + scores in a single response.

---

## Deployment state

Listwise is the default; no env vars required to enable it. The MCP server uses it on a fresh install:

```bash
# Required for any Sonnet rerank path (listwise or pointwise)
export ANTHROPIC_API_KEY=...           # missing → REASON_API_KEY_MISSING, hybrid fallback

# Optional overrides (defaults shown)
export SONNET_LISTWISE_TIMEOUT=12.0    # hard deadline; see "Why 12s default" below

# Then start the MCP server normally
.venv/Scripts/python.exe -m mcp_server.server
```

To explicitly select pointwise (the prior default, kept as fallback for regression observation):

```bash
export RERANKER=sonnet
```

The dispatcher in `search/searcher.py` routes to `listwise_rerank_with_sonnet` when `RERANKER` is `listwise` (the new default) or unset; pointwise (`rerank_with_sonnet`) is reachable via `RERANKER=sonnet`.

---

## Why 12s default

[Phase C v2 simulated-deadline analysis](https://github.com/redacted-org/code-search/pull/181) computed budgeted-quality at deadlines 3s, 4s, 5s, 6s, 8s, 10s, 12s, 15s, 20s — treating queries exceeding the deadline as if they fell back to hybrid (cheap recompute on the v2 listwise replay data, no API spend).

| Deadline | Harvested applied % | Worst Δ MRR vs hybrid | Worst Δ nDCG@10 vs hybrid | All 4 fixtures favorable |
|---:|---:|---:|---:|---:|
| 6s | 66.7% | +0.008 | −0.002 | ✗ |
| 8s | 70.5% | +0.008 | −0.002 | ✗ |
| 10s | 76.5% | +0.008 | +0.004 | ✓ (tightest favorable floor) |
| **12s** | **93.4%** | **+0.008** | **+0.010** | **✓ (default chosen)** |
| 15s | 94.5% | +0.008 | +0.012 | ✓ (diminishing returns) |

10s is the smallest deadline where all fixtures stay favorable. 12s captures 17pp more applied rate and slightly more nDCG@10 lift, at the cost of 2s additional user-visible p99. User chose 12s on 2026-05-16 for the higher applied rate; if SLO tightens, drop to 10s via `SONNET_LISTWISE_TIMEOUT=10.0` — quality stays favorable.

**Anything below 8s degrades quality below hybrid baseline on PSM harvested. Do not go below 8s.**

---

## Fallback behavior

`listwise_rerank_with_sonnet` never raises. On any failure it returns the input candidates in baseline (hybrid) order with a `REASON_*` constant in `_metadata.reranker.reason`:

| Reason | When it fires | What the user sees |
|---|---|---|
| `ok` | Listwise applied successfully | Reranked top-K |
| `empty_input` | 0 candidates | Empty list |
| `api_key_missing` | `ANTHROPIC_API_KEY` not set | Hybrid order |
| `package_not_installed` | `anthropic` SDK missing | Hybrid order |
| `timeout` | Exceeded `SONNET_LISTWISE_TIMEOUT` | Hybrid order |
| `rate_limit` | Anthropic 429 | Hybrid order |
| `parse_failed` | Response not valid JSON even after brace-balanced extraction | Hybrid order |
| `id_mismatch` | Missing / duplicate / unknown candidate IDs in response | Hybrid order |
| `unexpected_error` | Any other exception | Hybrid order |

The contract: **failures degrade quality, never break the search response.**

---

## What to monitor in production

These signals were the canary graduation criteria; they remain the post-ship p99 monitoring contract. A sustained breach is the rollback signal.

### Log signals

Search the MCP server log (`~/.claude/logs/code-search-mcp.log` by default) for these markers:

```bash
# Listwise success rate
grep -c "LISTWISE_REASON.*ok" ~/.claude/logs/code-search-mcp.log

# Fallback breakdown
grep "LISTWISE_REASON" ~/.claude/logs/code-search-mcp.log | \
  awk '{for(i=1;i<=NF;i++) if($i ~ /^[a-z_]+$/) {print $i; break}}' | \
  sort | uniq -c

# Latency distribution
grep "LISTWISE_REASON.*ok.*latency_ms=" ~/.claude/logs/code-search-mcp.log | \
  grep -oE "latency_ms=[0-9]+" | cut -d= -f2 | sort -n
```

### _metadata.reranker per query

Every `search_code` response includes `_metadata.reranker.{applied, reason, latency_ms}`. The MCP layer surfaces this to consumers; LLM agents observing `applied: false` know they got hybrid fallback.

### Production SLO thresholds (originally the canary graduation criteria, now the rollback triggers)

| Criterion | Threshold | Source signal | Breach action |
|---|---|---|---|
| `applied: true` rate ≥ 85% | over rolling 1000 queries | `LISTWISE_REASON.*ok` count / total | Investigate API key / timeout cohort before rollback |
| `parse_failed` rate < 5% | over rolling 1000 queries | grep `parse_failed` | Inspect a sample of failed queries; if persistent, rollback |
| `timeout` rate < 15% | over rolling 1000 queries | grep `LISTWISE_REASON.*timeout` | Tighten `SONNET_LISTWISE_TIMEOUT` (do NOT go below 8s) or rollback |
| User-visible p99 ≤ 13s | wall-clock latency from query start to response | external timing or `_metadata.reranker.latency_ms` | Rollback to `RERANKER=sonnet` if SLO breach is sustained |
| No new user complaints on nix/exact-match queries | qualitative | issue tracker / user reports | Investigate first; rollback if pattern reproduces |

---

## Rollback

To fall back to pointwise (kept as the regression-observation fallback):
```bash
export RERANKER=sonnet
```

To skip reranking entirely and return RRF+boost order:
```bash
export RERANKER=off
```

**No code rollback needed.** Both the pointwise (`search/sonnet_reranker.py`) and listwise (`search/listwise_sonnet_reranker.py`) modules remain in tree; the dispatcher in `search/searcher.py` routes on the `RERANKER` env. Unsetting `RERANKER` returns to the listwise default.

---

## Reference

| Artifact | What it is |
|---|---|
| `search/listwise_sonnet_reranker.py` | The listwise reranker module |
| `search/searcher.py` (line 638+) | Dispatcher branch for `RERANKER=listwise` |
| `tests/unit/test_listwise_reranker.py` | 20 unit tests |
| `tests/unit/test_searcher_listwise_dispatch.py` | 5 dispatcher integration tests |
| `bench/research/listwise_replay.py` | Offline replay harness (3-arm comparison) |
| `bench/research/phase_c_verdict.py` | Verdict-formation script |
| `bench/research/phase_c_bootstrap_ci.py` | Paired bootstrap CI calculator |
| `bench/research/phase_c_simulated_deadlines.py` | Simulated-deadline offline recompute |

### Pull requests

| PR | What it added |
|---|---|
| [#175](https://github.com/redacted-org/code-search/pull/175) | Stemmer bug fix in query expansion (unrelated but found in same session) |
| [#176](https://github.com/redacted-org/code-search/pull/176) | README rerank claim aligned with Sonnet default |
| [#177](https://github.com/redacted-org/code-search/pull/177) | Listwise reranker module + 19 unit tests + replay harness |
| [#178](https://github.com/redacted-org/code-search/pull/178) | Phase C metric panel (nDCG@10, HR@20, pairwise win rate) + verdict-formation + top-20 freezer |
| [#179](https://github.com/redacted-org/code-search/pull/179) | Nix-aware rubric clause + brace-balanced JSON extraction |
| [#180](https://github.com/redacted-org/code-search/pull/180) | Paired bootstrap CI script — all 3 kill-switch gates pass |
| [#181](https://github.com/redacted-org/code-search/pull/181) | RERANKER=listwise dispatcher branch with 10s hard deadline (canary path) |
| [#182](https://github.com/redacted-org/code-search/pull/182) | Default deadline 10s → 12s |

### External reviews (knowledge-base/research/)

Six GPT-5.5-pro consultation passes informed the design. Most important:

- [pass 1 — initial code-stack assessment](https://github.com/redacted-org/claude-knowledge-base/pull/544) — identified pointwise as the deepest architectural issue
- [pass 4 — Phase C verdict consultation](https://github.com/redacted-org/claude-knowledge-base/pull/553) — applied kill-switch gates; v1 fired on nix regression; recommended Option 3 fix
- [pass 5 — Phase C v2 ship-readiness](https://github.com/redacted-org/claude-knowledge-base/pull/554) — verdict CANARY-GO, DEFAULT-NO-GO with 6-step roadmap
- [pass 6 — what's next](https://github.com/redacted-org/claude-knowledge-base/pull/556) — "stopping point; write the runbook" → this document

---

## Open questions / future work

- **Path-override retirement**: `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES` is the containment hack the listwise architecture was meant to retire. Now that listwise is the default, deprecate the env var with a 1-version warning, then remove.
- **Phase B′′ subsumption**: the "per-query routing selective rerank" target from yesterday's ABC roadmap terminal is now subsumed by listwise (which already routes deadline-based). Terminal-doc Phase B′′ after canary stabilizes.
- **Listwise + PPR additive**: the 2026-05-11 Nix arc shipped PPR env-off in `searcher.py:500-524`. A separate measurement could test whether `CODE_SEARCH_PPR_ENABLED=1` + `RERANKER=listwise` is additive (lift on Nix specifically). Cost ~$2 + 30min if pursued.
- **Phase A′′′′ Option α (syn-based Rust extractor)** for code-graph remains queued at 4-6 session arc — separate from the code-search listwise work.
