# Sonnet Reranker Default Policy — Decision Record

**Status**: Decision recorded. `RERANKER=sonnet` remains the default.
**Date**: 2026-05-05
**Plan**: Plan-2 Phase C4 (`~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md`).
**Resolves**: 2026-05-05 roundtable Disagreement 1 (`META_SYNTHESIS.md`).

## Question

Should `RERANKER=sonnet` (Sonnet 4.6 query-time reranker) remain the default mode for `search_code`? The roundtable flagged this as empirically underdetermined — the +0.087 MRR was real, but operational risk (silent fallback on rotated API keys) was not measured.

The Plan-2 plan originally framed this as gated on "30 days of telemetry from A1." Per the no-calendar-gating + no-wait-and-measure rules (`~/.claude/rules/feedback_no-*.md`), that framing is wrong. This document resolves the decision in-session using existing measurement + the new A1 observability.

## Falsifier (signal-gated, in-session)

**Keep default-on** if BOTH:
1. Existing measurement on the frozen multi-target holdout (n=183) shows MRR lift ≥ +0.05 vs `RERANKER=off`. ✅ **+0.087** measured (PR #93+).
2. The A1 reranker metadata (`_metadata.reranker.applied`) returns `true` reliably when `ANTHROPIC_API_KEY` is present and produces a discriminating reason when it's not. ✅ **Verified** in test_sonnet_reranker.py (12 reason-path tests + shape contract).

**Flip default-off** if EITHER:
1. MRR lift on holdout < +0.05.
2. A1 metadata cannot discriminate fallback paths (e.g., all failures return same reason).

## Evidence

### Measurement (PR #93+, 2026-05-03)

| Mode | MRR | HR@1 |
|------|-----|------|
| `RERANKER=off` (RRF + boosts only) | 0.763 | (baseline) |
| `RERANKER=sonnet` | 0.850 | (baseline + 0.137) |
| Δ | **+0.087** | **+0.137** |

Source: D4b experiment, n=183 multi-target real_session gold set (now frozen as `multitarget_v1.lock`, sha256 `bd430cae72bef...`). Cited in `CLAUDE.md` Environment Variables section.

### A1 metadata discrimination (PR #104, 2026-05-05)

The `_metadata.reranker = {applied, reason, latency_ms}` envelope surfaces 16 distinct reasons across happy path, fallback paths, and not-invoked paths. Test pinning (`test_metadata_reason_vocabulary_is_stable`) prevents accidental string-vocabulary breaks. Operators / LLM agents can detect:
- `applied: false, reason: api_key_missing` → rotated API key, immediately actionable
- `applied: false, reason: rate_limit` → sustained throttling, back off
- `applied: false, reason: hybrid_prior_fallback` → prompt coverage issue (Sonnet uncertain)
- `applied: false, reason: timeout` → backend unhealthy

Without A1, all of these were indistinguishable from successful rerank — silent degradation.

### Cost vs benefit

- Cost: ~$0.005/query, ~1-2s added latency. At ~100 search_code calls/day on a saturated dev workflow, ~$0.50/day.
- Benefit: +0.087 MRR is the largest single-config win measured against the holdout in 6 months of code-search history.

The cost-benefit clearly favors keeping it on. This decision was also validated by the original PR #93+ ship (rolled out 2026-05-03 with no rollback signal in 2+ days of operator use prior to this session).

## Decision

**Keep `RERANKER=sonnet` as the default.** No code change needed — already the production default per `search/sonnet_reranker.py:13` and `CLAUDE.md` env-var table.

The A1 metadata closes the silent-degradation observability gap. If the metadata starts surfacing sustained `api_key_missing` or `rate_limit` reasons (>1% of calls over a 24-hour window — observable in any operator's tail-f of the sidecar log post-A1), revisit the decision with fresh measurement against the locked holdout.

## How the decision is monitored going forward

Not via 30-day telemetry. Instead:

1. **Anomaly trigger** (signal-gated): if the operator notices `applied: false` rates climbing in their normal usage, run `bench/research/freeze_holdout.py --verify` (sanity-check holdout still locked) and re-run the eval as `RERANKER=sonnet` vs `RERANKER=off`. If the lift has shrunk below +0.05, flip the default.

2. **Pipeline-version trigger**: when `EMBEDDING_MODEL` or grammar versions change (per Plan-2 B3), `pipeline_version` shifts → full reindex on next call. Re-run the holdout eval after that reindex completes; if Sonnet's lift changed materially, revisit.

3. **Cost trigger**: if `$0.005/query × call volume` ever exceeds the operator's threshold, surface the cost in `_metadata` (future enhancement) and let the operator opt out per project.

None of these are calendar-gated. They are signal-gated on observable behavior the A1 metadata + B3 fingerprint already surface.

## Out of scope

- Per-query Sonnet skip heuristics ("don't run Sonnet if the query is short"): potential future work; today the hybrid-prior threshold (PR #95+) covers the most-uncertain cases, and removing the rest would need a query-classifier we don't have.
- Multi-model rerank ensemble (Sonnet + cross-encoder weighted combination): out of scope; cross-encoder regressed quality in 2026-03-22 A/B and was disabled.
