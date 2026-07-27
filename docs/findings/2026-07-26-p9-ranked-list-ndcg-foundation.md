# P9 ranked-list and nDCG foundation

**Date:** 2026-07-26

**Roadmap item:** P9 — Public benchmark anchor

**Outcome:** HARNESS READY; QUALITY OUTCOME BLOCKED ON MEASUREMENT

## Decision

Advance the bounded, corpus-independent part of P9 without presenting harness
work as a benchmark result. The evaluation worker now emits stable,
query-ID-addressable document rankings and can score them directly against
graded qrels. The existing manual CoIR workflow persists those results as CI
artifacts.

This is useful infrastructure for the planned Loc-Bench file-level harness,
but it does **not** satisfy P9's ship gate. No Loc-Bench reachable-instance lock
or current-main baseline was produced in this change.

## Implemented evidence

- `benchmarks/_eval_worker.py` accepts `--k`, `--qrels`, `--output`, and
  `--unique-documents`.
- Chunk results are stably deduplicated to first-hit document order. The worker
  retrieves four times the requested document cutoff before deduplication so
  repeated chunks do not trivially truncate the ranked document list. It also
  records the actual document count and underfill state.
- `bench/research/coir_adapter.py` preserves dataset query IDs in the golden
  records and keys graded qrels by those IDs. Duplicate query text therefore
  cannot overwrite a different query's judgments. It downloads the complete
  candidate corpus and complete qrels split, checks reported row totals, and
  only then selects the bounded query subset.
- `bench/research/ndcg.py` reports per-query and aggregate graded nDCG@k and
  Recall@k. It fails closed for zero-query measurements, absent or non-positive
  qrels, duplicate ranked document IDs, and underfilled top-k rankings.
- `.github/workflows/external-benchmarks.yml` requests document-level `k=10`,
  scores both configured Voyage models, and uploads complete result JSON.
- Unit contracts exercise graded scoring, duplicate query text, complete-corpus
  and qrels acquisition, dataset row-count failures, document deduplication,
  ranked-list underfill, CLI discoverability, and workflow wiring.

These checks validate the mechanics with synthetic fixtures. They do not
measure retrieval quality.

## Measurement still required

1. Run the manual `external-benchmarks` workflow with `VOYAGE_API_KEY`
   configured and retain the `coir-metrics` artifact.
2. Compare `external_metrics.ndcg@10` and `external_metrics.recall@10` for
   `voyage-4-large` and `voyage-code-3`; record the run URL and exact dataset
   input.
3. For P9 proper, port the Loc-Bench file-level adapter, freeze a reachable
   instance lock, and run the bounded 50-instance current-main baseline.

Until those runs exist, provider comparisons and the P9 baseline remain
**BLOCKED ON MEASUREMENT**. No default, routing, or quality claim changes here.
