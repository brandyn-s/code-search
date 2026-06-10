# P5 — Stale-vector auto-compaction: DONE

**Date**: 2026-06-10
**Plan**: `docs/plans/2026-06-10-retrieval-improvement-roadmap.md` P5
**Verdict**: **DONE** — implemented, unit-pinned (9 tests incl. the full
churn→escalate→compact scenario), default-on with hard-coded thresholds.

## What shipped

FAISS rows are never removed in place, so modify/delete churn accumulates
stale vectors (the bloat half of the PR #224 hygiene class). PR #224 added
the `live_chunks`/`stale_vectors` stats; P5 acts on them:

- `CodeIndexManager.stale_ratio()` — live measurement
  (`(ntotal − metadata rows) / live`), `None` when empty/unknown.
- **Thresholds** (hard-coded, no knobs): `STALE_ADVISORY_RATIO = 0.25`,
  `STALE_COMPACTION_RATIO = 0.5`.
- `IncrementalIndexer.incremental_index` escalates to a full reindex when
  the pre-run ratio exceeds 0.5 — at that point the index holds more
  garbage than live data and a rebuild is strictly better. Self-limiting:
  the full reindex resets the ratio to 0. Logged as
  `[REINDEX_PROGRESS] compaction: …` in the sidecar.
- `search_code` `_metadata.stale_index` advisory above 0.25 — a separate
  additive object (ratio, counts, recommendation), **deliberately not a new
  `freshness` string**: `freshness` tracks index-vs-source-tree state
  through the auto-reindex flow and overloading its vocabulary would
  clobber that signal. (Deviation from the plan's literal wording, same
  intent.) Absent below threshold; probe failures never break a search.
- `verify_index_integrity` per-project `stale_vectors`/`stale_ratio` (from
  stats.json; legacy stats without the keys are skipped), summary
  `total_stale_vectors` + `projects_needing_compaction`, and a remediation
  line.

## Escalation cost note

The escalation's cost is one full reindex. In-container reference point
(local model, 309-chunk corpus): 12.6 s cold / 0.6 s with the P4 embedding
cache warm — i.e., **with P4 merged, compaction is nearly free** for
unchanged content, since the rebuild's embedding step is served from cache.
PSM-scale wall time should be recorded locally on first real escalation
(grep the sidecar for `[REINDEX_PROGRESS] compaction:`).

## Tests

`tests/unit/test_stale_compaction.py` — ratio unit tests (empty→None,
healthy→0.0, re-add churn→1.0, removed-file→1.0), below-threshold runs stay
incremental, churn-past-threshold no-change run escalates and resets ratio
to 0, force_full unaffected. Plus two `verify_index_integrity` cases
(stale fields surface + legacy stats skipped) in
`tests/unit/test_verify_index_integrity.py`.

One behavioral note from testing: a single modify-all-files cycle already
produces ratio ≈ 1.0 (> 0.5) on a 2-file project — small projects will
compact aggressively. That is acceptable (their full reindex is cheap and
P4-cached); large projects need proportionally more churn to cross 0.5.
