# V&V session 2 — corruption robustness, lifecycle state machine, stress passes

**Date**: 2026-06-10
**Scope**: second container V&V pass: torn-write corruption fuzz (bug class
found + fixed), stateful lifecycle property testing, fd-leak and
pathological-input stress (clean).
**Verdicts**: corruption hardening **DONE** (26-case contract pinned);
lifecycle/stress passes are validation records.

## Found + fixed: 13/24 corruption shapes crashed the read path

Fuzz: build a healthy index, corrupt each artifact (truncate-half /
truncate-zero / garbage / delete), probe constructor + `search` +
`search_bm25` + `get_stats` + `stale_ratio` on a fresh manager.

Pre-fix: **13 of 24 shapes raised unhandled exceptions**, including:
- corrupt `fts5.db` → `sqlite3.DatabaseError` **in the constructor** —
  `CodeIndexManager` unconstructable until manual cleanup;
- corrupt `chunk_ids.pkl` → `UnpicklingError` from every search, even
  though lossless rebuild from metadata existed one call away
  (`_maybe_rebuild_chunk_ids`);
- corrupt `code.index` → raw faiss `RuntimeError` from every search;
- corrupt `stats.json` → `JSONDecodeError` from `get_stats`;
- corrupt `metadata.db` → raw sqlitedict traceback.
(The epoch-manifest layer was fully graceful in all modes — as designed.)

Contract now (pinned by `tests/unit/test_corruption_robustness.py`, 26 cases):
- **Derived artifacts degrade loudly, never crash**: corrupt `fts5.db` is
  quarantined (`fts5.db.corrupt.<ts>`) and recreated empty (BM25 leg empty
  until reindex); corrupt `code.index` disables the vector leg (BM25 keeps
  serving); corrupt `chunk_ids.pkl` rebuilds **losslessly** from metadata
  (search still returns results — asserted, not just no-crash); corrupt
  `stats.json` returns defaults.
- **`metadata.db` (not rebuildable) raises an actionable `RuntimeError`**
  naming the remedy (`index_directory(incremental=false)`), instead of a
  deep sqlitedict traceback. Post-fix matrix: 22/24 graceful + 2/24
  actionable-by-design.

## Stateful lifecycle property test: PASSED

Hypothesis `RuleBasedStateMachine` driving random add/modify/delete/
incremental/full sequences (25 sequences × ≤18 steps) against invariants:
live-file tokens always retrievable (recall), deleted files never returned
(staleness), superseded tokens never hit their file (modify), `ntotal ≥
live rows`, and a fresh manager over the same storage agrees (reopen). All
held — the PR #224 churn-hygiene + P5 compaction behavior is stable under
random operation orderings. Kept as a session script (multi-second
runtime); promote to `tests/integration/` if lifecycle code churns again.

## Stress passes (clean, no action)

- **FD leak**: 100 open/use/close manager cycles AND 50 `__del__`-only
  cycles → fd delta **0** (the 2026-05-30 `__del__` deadlock fix holds and
  releases handles).
- **Pathological chunker inputs**: 10MB single-line file (0.1s), 50k tiny
  functions (755 chunks, 2.0s), 200-deep nesting, NUL bytes, BOM+CRLF,
  latin-1 bytes (zero-chunk with structured `[CHUNKING_DIAG_FILE]` log —
  the documented contract), CJK/emoji identifiers — no hangs, no crashes,
  no slow paths.

## Not run (recorded for completeness)

Mutation testing (test-strength scoring) — feasible but hours of wall
time; run scoped to `search/indexer.py` + `chunking/chunk_merging.py`
locally if desired. Kill-9 mid-write injection — the torn-write matrix
covers the resulting on-disk states; live signal-injection adds little
over it for SQLite/FAISS single-file artifacts.
