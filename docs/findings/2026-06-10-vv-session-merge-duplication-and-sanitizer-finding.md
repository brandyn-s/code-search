# V&V session — chunk-merge duplication bugs, FTS5 NUL bug, e2e battery

**Date**: 2026-06-10
**Scope**: container-side testing/verification beyond the roadmap phases:
property/fuzz testing of parsing+fusion primitives, end-to-end local-model
battery, concurrency smoke, static-analysis triage.
**Verdicts**: two bug fixes **DONE** (pinned by tests); battery/parity/
concurrency results are smoke-level validation — **not** retrieval-quality
claims (those remain on the PSM harness per EVAL_RUNBOOK).

## Bug 1 (fixed): overlapping chunk emissions duplicated content at index time

Production chunkers emit nested/overlapping ranges by design (a class chunk
plus its method chunks — multi-granularity). Two `merge_file_chunks` defects
turned that into *extra* duplication beyond the intended overlap:

1. **Coverage rewind** — a nested chunk reset `last_end_line` below its
   parent's end; the trailing-gap logic then re-emitted already-covered
   parent lines as phantom `module_level` chunks.
2. **Contentful-hole spanning** — merged-group content is rebuilt as one
   contiguous source span; a group of method segments whose hole was the
   parent's exclusive region (e.g., a trailing class attr after the last
   method) silently absorbed those lines into a second chunk. Repro with the
   real Python chunker: a class trailing attribute was indexed **twice**
   (`docs`-shape: `class C: … methods … ATTR = x` + module trailer).
3. (Adjacent) **group end used `group[-1].end_line`** — wrong when an
   overlapping parent sorts before a shorter nested segment; now
   `max(seg.end_line)`.

**Fixes** in `chunking/chunk_merging.py`: monotonic coverage
(`last_end_line = max(...)`), group break on holes containing non-whitespace
content, max-based group end. Deliberately NOT touched: the chunkers'
overlapping emission itself — multi-granularity retrieval is plausibly
intentional and changing it is a measured retrieval change (rule 10).

**Impact**: phantom/duplicated chunks inflated FAISS+BM25 with near-duplicate
content (and wasted embedding spend). On this repo's corpus the footprint was
small (≤1 chunk per budget arm in the P1 sweep, table refreshed in the P1
finding); class-heavy corpora (trailing class attrs after methods) are the
larger exposure. Existing indexes shed the duplicates on their next full
reindex — no migration needed.

**Tests**: `tests/unit/test_chunk_merge_overlap.py` — deterministic repros
for both mechanisms + a 200-example Hypothesis property:
*merge never increases any line's multiplicity beyond the input's, and never
loses a non-blank line* (`hypothesis` added to requirements-dev.txt;
property tests skip if it's absent).

## Bug 2 (fixed): NUL bytes broke the BM25 leg silently

`_sanitize_fts5_query` passed `\x00` through into quoted FTS5 tokens; SQLite
raises "unterminated string", `search_bm25` swallows the exception and
returns `[]` — the BM25 leg silently vanished for any query containing a NUL
(e.g., binary-pasted content through the MCP arg). Found by a 20K-case fuzz
against a real FTS5 table; NUL/C0 controls were the only failing class
(exotic unicode, zero-width chars, operator soup all survived). Fix: C0
controls added to the sanitizer's strip class. Tests:
`tests/unit/test_fts5_sanitizer_fuzz.py` (deterministic NUL/control/operator
cases + 500-example Hypothesis sweep asserting MATCH never raises).

## E2E local-model battery (smoke-level, container)

Corpus: this repo's own core modules (308 chunks), `EMBEDDING_PROVIDER=local`
(MiniLM), `RERANKER=off`, three QUANTIZATION arms:

- **Known-item file hit@5: 7/8 on all three arms** (int8/float32/binary);
  the single "miss" was a mislabeled expectation — the engine returned the
  semantically correct file (`merkle/change_detector.py`).
- **Quantization parity**: top-10 chunk_id Jaccard vs float32 = **1.000**
  for int8 AND binary (n=8 queries) — the binary hamming→float-rescore path
  is doing its job; a large gap here is the QT_8bit_direct regression
  signature. Caveat: 384-dim model, 308-chunk corpus — parity at PSM scale
  is expected lower for binary per the quantization literature.
- **Binary-mode `get_similar_chunks`** exercised e2e against a real binary
  index (the PR #224 float-store fix) — no crash, sane neighbors.
- **Concurrency smoke**: 4 search threads × 25 queries racing 3 incremental
  reindex cycles (including P5 compaction escalations): **0 errors**. Not
  committed as a unit test (thread-race flake risk); rerun ad-hoc when
  touching locking paths.
- Durable subset committed as
  `tests/integration/test_local_e2e_battery.py` (importorskip on
  sentence-transformers; the only no-API-keys full-stack retrieval test in
  the repo).

## Static-analysis triage (ruff)

Serious classes (E9/F821/F811/F632/F841) in production code: **zero**.
The repo-wide 252 findings are unused imports/f-string nits, including three
FALSE positives in `search/searcher.py` (deliberate backwards-compat
re-exports). The only F821s live in
`tests/integration/test_full_flow.py::test_project_manager_operations` —
dead code behind `@pytest.mark.skip` + early `return` for a
never-implemented ProjectManager; left as-is. A lint config is a separate
decision, not taken here.

## Suite state

Full unit suite after this session's changes: see PR — all green in the
container including the new 11 tests (5 merge-overlap incl. property, 5
sanitizer-fuzz, 1 e2e battery).
