# P1 pre-work — chunk-budget mechanical sweep: BLOCKED ON MEASUREMENT (quality)

**Date**: 2026-06-10
**Plan**: `docs/plans/2026-06-10-retrieval-improvement-roadmap.md` P1
**Verdict**: **BLOCKED ON MEASUREMENT** — this note records the *mechanical*
(no-embeddings) half only. The quality verdict requires the full P1 protocol
(per-arm reindex + PSM eval + paired bootstrap) with API keys; nothing here
authorizes changing `MAX_CHUNK_NWS`.

## Distribution sweep (container, 2026-06-10)

Corpus: code-search's own `search/ embeddings/ chunking/ mcp_server/ merkle/
scripts/` (61 Python files). Pure tree-sitter chunking + merge at three
budgets; NWS = non-whitespace chars per chunk.

| budget | chunks | mean | median | p90 | max | <400 NWS | cross-merged |
|---|---|---|---|---|---|---|---|
| 1500 (current) | 329 | 2127 | 1269 | 3978 | 66331 | 43 (13%) | 104 (31%) |
| 2000 | 282 (−14%) | 2471 | 1637 | 4085 | 66331 | 28 (9%) | 106 (37%) |
| 2500 | 248 (−25%) | 2783 | 2002 | 4519 | 66331 | 22 (8%) | 110 (44%) |

## What this does and does not say

- Raising the budget to 2000/2500 cuts chunk count 14%/25% and roughly
  halves the sub-400-NWS tail (the size band cAST's cited Ekimetrics figure
  flags as degrading retrieval) — the *mechanical* preconditions for the
  cAST-predicted gain are present in our corpus shape.
- Cross-boundary merges (`multi_chunk_merge` tag) rise 31%→44%. The R10
  eval (2026-05-23) found deboosting such chunks does NOT help, so this is
  not presumed harmful — but it is the main thing to watch per-subproject
  in the quality eval.
- **Index-cost note**: fewer chunks = fewer vectors (−14%/−25% index size)
  but larger per-result payloads (median result roughly +29%/+58% NWS).
  Token cost per formatted response should be reported alongside MRR in the
  P1 eval, as the plan specifies.
- Outlier: max chunk is 66,331 NWS at every budget — a single oversized AST
  node is never split by the merge pass (by design). Unaffected by P1;
  noted for completeness.

## Sweep mechanics gotcha (for the local run)

`merge_file_chunks` binds `max_nws=MAX_CHUNK_NWS` as a **default parameter at
def time** — monkeypatching the module constant at runtime does NOT change
the chunker's behavior. Edit the constant in source per arm (as the plan
instructs) or wrap `chunk_merging.merge_file_chunks`; the constant is
re-bound on import, so source edits work.
