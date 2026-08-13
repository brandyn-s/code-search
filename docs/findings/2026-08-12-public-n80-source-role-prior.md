# Public n=80 source-role ranking finding

## Decision

Adopt a bounded source-role prior in `CONTENT_MODE=code`. Test and
documentation results receive a 0.82 multiplier unless the first line of the
query explicitly asks for that artifact type. Candidate files are diversified
before final truncation.

## Diagnosis

The frozen public LocBench n=80 artifact contained 30 rank-1 cases, 33 cases
with an expected file at ranks 2-10, and 17 misses. Among the 33 near misses,
10 ranked a test file first and two ranked documentation first even though 31
of the 33 oracles named implementation files. Five representative cases were
read and replayed before changing the ranking policy. They showed the same
mechanism: supporting artifacts repeated the issue's symbols and prose and
outranked the implementation.

## Current-main versus candidate replay

Both arms queried the same 80 frozen repositories, revisions, existing index
generations, queries, and oracle. The only arm difference was `PYTHONPATH`:
current `main` at `f5888855e7ad368ab56987f905706f8e2e3aa210` versus this candidate.
No model or re-indexing was used; `RERANKER=off` and `auto_reindex=false`.

| Metric | Current main | Candidate | Delta |
|---|---:|---:|---:|
| Acc@1 | 0.3625 | 0.3875 | +0.0250 |
| Acc@3 | 0.6125 | 0.6250 | +0.0125 |
| Acc@10 | 0.7625 | 0.7750 | +0.0125 |
| MRR@10 | 0.49147 | 0.51608 | +0.02460 |

Thirteen cases improved and none regressed. This is a modest, measured
improvement, not a superiority claim. The earlier published n=80 evidence was
produced by plugin 0.4.23; the side-by-side replay above uses the current 0.4.27
code-search baseline so release drift is not credited to this change.

## Boundary

This prior is code-mode only, title-aware, and intentionally small. It does not
infer an answer path, use oracle labels, demote explicitly requested tests or
docs, or replace semantic/BM25 retrieval. The remaining dominant opportunity
is still the 18 current-main cases without an expected file in the top ten.
