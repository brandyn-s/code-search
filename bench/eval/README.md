# Retrieval evaluation harness

## Floor gate (`check_retrieval_floor.py`)

Defensive guard that asserts retrieval MRR / HR@1 stays above a floor.
Two modes — local PSM check (cheap, reads existing eval summary) and
index-and-eval (CI mode, indexes a target project from scratch).

### Local PSM workflow

Before opening a PR that touches retrieval code (`mcp_server/`,
`embeddings/`, `chunking/`, `search/`, `fusion/`, `benchmarks/`), run the
PSM eval and assert the floor:

```bash
# 1. Re-baseline PSM (≈5-10 min, ≈$1.50 in API costs)
.venv/Scripts/python.exe benchmarks/eval_against_psm_full.py \
    --provider voyage --gold multi-target

# 2. Assert the floor against the fresh summary
.venv/Scripts/python.exe bench/eval/check_retrieval_floor.py \
    --mode summary \
    --summary benchmarks/eval_v4/run_psm-full-voyage-multitarget/summary.json \
    --floor-golden-mrr 0.62 \
    --floor-golden-hr1 0.50 \
    --floor-harvested-mrr 0.73 \
    --floor-harvested-hr1 0.65
```

The gate exits `0` on PASS and `1` on FAIL with the offending metric
named. Floors are set ≈2-3 pp below the 2026-05-14 baseline (golden MRR
0.640, HR@1 0.529, harvested MRR 0.752, HR@1 0.672) so the gate fires
only on real regression, not bootstrap noise.

### Index-and-eval mode (CI / smoke)

Same script can index a target project fresh and run a small gold file:

```bash
.venv/Scripts/python.exe bench/eval/check_retrieval_floor.py \
    --mode index-and-eval \
    --project /path/to/some-repo \
    --gold /path/to/gold.json \
    --floor-mrr 0.50 \
    --floor-hr1 0.40 \
    --rerank off
```

Gold file format (list of objects):

```json
[
  {"query": "find auth handler", "expected_files": ["src/auth.py"]},
  {"query": "voyage embedder", "expected_files": ["embeddings/voyage_embedder.py"]}
]
```

`expected_files` is matched against indexed paths via exact match or
suffix match (the indexer stores relative paths; gold paths can be
suffix-only).

### Why CI does not run the live PSM eval

PSM is read-only and ≈50K files; cloning + indexing in CI costs
≈30 min wall + ≈$1 in Voyage embedding charges per run. Per the Phase α
falsifier in `knowledge-base/plans/2026-05-14-code-search-consolidation-roadmap.md`,
the gate is scoped to local-PSM pre-commit; CI tests the gate script
itself so the local workflow stays trustworthy. To extend to CI-eval,
either (a) commit a small frozen-PSM snapshot index, or (b) build a
synthetic self-fixture (open work).

## Holdout lock

`holdout/multitarget_v1.lock` pins the SHA256 of
`benchmarks/golden_multitarget.json`. See
`bench/research/freeze_holdout.py` for promotion.
