# Retrieval evaluation harness

> **Public repository note (2026-09):** the research harness under
> `bench/research/` and `benchmarks/` (gold sets harvested from
> private internal codebases, sweep results, and eval outputs) is not included in this
> repository. References to those paths below are historical. The frozen
> offline retrieval floor under `bench/eval/` is included and runs in CI.

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

### Frozen offline merge gate

From the repository root, this literal invocation builds the tiny
deterministic model, verifies every committed fixture checksum, creates an
index in empty temporary storage, switches to that successful index, and
evaluates five objective queries independently through the production
semantic/vector and keyword/BM25 search paths:

```bash
FROZEN_TMP="$(mktemp -d)"
trap 'rm -rf "$FROZEN_TMP"' EXIT
python3 bench/eval/build_frozen_model.py --output "$FROZEN_TMP/model"
CODE_SEARCH_STORAGE="$FROZEN_TMP/storage" \
PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 bench/eval/check_retrieval_floor.py \
    --mode index-and-eval \
    --project bench/eval/fixtures/frozen-v1/corpus \
    --gold bench/eval/fixtures/frozen-v1/gold.json \
    --manifest bench/eval/fixtures/frozen-v1/manifest.json \
    --provider local \
    --model "$FROZEN_TMP/model" \
    --floor-semantic-mrr 0.80 \
    --floor-semantic-hr1 0.80 \
    --floor-keyword-mrr 0.80 \
    --floor-keyword-hr1 0.80 \
    --rerank off
```

The local provider, BoW model, float32 index, disabled reranker, disabled
query expansion, fixed hash seed, and Hugging Face offline settings make this
a keyless catastrophic-regression check. It intentionally exercises real
chunking, `CodeEmbedder`, FAISS, FTS5/BM25, server indexing, project switching,
and search. Each production retrieval arm must independently clear its floor.

### Index-and-eval mode (manual / external provider)

Same script can index a target project fresh and run a small gold file:

```bash
.venv/Scripts/python.exe bench/eval/check_retrieval_floor.py \
    --mode index-and-eval \
    --project /path/to/some-repo \
    --gold /path/to/gold.json \
    --provider voyage \
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
the live PSM gate remains a local pre-commit measurement. CI instead builds
the small checked-in synthetic fixture above from source on every run; it is
a catastrophic floor, not a replacement for the PSM quality benchmark.

## Holdout lock

`holdout/multitarget_v1.lock` pins the SHA256 of
`benchmarks/golden_multitarget.json`. See
`bench/research/freeze_holdout.py` for promotion.
