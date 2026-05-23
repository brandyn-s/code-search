# Eval Runbook — paired-bootstrap CI for ship-decisions

**Status**: Operational. Run this when a PR carries a "NEEDS EVAL BEFORE
MERGE" caveat in its commit body or PR description. Examples currently
queued for verification:

| Change | PR | What to measure |
|---|---|---|
| Nix-aware pointwise rubric (R9) | #193 | Pointwise pre-R9 vs post-R9, regression on non-nix subprojects |
| `multi_chunk_merge` deboost knob (R10) | #195 | Default off vs `MULTI_CHUNK_MERGE_DEBOOST=0.7` (or sweep) |
| Listwise → default flip | n/a yet | Per `docs/LISTWISE_CANARY.md` graduation criteria |

## What this runbook is NOT

- **Not a quick sanity check.** This is the ship-discipline gate; budget
  30-60 min of wall time + ~$1-5 in Anthropic / Voyage API spend per run.
- **Not a substitute for the unit-test suite.** Unit tests pin code
  contracts; this measures retrieval quality on labeled holdouts.

## Prerequisites

1. **A PSM-indexed corpus**. The eval harness operates on
   `~/.claude_code_search/projects/<hash>/index/` produced by indexing
   `~/PSM` (or equivalent). If the corpus has shifted since the last
   eval, run `index_directory(directory_path="~/PSM",
   incremental=false)` to reindex first.
2. **API keys** in the shell environment:
   - `VOYAGE_API_KEY` for embedding the query at search time
   - `ANTHROPIC_API_KEY` for Sonnet rerank (omit for `RERANKER=off` runs)
3. **The locked golden + harvested holdouts**:
   - `benchmarks/golden_multitarget.json` (102 + 183 queries)
   - `multitarget_v1.lock` (SHA-pinned; verify with
     `bench/research/freeze_holdout.py --verify` before each eval)

## Workflow: paired-bootstrap CI

### Step 1: produce two eval runs

```bash
# Baseline: production behavior at HEAD~1 (or whatever the comparison is)
RERANKER=sonnet python benchmarks/eval_against_psm_full.py \
    --label baseline \
    --output-dir benchmarks/eval_v4/run_baseline

# Treatment: the change under test
# (Example for R10 — switch on the deboost knob)
RERANKER=sonnet MULTI_CHUNK_MERGE_DEBOOST=0.7 python benchmarks/eval_against_psm_full.py \
    --label treatment \
    --output-dir benchmarks/eval_v4/run_treatment
```

Each run writes a `per_query_results.json` to its output directory and a
`summary.json` with aggregate MRR, HR@1, nDCG@10. Expect ~10-15 min wall
per run depending on candidate pool size and rerank latency.

### Step 2: paired bootstrap CI

```bash
python bench/research/paired_bootstrap_per_subproject.py \
    --baseline-dir benchmarks/eval_v4/run_baseline \
    --treatment-dir benchmarks/eval_v4/run_treatment \
    --label-baseline "off" --label-treatment "deboost-0.7"
```

Output is a per-cohort + per-subproject table. Read:
- `n`: paired query count (queries appearing in BOTH runs)
- `mean_delta`: average per-query MRR change
- `95% CI [lo, hi]`: bootstrap confidence interval (10K resamples)
- `*` flag: CI excludes zero — signal exists in the indicated direction

### Step 3: apply the ship gate

Per `~/.claude/rules/eval-shipping-discipline.md`:

| Outcome | Decision |
|---|---|
| Aggregate CI excludes zero in **favorable** direction AND no per-subproject regression with CI excluding zero | **SHIP** |
| Aggregate CI includes zero | **HOLD** — no signal; do not flip default |
| Aggregate or per-subproject CI excludes zero in **unfavorable** direction | **REVERT** or refine before re-eval |

A treatment that "wins on average but regresses one subproject with CI
excluding zero" is the most common ambiguous outcome. The 2026-05-15
voyage-code-3 finding is a textbook case: aggregate within noise,
mithrandir +0.119 (excludes zero, win), nix -0.091 (excludes zero,
loss). Ship gate FAIL → HOLD. Documented at
`docs/findings/2026-05-15-voyage-code-3-ab-finding.md`.

## R9-specific procedure (Nix-aware pointwise rubric)

```bash
# Baseline: pointwise WITHOUT the Nix-aware clause (revert PR #193's
# JUDGE_PROMPT change locally, or check out the commit before #193).
git stash  # if needed
git checkout <commit-before-#193>
RERANKER=sonnet python benchmarks/eval_against_psm_full.py \
    --label pre-r9 \
    --output-dir benchmarks/eval_v4/run_pre_r9

# Treatment: pointwise WITH the Nix-aware clause (current main).
git checkout main
RERANKER=sonnet python benchmarks/eval_against_psm_full.py \
    --label post-r9 \
    --output-dir benchmarks/eval_v4/run_post_r9

python bench/research/paired_bootstrap_per_subproject.py \
    --baseline-dir benchmarks/eval_v4/run_pre_r9 \
    --treatment-dir benchmarks/eval_v4/run_post_r9 \
    --label-baseline "pointwise-no-nix" --label-treatment "pointwise-nix-aware"
```

**Expected**: positive delta on nix subproject; near-zero or no-signal
on assetman / libnet / mithrandir. If any non-nix subproject's CI
excludes zero in the unfavorable direction, refine the clause text
(narrow the trigger words) and re-eval.

## R10-specific procedure (multi_chunk_merge deboost)

The tag was added in PR #193; the consumer knob shipped in #195.
Default `MULTI_CHUNK_MERGE_DEBOOST=1.0` is a no-op, so no eval is
needed to merge the knob itself. The eval sizes the LIFT and picks
the production deboost factor.

Suggested sweep: `0.5, 0.7, 0.85, 1.0`. The 1.0 cell is the baseline.

```bash
for deboost in 0.5 0.7 0.85; do
    RERANKER=sonnet MULTI_CHUNK_MERGE_DEBOOST=$deboost python benchmarks/eval_against_psm_full.py \
        --label "deboost-$deboost" \
        --output-dir "benchmarks/eval_v4/run_deboost_$deboost"
done

# Baseline (deboost=1.0, the default no-op).
RERANKER=sonnet python benchmarks/eval_against_psm_full.py \
    --label baseline \
    --output-dir benchmarks/eval_v4/run_baseline

# Pairwise bootstrap against each treatment.
for deboost in 0.5 0.7 0.85; do
    echo "=== Deboost $deboost vs baseline ==="
    python bench/research/paired_bootstrap_per_subproject.py \
        --baseline-dir benchmarks/eval_v4/run_baseline \
        --treatment-dir "benchmarks/eval_v4/run_deboost_$deboost" \
        --label-baseline "1.0" --label-treatment "$deboost"
done
```

**Expected**: the lift (if any) should grow with deboost magnitude up
to some optimum, then degrade. Pick the deboost value whose aggregate
CI excludes zero in the favorable direction with the smallest deboost
magnitude (the smallest change to the production scoring path).

## Holdout integrity check

Before every eval run, verify the holdout hasn't drifted:

```bash
python bench/research/freeze_holdout.py --verify
```

Should print "OK" with the SHA matching `multitarget_v1.lock`. If it
fails, the holdout was modified — re-lock intentionally
(`freeze_holdout.py --lock`) or restore from git.

## Cost summary

| Activity | Wall | API spend |
|---|---|---|
| Index PSM-full (if stale) | 5-50 min | ~$1-2 (Voyage batch) |
| One eval run (n=285) at `RERANKER=sonnet` | 10-15 min | ~$0.5-1 (Sonnet rerank) |
| One eval run at `RERANKER=off` | 5-8 min | $0 |
| Paired-bootstrap CI | <30 sec | $0 |
| Holdout verify | <1 sec | $0 |

A full A/B (2 runs + bootstrap) is typically ~30 min + $1-2.

## Where to file the result

If the eval is a ship-decision artifact, add a finding to
`docs/findings/YYYY-MM-DD-<change>-eval-finding.md`. Pattern after
`docs/findings/2026-05-15-voyage-code-3-ab-finding.md`:

- Verdict in the first line (SHIP / HOLD / REVERT)
- Per-subproject deltas table
- Bootstrap CI values
- Cost summary
- Decision rationale tying to the ship gate
