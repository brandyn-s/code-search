# Eval Runbook — paired-bootstrap CI for ship-decisions

> **Public repository note (2026-09):** the research harness under
> `bench/research/` and `benchmarks/` (gold sets harvested from redacted's
> internal codebases, sweep results, and eval outputs) is not included in this
> repository. References to those paths below are historical. The frozen
> offline retrieval floor under `bench/eval/` is included and runs in CI.

**Status**: Operational. Run this when a PR carries a "NEEDS EVAL BEFORE
MERGE" caveat in its commit body or PR description. Examples currently
queued for verification:

| Change | PR | What to measure |
|---|---|---|
| Listwise → default flip | n/a yet | Per `docs/LISTWISE_CANARY.md` graduation criteria |

History (verdict already shipped, kept here as worked examples — see
`docs/findings/` for the full per-cohort tables):

| Change | Verdict | Finding doc |
|---|---|---|
| Nix-aware pointwise rubric (R9, PR #193) | SHIP | `2026-05-23-r9-nix-aware-pointwise-eval-finding.md` |
| `multi_chunk_merge` deboost knob (R10, PR #195) | RETIRE | `2026-05-23-r10-multi-chunk-merge-deboost-eval-finding.md` |
| Containment-aware chunk merge (PR #227 + #229) | SHIP (neutral; correctness-motivated) | `2026-06-11-chunk-merge-containment-eval-finding.md` |

R10 is the worked retire example: shipped with `1.0 = no-op default`
("eval first, set value later") which is exactly the opt-in canary
pattern `~/.claude/rules/eval-shipping-discipline.md` plus
`feedback_no-opt-ins.md` say to refuse. When the deboost sweep
({0.5, 0.7, 0.85}) showed unfavorable-mean / CI-includes-zero on the
golden primary cohort across all three values, the rule's gate fired
RETIRE. The knob plumbing was removed in a follow-up PR (this file's
update was part of the same retire PR).

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
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label baseline

# Treatment: the change under test (set whatever env var or check out
# whatever commit puts the production extractor in the treatment state)
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label treatment
```

The harness uses `--label <run_label>` to construct the output path; it
does NOT accept `--output-dir`. Each run writes to
`<REPO_ROOT>/benchmarks/eval_v4/run_psm-full-<label>/` containing
`summary.json` + `golden_rows.json` + `harvested_rows.json`. The
bootstrap script reads `golden_rows.json` + `harvested_rows.json` from
each dir. Expect ~10-15 min wall per run depending on candidate pool
size and rerank latency.

To run a baseline against a different commit (e.g., the pre-PR-X state):
use `git worktree add <abs_path> -b <branch> <commit>` and invoke the
harness from inside the worktree — `REPO_ROOT` (computed by the harness
as `Path(__file__).parent.parent`) resolves to the worktree, so its
output writes to the worktree's `benchmarks/eval_v4/`. Then pass both
the worktree's and main checkout's output dirs to the bootstrap script
by absolute path. The R9 finding (`2026-05-23-r9-*.md`) walks this in
detail.

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

Per `~/.claude/rules/eval-shipping-discipline.md` (binary-decision rule,
2026-05-18 update):

| Per-cohort signal on the primary metric | Decision |
|---|---|
| Aggregate CI excludes zero in **favorable** direction AND no per-subproject CI strictly excludes zero in unfavorable direction | **SHIP DEFAULT-ON** |
| Aggregate CI includes zero AND mean delta is **favorable** on the primary metric | **SHIP DEFAULT-ON** (sub-clean CI on n=99-100 is a sample-size limit, not signal absence; document the CI explicitly in the PR) |
| Aggregate CI includes zero AND mean delta is **unfavorable** on the primary metric | **RETIRE** the knob entirely; do not ship as opt-in |
| Aggregate or per-subproject CI excludes zero in **unfavorable** direction | **REVERT** or refine before re-eval |

**Correctness fixes are exempt from the metric gate.** Per
`~/.claude/rules/eval-shipping-discipline.md` ("WHAT DOES NOT REQUIRE THIS
CHECK: security or correctness fixes — policy threshold isn't the bar at
all"), a change that removes objectively wrong index content (phantom
chunks, duplicated text, wrong-granularity stitches) stays on main on
correctness grounds even when its measured retrieval delta is neutral or
mildly unfavorable-mean. For these changes the eval's role is the REVERT
guard only: revert/refine if any CI strictly excludes zero in the
unfavorable direction. Worked example:
`2026-06-11-chunk-merge-containment-eval-finding.md` (golden mean −0.009,
CI includes zero → fix retained, claim recorded as measured-neutral).

**`HOLD` is not a verdict.** Earlier versions of this runbook had a
"CI includes zero → HOLD" cell; that bucket was removed when the rule
revised on 2026-05-18 to enforce the binary {SHIP, RETIRE} framing.
Shipping a knob with a no-op default and the rationale "eval first,
set later" is the exact opt-in canary pattern `feedback_no-opt-ins.md`
refuses — the operator does not remember to enable the knob, the
lever's value is never realized, the work becomes shelfware. When the
eval doesn't produce a default-flip, the verdict is RETIRE.

Examples in this repo:

- **SHIP example (R9, PR #193)**: golden aggregate +0.0495 MRR, CI
  [+0.0112, +0.0900] excludes zero favorable. nix subproject +0.1330,
  CI [+0.0633, +0.2144] confirms target lift. No per-subproject CI
  strictly excludes zero unfavorable. → SHIP default-on (the prompt
  change was already on `main`; the eval unblocked the
  NEEDS-EVAL-BEFORE-MERGE caveat).
- **RETIRE example (R10, PR #195)**: sweep over `MULTI_CHUNK_MERGE_DEBOOST
  ∈ {0.5, 0.7, 0.85}` against the 1.0 no-op default. All three values
  showed **negative-mean** golden delta with CI including zero
  (deboost=0.5: -0.0134 [-0.0379, +0.0077]; deboost=0.7: -0.0073
  [-0.0377, +0.0234]; deboost=0.85: -0.0009 [-0.0155, +0.0138]). Per
  the rule's "unfavorable mean + CI includes zero" cell → RETIRE the
  knob. Plumbing removed in the retire PR alongside this runbook
  update.
- **Mixed-signal example (voyage-code-3, 2026-05-15)**: aggregate
  within noise (-0.017, CI includes zero), mithrandir +0.119 (excludes
  zero, win), nix -0.091 (excludes zero, loss). Per-subproject
  unfavorable CI excludes zero → **REVERT-or-refine** branch fires.
  See `docs/findings/2026-05-15-voyage-code-3-ab-finding.md`.

## Worked example — R9 retro-procedure (Nix-aware pointwise rubric)

```bash
# Baseline: pointwise WITHOUT the Nix-aware clause. Use a worktree at
# the pre-PR-#193 commit so the baseline run's REPO_ROOT and the
# treatment run's REPO_ROOT differ, and the harness writes outputs
# under each worktree's `benchmarks/eval_v4/` separately.
git -C ~/Documents/GitHub/code-search worktree add \
    "C:/Users/<user>/worktrees/code-search-pre-r9" <commit-before-#193>
cd "C:/Users/<user>/worktrees/code-search-pre-r9"
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label pre-r9

# Treatment: pointwise WITH the Nix-aware clause (current main).
cd ~/Documents/GitHub/code-search
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label post-r9

# Bootstrap CI, pointing at the absolute output paths from both worktrees.
python bench/research/paired_bootstrap_per_subproject.py \
    --baseline-dir "C:/Users/<user>/worktrees/code-search-pre-r9/benchmarks/eval_v4/run_psm-full-pre-r9" \
    --treatment-dir "benchmarks/eval_v4/run_psm-full-post-r9" \
    --label-baseline "pointwise-no-nix" --label-treatment "pointwise-nix-aware"
```

**Expected**: positive delta on nix subproject; near-zero or no-signal
on assetman / libnet / mithrandir. If any non-nix subproject's CI
excludes zero in the unfavorable direction, refine the clause text
(narrow the trigger words) and re-eval. Outcome shipped 2026-05-23 at
`docs/findings/2026-05-23-r9-nix-aware-pointwise-eval-finding.md`.

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
`docs/findings/2026-05-23-r10-multi-chunk-merge-deboost-eval-finding.md`
(RETIRE example) or `2026-05-23-r9-nix-aware-pointwise-eval-finding.md`
(SHIP example):

- **Verdict in the first line: one of `SHIP` or `RETIRE`.** `HOLD` is
  not a valid verdict (see Step 3 above). `REVERT` is a sub-case of
  RETIRE — use REVERT when the change is on `main` and needs reverting;
  use RETIRE when the change shipped as an opt-in knob that needs full
  removal of the plumbing.
- Per-cohort mean delta + 95% CI table (one row per cohort, mark
  cells that exclude zero)
- Per-subproject deltas table (drives the regression check)
- **Rule-gate walk**: explicitly cite which row of Step 3's table
  fired and why. If the runbook and the rule disagree, the rule wins
  — surface the disagreement and update the runbook in the same PR.
- Cost summary (wall + $ spend)
- Decision rationale tying to the ship gate, and (for RETIRE) the
  follow-up PR that removed the plumbing
