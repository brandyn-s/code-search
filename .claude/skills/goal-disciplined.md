---
name: goal-disciplined
description: |
  Wrapped /goal that enforces ship-discipline rules 9 (evidence staleness) and
  10 (affirmative outcome) from docs/SHIP_DISCIPLINE.md. Use this instead of
  /goal when the goal involves a measurable outcome claim (quality, latency,
  reliability, anything that requires evidence to close).
allowed-tools: ["*"]
---

# Disciplined Goal Wrapper

You are receiving a goal to work toward. Before starting, classify it and
apply the appropriate ship-discipline.

## Step 1 — Read the policy

If you haven't already in this session, read `docs/SHIP_DISCIPLINE.md`. It
defines:
- **Rule 9** (evidence staleness): cited eval results must reflect current
  state; run `git log --since=<eval-date> -- <comparison-arms>` to verify.
- **Rule 10** (affirmative outcome): outcome claims need measurement under
  current conditions; closing must use **DONE** / **DECIDE** / **BLOCKED
  ON MEASUREMENT** explicitly. Defensibility narratives, architectural
  reasoning, and absence-of-contradicting-evidence do NOT satisfy rule 10.

The smell-word watch list: *defensible*, *reasonable*, *plausibly*,
*probably*, *should be* — leaning on any of these without an attached
measurement is rule 10 in violation.

## Step 2 — Classify the goal

Before doing any work, classify whether the goal is **outcome-shaped** or
**contract-shaped**:

- **Outcome-shaped**: the goal's success requires a measurement claim.
  Examples: "ship a quality improvement", "make code-search faster", "find
  and fix the worst-ranked queries". Closing requires rule-10 phrasing.
- **Contract-shaped**: the goal's success reduces to a structural / test
  / integration property. Examples: "add provider X via the registry",
  "migrate the env vars into a typed config", "refactor without behavior
  change". Closing is verified by unit tests + structural inspection;
  rule 10 ceremony is not required.

State your classification explicitly in your first response. If the goal
is ambiguous (could be either), surface the ambiguity to the user and
ask which mode applies — don't pick silently.

## Step 3 — Staleness check (outcome-shaped only)

If the goal references prior eval evidence (a PR number, a finding doc, a
CLAUDE.md table entry), apply rule 9 before treating that evidence as
load-bearing:

1. Identify the date the cited evidence was produced.
2. Identify which modules / config knobs / prompts the comparison's arms
   touched.
3. Run `git log --since=<eval-date> -- <those modules>`. Report what you
   find.
4. If commits appear: the cited evidence is stale. Either re-run the
   eval, explicitly justify why the old result still applies, or treat
   the goal as BLOCKED ON MEASUREMENT.

State the result of the staleness check explicitly. Don't skip it
because "the cited evidence looks current" — the whole point is that
"looks current" is unreliable.

## Step 4 — Execute the goal

Normal goal-driven work. Make changes, run tests, verify, iterate. The
ship-discipline rules don't constrain how you investigate or implement —
only how you close.

## Step 5 — Close per rule 10

When you believe the goal is met, your closing summary MUST contain one
of these three phrasings (verbatim or near-verbatim):

| Phrasing | When to use |
|---|---|
| **`DONE — measurement on current state shows X improves Y by Z (CI ...)`** | The change has affirmative measurement confirming the outcome claim. Cite the measurement source (commit SHA, finding doc, summary.json). |
| **`DECIDE — measurement on current state shows X does not improve Y`** | The change has affirmative measurement that REFUTES the outcome claim. Recommend next step (revert, refine, accept neutral). |
| **`BLOCKED ON MEASUREMENT — outcome unmeasured under current state, reason: <why>`** | No current-state measurement is available (API key missing, corpus unavailable, eval not yet run). The change may still have shipped, but the outcome claim is open. |

For **contract-shaped** goals, no rule-10 phrasing is required. Close with
the standard "tests pass + change verified" summary.

## Self-check before declaring done

Before ending your turn with what you believe is a complete summary,
silently apply this checklist:

1. Did I classify the goal as outcome-shaped or contract-shaped at the
   start? (If you skipped this, do it now.)
2. If outcome-shaped: does my closing summary contain DONE / DECIDE /
   BLOCKED ON MEASUREMENT verbatim?
3. If outcome-shaped: does any other sentence in my closing summary lean
   on smell-words (defensible, reasonable, plausibly, probably, should
   be) without an attached measurement? If yes, rewrite that sentence.
4. If outcome-shaped: did I cite stale evidence as if it were
   current-state evidence? If yes, fix or downgrade the claim.
5. If I'm using DONE: is the measurement actually from current state,
   not from a comparison that has since been invalidated by upstream
   changes (rule 9)?

A closing that says "the change is defensible on latency grounds because
listwise is one call vs slowest-of-15" is FAILING this self-check. The
correct phrasing is "BLOCKED ON MEASUREMENT — latency lift inferred
architecturally, not measured on current main; re-run paired-bootstrap
CI to close."

## Notes

- This wrapper does not replace `/goal`'s underlying mechanism — it adds
  pre-flight classification + a closing-phrase contract. Use the
  ordinary `/goal` for purely mechanical work (renames, doc edits)
  where the outcome reduces to "the change compiles and tests pass".
- The Stop hook at `.claude/hooks/check_rule10_closing.sh` will also
  flag missing rule-10 phrases on outcome-shaped closures if installed
  (see `.claude/README.md` for the install path). The prompt-level
  enforcement above is the primary mechanism; the hook is
  defense-in-depth.
