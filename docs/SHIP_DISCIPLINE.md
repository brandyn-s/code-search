# Ship-Discipline Rules: Evidence Freshness and Affirmative Outcomes

**Status**: Operational policy. Rules 9 and 10 below extend the deep-assessment
workflow that introduced rules 1-8 (sub-agent triage, reproductions for
behavioral claims, scope tagging, etc., established across the 2026-05 deep
assessment of code-search). Both rules close gaps surfaced by the
PR #191 → #199 arc.

**Audience**: Any Claude session — agentic or operator-driven — that
proposes, ships, or closes a change to code-search that carries an outcome
claim (quality, latency, reliability, anything measurable).

---

## Rule 9 — Evidence Staleness Check

When citing eval / measurement results to justify a change, verify the
cited measurement still reflects current state.

**Trigger**: "I'm justifying this change with eval numbers from <PR X /
finding doc / CLAUDE.md table>."

**Required steps**:

1. Note the eval date AND the commit it ran against (e.g., "Phase C v2 ran
   2026-05-16 on commit `0cb78eb`").
2. List which modules / config knobs / prompts the comparison's arms
   touched. For a reranker A vs B comparison, that's typically
   `search/sonnet_reranker.py`, `search/listwise_sonnet_reranker.py`,
   `search/searcher.py` (dispatcher), `search/config.py` (knobs).
3. Run `git log --since=<eval date> -- <those modules>`. If ANY commits
   appear, the cited evidence is stale relative to current main.

**Stale evidence requires one of**:

- **(a) Re-run** the comparison on current main and cite the new numbers.
- **(b) Explain** in the PR body why the cited measurement still applies
  (e.g., "the upstream change was tested independently and provably
  doesn't interact with this comparison's signal" — and back that up).
- **(c) Don't ship** — block until (a) or (b) is in place.

**Concrete failure mode rule 9 catches**: PR #199 (listwise → default)
shipped citing Phase C v2 (PR #180, 2026-05-16) numbers. R9 (PR #193,
2026-05-23) modified one arm of the comparison (`search/sonnet_reranker.py`'s
`JUDGE_PROMPT`) between the eval and the ship. A `git log --since="2026-05-16"
-- search/sonnet_reranker.py search/listwise_sonnet_reranker.py` would
have surfaced #193's prompt edit, triggering re-eval or explicit
justification before flip.

---

## Rule 10 — Affirmative-Outcome Rule

Outcome claims ("X improves Y") require **affirmative measurement under
current conditions**. They are NOT met by any of:

- **(a) Stale measurements** (rule 9 applies)
- **(b) Diagnostics that rule out specific failure modes** but don't
  measure the outcome — e.g., "we verified listwise has the Nix-aware
  clause" rules out one failure mode but doesn't measure whether listwise
  outperforms current pointwise
- **(c) Defensibility narratives** — "the ship is reasonable because..."
  is rhetoric, not measurement
- **(d) Absence of contradicting evidence** — "we didn't find a problem"
  is logically distinct from "we found that the change works"
- **(e) Architectural reasoning substituted for measurement** —
  "listwise has lower p99 because it's one call instead of 15" is
  *architectural reasoning*; "listwise has measured p99=X on current
  main" is *empirical*. The former cannot close the latter's gap.

**Closing an outcome question must use one of three explicit phrasings**:

| Closing phrase | Meaning | Action |
|---|---|---|
| `"Measurement on current state shows X improves Y by Z (CI ...). DONE."` | Affirmative measurement confirms claim | Ship / keep shipped |
| `"Measurement on current state shows X does not improve Y."` | Affirmative measurement refutes claim | Decide: revert, refine, accept neutral |
| `"Outcome unmeasured under current state."` | No affirmative measurement available | BLOCKED ON MEASUREMENT — do not close as DONE |

**Smell-word watch list**: if a closing summary leans on any of
"defensible", "reasonable", "plausibly", "probably", "should be" without
attaching a measurement, rule 10 is being violated.

**Asymmetric evidence quality**: ship decisions often have multiple
justifications (e.g., #199 cited both latency AND quality). Each
justification must independently satisfy rule 10. A "the ship is defensible
on latency grounds" closing is only valid if latency is *measured* on
current main — not inferred architecturally.

---

## How these rules interact with the deep-assessment workflow (rules 1-8)

Rules 1-8 govern *how to investigate*. Rules 9-10 govern *what counts as
done* once a change ships.

| Stage | Rules that apply |
|---|---|
| Investigation: inventory existing context, baseline tests, sub-agents for triage, reproductions, fresh line numbers | 1-6 |
| Producing recommendations: tag scope, distinguish verified / documented / alleged | 7-8 |
| Justifying a ship: cite evidence | 9 |
| Closing a goal / merging a PR: claim an outcome | 10 |

A change can be SHIPPED under rule 8 (scope-tagged tractable / medium /
sweeping) while its outcome status remains BLOCKED under rule 10. This
is the correct shape for "ship the code; outcome eval blocks default
flip" or "ship the knob; eval decides what value to set."

---

## Operational notes

- **Where to record outcome status**: PR descriptions, finding docs in
  `docs/findings/`, and CLAUDE.md (for production-default claims). The
  status string ("DONE / DECIDE / BLOCKED ON MEASUREMENT") is part of
  the artifact's contract, not just commentary.
- **Granularity**: rules 9-10 apply per outcome claim. A PR can have
  multiple claims with different statuses (e.g., "latency: BLOCKED;
  graceful fallback: DONE via tests; quality: BLOCKED").
- **Cost**: rule 9's git-log check is free. Rule 10's affirmative
  measurement typically costs ~30 min wall + ~$1-5 API spend per
  paired-bootstrap CI (see `docs/EVAL_RUNBOOK.md`). When that cost is
  unavailable, the correct closing is BLOCKED, not a softened DONE.

---

## Case study — PR #191 → #199 retrospective applied to rules 9-10

| PR | Outcome claim | Rule 10 status |
|---|---|---|
| #191 manifest fatal | "Cross-artifact corruption now raises" | DONE — unit tests affirm |
| #192 search input validation | "Invalid inputs no longer crash" | DONE — unit tests affirm |
| #193 R9 Nix-aware pointwise | "Improves Nix queries" | Originally BLOCKED at merge; eval ran 2026-05-23 (see `docs/findings/2026-05-23-r9-nix-aware-pointwise-eval-finding.md`); now DONE with +0.1330 nix MRR CI [+0.0633, +0.2144] |
| #194 SearchConfig + registry | "No quality regression; same defaults" | DONE — unit tests affirm; no quality eval needed |
| #195 multi_chunk_merge deboost knob | "Knob is default-off no-op until tuned" | DONE — unit tests + finding doc affirm |
| #197 R10 retired | "Deboost provided no measured lift" | DONE — eval refuted; retire is correct response |
| #198 voyage-code-3 provider | "Available as non-default" | DONE — unit tests affirm; no quality claim made |
| #199 listwise default flip | "Listwise improves outcome over pointwise on current main" | **BLOCKED ON MEASUREMENT** — Phase C v2 cited but is stale per rule 9 (R9 modified the comparison's pointwise arm 2026-05-23); listwise advantage on current main is unmeasured |

The pattern that emerges: changes whose outcome reduces to unit-test
contracts (config refactor, registry pattern, knob defaults, structural
invariants) close as DONE naturally. Changes whose outcome is a quality
measurement need the eval gate, and absence of the gate means
BLOCKED — not a softened DONE.
