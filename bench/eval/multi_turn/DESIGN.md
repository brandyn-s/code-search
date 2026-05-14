# Multi-turn retrieval eval — design decisions

**Date**: 2026-05-14
**Plan**: Phase C of `knowledge-base/plans/2026-05-14-code-stack-future-arcs-abc.md`
**Status**: First-author convention; refine as the corpus grows.

## Motivation

Today's code-search evals (PSM golden, PSM harvested, flask/requests adversarial) measure **single-turn retrieval**: one query, one expected target, MRR/HR@k on rank-of-target. Real Claude Code usage spans turns — the user asks "find auth handler", reads the result, then asks "show the verifier" or "where's the JWT issuer" depending on what they saw. The retrieval stack's quality across a multi-turn dialogue is not captured by single-turn metrics.

Phase C builds the first multi-turn benchmark for code-search to surface whether multi-turn is a different eval class on this corpus.

## Decision 1: metric

Choose **cumulative recall@k over a turn budget** rather than per-turn MRR.

| Option | Rationale | Chosen? |
|---|---|---|
| Per-turn MRR (avg over turns of within-turn MRR) | Mirrors single-turn metric; easy to compute | No |
| Cumulative recall@k at turn-N (was the gold file in top-k of ANY turn 1..N?) | Matches user experience ("did I find it in N tries?") | **Yes** |
| First-turn-to-gold rank (turn-number where gold first appears in top-k) | More precise but interpretable only for bundles where gold IS found within budget | Reported as secondary |

For each conversation bundle (3-5 turns, one gold target), report:

- `recall_at_5_in_3_turns` — was the gold in top-5 of ANY of turns 1, 2, or 3?
- `recall_at_10_in_5_turns` — was the gold in top-10 of ANY of turns 1-5?
- `first_turn_to_gold` — turn number where gold first appears in top-10 (None if not found within budget)

These are the headline aggregates; per-bundle per-turn rows are saved for downstream bootstrap CI on subsequent A/B tests.

## Decision 2: conversation-state model

Choose **Model A (agentic)** — turn-N's query is hand-authored knowing turn-(N-1)'s top-1 result.

| Option | Rationale | Chosen? |
|---|---|---|
| Model A — agentic | Simulates "user reads top-1, then refines based on what they saw". More realistic for Claude Code where the agent acts on results. | **Yes** |
| Model B — blind | Turn-N is authored from original intent only. Simpler to maintain; doesn't condition on stack output. | No (rejected; treats multi-turn as independent reformulations, missing the agentic feedback) |
| Model C — synthetic agent | Use Claude itself to generate turn-N from turn-(N-1)'s results. Adds LLM cost per query + non-determinism. | Future work |

Cost of Model A: when the retrieval stack changes (rerank weights, chunking, embedding), turn-2+'s queries may need re-authoring (their relevance was conditioned on the OLD top-1). Mitigation: design conversation bundles where turn-N depends on turn-(N-1)'s **expected** top-1, not the **actual** top-1. The bundle is then stable across stack changes; it just answers "does the stack return the expected top-1 at each turn?".

## Decision 3: fixture choice

Choose **flask + requests** (the β fixtures already indexed) for the first multi-turn corpus.

| Option | Rationale | Chosen? |
|---|---|---|
| PSM | Production representative; already indexed; aligns with PSM golden eval | No (read-only, authoring burden, large scope) |
| flask + requests | Already indexed at pinned SHAs; small, well-understood; supports turn-bundles like "find the request lifecycle entry → trace to dispatch → land at the adapter" | **Yes** |
| Synthetic small corpus | Easy to author against; control over difficulty | No (synthetic risk; want realistic shape) |

flask + requests are small enough that 20-30 conversation bundles × 3-5 turns is tractable in one authoring session; large enough that turns 2-N are meaningfully constrained by turn-1's choice.

## Decision 4: harness shape

The driver (`run_multi_turn.py`) loads conversation bundles, runs each turn's query via `search_code`, collects per-turn rank, computes per-bundle metrics, writes JSON. Mirrors `eval_against_psm_full.py` for per-query JSON dump (downstream bootstrap CI).

Bundle JSON schema:

```json
{
  "id": "flask-auth-chain-001",
  "category": "trace",
  "fixture": "flask",
  "turns": [
    {"turn": 1, "query": "Flask route decorator binding", "expected_files": ["src/flask/sansio/scaffold.py"]},
    {"turn": 2, "query": "url_for build URL with endpoint", "expected_files": ["src/flask/app.py", "src/flask/helpers.py"]},
    {"turn": 3, "query": "Werkzeug Map adapter and rules", "expected_files": ["src/flask/app.py"]}
  ]
}
```

Per-turn `expected_files` allows EACH turn to have its own gold; the cumulative-recall metric checks whether ANY turn's expected files appear in that turn's top-k. The driver supports a flag `--cumulative-target` for the alternative semantic where the bundle has ONE gold target and any turn finding it counts.

For first-author convention: each bundle has per-turn gold (more diagnostic; per-turn failure surfaces which retrieval stage breaks the chain).

## What this DESIGN.md is not

This is first-author convention. The first 20-30 bundles will surface gaps:
- If too many bundles have turn-1 already finding the gold (cumulative-recall ceiling at turn 1), the bundles aren't hard enough.
- If turn-3+ MRR doesn't differ from turn-1 MRR, multi-turn isn't a new eval class on this corpus (combined falsifier per the parent plan).
- If conversation-state Model A turns out to require frequent re-authoring as the stack changes, fall back to Model B.

Update this doc as the corpus grows.
