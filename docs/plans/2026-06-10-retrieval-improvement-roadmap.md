# Retrieval improvement roadmap — 9 evidence-backed phases

**Date**: 2026-06-10
**Status**: PLAN — no code change in this commit. Each phase carries its own
ship gate; run phases independently in a local Claude Code session.
**Provenance**: 2026-06-10 deep-research survey of the semantic code-search
landscape (5 parallel research passes over vendor engineering blogs, ACL/EMNLP/
ICLR/NeurIPS papers, and OSS repos; 25 load-bearing claims adversarially
verified by a 3-vote panel against primary sources — 0 claims killed, 5
corrected; corrected values are used throughout). Full citation list at the
bottom.

**How to use this doc**: each phase is self-contained — evidence, hypothesis,
exact change, measurement protocol, and a binary ship gate per
`docs/EVAL_RUNBOOK.md` and `~/.claude/rules/eval-shipping-discipline.md`
({SHIP, RETIRE, REVERT} — no HOLD, no opt-in knobs with no-op defaults).
Quality phases (P1, P2, P3, P6) need `VOYAGE_API_KEY` + `ANTHROPIC_API_KEY`,
the PSM index, and the locked holdouts (`benchmarks/golden_multitarget.json`,
verify with `bench/research/freeze_holdout.py --verify`). Engineering phases
(P4, P5) gate on unit tests + cost/latency numbers, not retrieval CIs.

## Execution order and interactions

Two classes of change — do not stack them in one eval:

- **Index-changing** (P1 chunk budget, P6 embedder): every arm needs a full
  reindex. Use a separate storage dir per arm
  (`CODE_SEARCH_STORAGE=~/.claude_code_search_arm_<label>`) so arms are
  reproducible and the production index stays untouched.
- **Search-time** (P2 reranker calibration, P3 fusion): run against one fixed
  index; cheap to A/B.

Recommended sequence:

1. **P4, P5** (engineering; no quality gate) — any time, independent.
2. **P1** chunk budget sweep — settle the index geometry first, because P2/P3
   deltas measured on a 1500-NWS index may not transfer to a 2500-NWS index.
3. **P3** fusion arm, then **P2** reranker calibration, on the winning index.
4. **P6** embedder bake-off (most expensive index-changing sweep).
5. **P9** public benchmark anchor (one-time harness work; do before P7 so the
   big bet has an external yardstick).
6. **P8** router (scoped to advisory + measurement).
7. **P7** trace-distilled embedder (multi-week; gate on P6's outcome).

Every phase closes with an explicit **DONE / DECIDE / BLOCKED ON MEASUREMENT**
finding doc in `docs/findings/` (rule 10), and cites this plan.

---

## P1 — Chunk merge budget sweep: 1500 → {2000, 2500} NWS chars

**Evidence**: cAST (EMNLP 2025 Findings, arXiv 2506.15655) is the only
controlled measurement of AST chunking for code RAG. Its budget sweep peaks at
**2,000–2,500 non-whitespace chars**; our `MAX_CHUNK_NWS = 1500`
(`chunking/chunk_merging.py`) sits below the measured optimum. Same paper's
ablation: removing the sibling-merge step costs 16.8–19.0 nDCG points
(verified against Table; an earlier "18–25" citation was drift) — the merge
we already do is where the value lives; the budget is the open constant.
Caveat: cAST's gains are embedder-dependent (+4.3 Recall@5 on the GIST arm,
+2.4 BGE, +1.8 CodeSage; one SWE-bench arm gained only +0.3) — Voyage may
respond differently. That is what the sweep measures.

**Hypothesis**: golden MRR improves (or holds) at 2000–2500 NWS with fewer,
denser chunks.

**Change**: sweep the constant per arm (edit `MAX_CHUNK_NWS`; no env knob —
the winner gets hard-set, losers are discarded; per feedback_no-opt-ins).

**Protocol**:

```bash
# Arm = {1500 (baseline), 2000, 2500}; for each:
#   1. edit MAX_CHUNK_NWS in chunking/chunk_merging.py
#   2. CODE_SEARCH_STORAGE=~/.claude_code_search_arm_nws<N> -> full reindex of ~/PSM
#   3. RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label nws<N>
python bench/research/paired_bootstrap_per_subproject.py \
  --baseline-dir benchmarks/eval_v4/run_psm-full-nws1500 \
  --treatment-dir benchmarks/eval_v4/run_psm-full-nws2000 \
  --label-baseline nws1500 --label-treatment nws2000
# repeat for nws2500
```

Also record per arm: chunk count, mean chunk NWS, and mean formatted-result
token size (bigger chunks raise per-result token cost for the MCP consumer —
report it; a quality win that doubles token cost needs the tradeoff stated).

**Ship gate**: runbook binary rule on golden MRR (primary), harvested MRR
(secondary). Winner's constant is hard-set; finding doc records all three
arms. Note in the finding that production indexes need a one-time full
reindex to realize the change.

**Effort**: ~1 day (3 reindexes + 3 eval runs ≈ 1–2 h wall each).

---

## P2 — Pointwise reranker tie mitigation (calibration, not architecture)

**Evidence**: coarse pointwise LLM scores produce ties in **67% of pairwise
comparisons** (arXiv 2603.12520); fine-grained ordinal scales close the
pointwise-listwise gap ("Likert or Not", arXiv 2505.19334). Our
`hybrid_prior_fallback` (max score < threshold → keep hybrid order,
`search/sonnet_reranker.py`) is the in-house symptom of exactly this failure
mode — uniformly-low/tied scores. External listwise evidence (EMNLP 2025
Findings, arXiv 2508.16757: listwise degrades least on novel queries — 8% vs
12% pointwise; note that 8% is a *degradation rate*, not an effectiveness
advantage) does NOT override our own 2026-05-23 measurement that listwise
regresses on harvested (CI excludes zero unfavorable; see
`docs/findings/2026-05-23-listwise-default-eval-finding.md`). So: keep
pointwise, fix its calibration. Precedent in-house: rank aggregation across
independent passes is the code-graph iter=2/MRR pattern (+38pp class accuracy
there — protocol-level, but the same tie-breaking mechanism).

**Hypothesis**: finer score granularity (and, if needed, 2-pass mean
aggregation) reduces tie rate and `hybrid_prior_fallback` rate without
regressing MRR — and may convert some fallbacks into wins.

**Change (arm A — free)**: widen `JUDGE_PROMPT`'s scale from 0–10 to 0–100 in
`search/sonnet_reranker.py`; scale `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD`
default 6 → 60 (same boundary, finer bins). One API call per candidate as
today; zero cost delta.

**Change (arm B — 2x rerank cost, only if A's CI includes zero with
favorable mean)**: score each candidate twice (independent calls), use the
mean; ties broken by hybrid order as today (stable sort).

**Protocol**: fixed index (P1 winner). Instrument first: log per-cohort tie
rate (top-2 equal scores) and fallback rate from `[RERANK_REASON]` sidecar
lines — `benchmarks/judge_bias_diagnostic.py` (PR #223) already parses judge
scores; extend it to report tie rate. Then:

```bash
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label scale10   # baseline
# apply arm A
RERANKER=sonnet python benchmarks/eval_against_psm_full.py --label scale100
python bench/research/paired_bootstrap_per_subproject.py ... --label-treatment scale100
```

**Ship gate**: runbook binary rule on golden MRR; secondary signals: tie rate
and `hybrid_prior_fallback` rate (report, don't gate). Arm B only runs if arm
A lands in the "CI includes zero + favorable mean" cell and the tie-rate drop
is small. Per the lever-class-exhaustion rule: A and B are the same lever
class (pointwise calibration) — two failures here mean stop, not a third
calibration variant.

**Effort**: arm A ~half a day; arm B ~1 day.

---

## P3 — Convex (normalized weighted-score) fusion arm vs weighted RRF

**Evidence**: Bruch, Gai & Ingber (ACM TOIS 2023, arXiv 2210.11934, verified):
RRF is sensitive to its parameters, and a **tuned convex combination of
normalized scores outperforms RRF in- and out-of-domain**. We already tuned
RRF's k (20) and weights (0.65/0.35, PR #90) — the score-fusion alternative
has never had an arm. `benchmarks/fusion_diagnostic.py` (PR #222) already
measures per-leg contribution and can host the comparison.

**Hypothesis**: min-max-normalized weighted score fusion ≥ weighted RRF on
golden MRR at equal weights, because score magnitudes carry information that
rank-only fusion discards.

**Change**: eval-only fork in `search/searcher.py::_hybrid_search` — compute
both fusions behind a harness-only flag (`FUSION_METHOD=convex` read ONLY by
the eval harness path); per-query: min-max normalize FAISS sims and BM25
scores (BM25 ranks are negative-better — invert to positive-better before
normalizing), fused = 0.65·v + 0.35·b, sweep vector weight {0.60, 0.65, 0.70,
0.75}. If SHIP: hard-set convex as the only path and delete RRF; if RETIRE:
delete the fork entirely. No shipped knob either way.

**Protocol**: fixed index; `RERANKER=off` first (isolate the fusion effect),
then confirm end-to-end with `RERANKER=sonnet`; labels
`rrf-baseline` / `convex-w<NN>`; paired bootstrap per arm.

**Ship gate**: runbook binary rule, golden MRR primary. Watch per-subproject
CIs — fusion changes historically split by language cohort.

**Effort**: ~1 day including the sweep.

---

## P4 — Chunk-hash-keyed document-embedding cache (cost/latency, no quality gate)

**Evidence**: Cursor caches embeddings keyed by chunk hash so Merkle-synced
re-indexes and branch switches only embed novel content
(cursor.com/blog/secure-codebase-indexing, docs); their org-wide reuse cut
p99 time-to-first-query from 4.03 h to 21 s (vendor numbers, but the
mechanism is the point). We re-embed every chunk of every changed file on
every reindex, and full reindexes re-embed the entire corpus.

**Change**: in `embeddings/embedder.py`, before each `embed_chunks` /
`embed_chunks_grouped` API call, look up
`sha256(embedding_content)` in a new SQLite table
`doc_embedding_cache(content_sha, provider, model, dim, embedding BLOB,
created_at)` in the storage dir; only novel texts go to the API; store on
return. **Key includes (provider, model)** — the 2026-06-10 query-cache
cross-provider poisoning fix (PR #224) is the cautionary tale; reuse its
keying discipline and add the same cross-provider unit tests
(`tests/unit/test_embedder_cache_keying.py` is the template). Grouped/
contextualized providers (voyage-context) embed chunks with document context —
**exclude `encode_grouped` providers from the cache** (same text in a
different file-group context legitimately yields a different vector); cache
only the flat `embed_chunks` path. Add cache-size cap + `clear_index`
integration.

**Measurement (not a retrieval gate — vectors are byte-identical on hits)**:
(a) full reindex of PSM twice — second run's wall time and API token count
(target: >90% token reduction on the no-change reindex); (b) simulated branch
switch (touch 5% of files) — same metrics; (c) unit tests: hit returns
identical vector; provider/model isolation; grouped-provider bypass.

**Ship gate**: unit suite green + measured cost table in the finding doc.
DONE = measured numbers reported.

**Effort**: ~1 day.

---

## P5 — Stale-vector auto-compaction (closes the loop PR #224 opened)

**Evidence**: internal. FAISS deletions are "rebuild on demand"; modify/delete
churn accumulates stale vectors (score inflation + recall decay class of bugs
fixed in PR #224; the remaining debt is bloat). PR #224 added
`live_chunks` / `stale_vectors` to `get_stats` — nothing acts on them yet.

**Change**:
1. `verify_index_integrity` (mcp_server) reports
   `stale_ratio = stale_vectors / max(live_chunks, 1)` with a recommendation
   string when > 0.25.
2. `search_code`'s `_metadata.freshness` gains `"stale_index"` advisory at the
   same threshold (additive, never breaks the response — observability-path
   failures must not break search).
3. `IncrementalIndexer.incremental_index`: when `stale_ratio > 0.5` at the
   START of a run, log `[REINDEX_PROGRESS] compaction: stale_ratio=… —
   escalating to full reindex` and dispatch to `_full_index` (which already
   clears + rebuilds). Default-on, threshold hard-coded (no knob): 0.5 means
   the index holds more garbage than live data — full reindex is strictly
   better and self-limiting (ratio resets to 0).
4. Tests: ratio computation; escalation fires at >0.5 and not at <0.5;
   `_metadata` advisory shape.

**Measurement**: unit tests + one manual churn scenario (index, modify 60% of
files twice, observe escalation + post-compaction ratio 0). Record
full-reindex wall time on PSM in the finding doc so the escalation cost is a
known quantity.

**Ship gate**: tests green; wall-time cost documented. DONE = merged with
numbers.

**Effort**: ~half a day.

---

## P6 — Embedder bake-off refresh (rule-9 staleness sweep)

**Evidence**: third-party measured MTEB-Code: voyage-code-3 **79.84**,
jina-code-embeddings-1.5b 78.94 (arXiv 2508.21290), **Qwen3-Embedding-8B
80.68** (arXiv 2506.05176) — open models now bracket the commercial SOTA.
Voyage's own docs (checked 2026-06-10) still recommend voyage-code-3 for code;
voyage-4 is general-purpose (MoE; no published code-domain breakout). Our
default voyage-4-large won the 2026-04/05 internal A/Bs — rule 9 says re-cite
or re-run when comparators move; the comparator set has moved.

**Arms** (each = registry factory in `embeddings/embedder.py` + full reindex
into its own `CODE_SEARCH_STORAGE`):
- `voyage-4-large` (baseline, current default)
- `voyage-code-3` (re-check; provider exists)
- `Qwen3-Embedding-0.6B` local via sentence-transformers (the 8B needs a GPU
  box — start with 0.6B as the local-feasibility probe; add 8B only if 0.6B
  lands within noise of baseline)
- `jina-code-embeddings-1.5b` — **license gate first**: CC-BY-NC-4.0;
  confirm internal-tooling use is acceptable before spending the reindex.

**Protocol**: per arm: full reindex → `eval_against_psm_full.py --label
emb-<arm>` with `RERANKER=off` (pure retrieval) AND `RERANKER=sonnet`
(end-to-end) → paired bootstrap vs baseline. Record $/1M tokens and reindex
wall time per arm (an open local model that ties the API baseline is a win on
cost/privacy even at ΔMRR=0 — say so in the gate).

**Ship gate**: default flips only on the runbook rule (favorable golden CI or
favorable mean); a tie for an open local model yields a documented DECIDE:
"local arm viable for airgapped deployments, default unchanged."

**Effort**: ~2 days (4 reindexes, 8 eval runs).

---

## P7 — Trace-distilled custom embedder (the big bet; design + data first)

**Evidence**: the only two large measured production retrieval wins both come
from custom embedders trained on usage signal: Cursor distills LLM hindsight
rankings of agent-session traces (+12.5% avg agent accuracy vs grep-only,
+2.6% code retention on 1k+-file repos — vendor benchmark, verified wording);
GitHub's custom Copilot embedder (+37.6% retrieval lift, 2× throughput, 8×
smaller index). Mechanism, not magic: train the embedder to rank what actually
helped real sessions.

**Scope discipline**: multi-week, GPU-dependent, and gated on P6 — if an open
base model (Qwen3-Embedding) can't get within striking distance of
voyage-4-large zero-shot, fine-tuning it has a longer hill. Run as three
sub-phases, each with its own exit:

- **P7a — data pipeline (1 week)**: extend `benchmarks/harvest_real_queries.py`
  into a triplet factory: for each harvested real-session query, replay the
  search, and have Sonnet label hindsight-positive chunks ("which retrieved
  chunk did the subsequent edit actually use?") and hard negatives (retrieved,
  plausible, unused — same file or same name, different body). Target ≥5k
  triplets; lock as `traces_v1.lock` with the freeze_holdout tooling. EXIT:
  labeled-set size + a 50-triplet human spot-check precision ≥0.8, else
  BLOCKED.
- **P7b — fine-tune (1 week)**: contrastive InfoNCE on the P6-winning open
  base (LoRA first); hold out 20% of triplets for training-side validation.
  EXIT: held-out triplet accuracy beats zero-shot base by ≥5pp, else DECIDE
  (negative finding, stop).
- **P7c — eval as a provider arm**: register `local-distilled` factory, full
  P6 protocol vs current default. Runbook gate decides the default.

**Effort**: 3+ weeks elapsed. Do not start before P6's finding is written.

---

## P8 — Query-shape routing to the code-graph localization agent (advisory)

**Evidence**: function-level localization is where flat embedding retrieval
caps out: on Loc-Bench, embedding baseline (CodeRankEmbed) 43.39% function
Acc@10 vs graph-agent (LocAgent + Claude-3.5) ≈59–61% (the two papers'
columns agree at 60.71 Acc@10 in SweRank's reproduction; panel voters split
59.29-vs-60.71 on column labels — within 1.4pp either way) vs code-tuned
rerank pipeline (SweRankLLM-Large) 71.25% (arXiv 2505.07849, ICLR 2026;
corrected from an Acc@15 misread). We own both halves: code-search (this
repo) and code-graph's `code_localize_agent` (86%/84.5%/73.5% file/class/func
internal, iter=2 protocol). Verbose multi-sentence issue-shaped queries are
the agent's documented win condition; short symbol queries are ours.

**Scope**: advisory routing only — no cross-MCP delegation.
1. In `mcp_server` `search_code`: when the query exceeds a shape heuristic
   (≥2 sentences or ≥25 tokens), add
   `_metadata.routing_hint = "verbose query: code-graph
   code_localize_agent typically wins on issue-shaped localization"`.
   Additive, never blocks results.
2. Mirror the guidance in CLAUDE.md's tool-selection note and the search
   orchestration skill (`docs/plans/2026-03-15-search-orchestration-skill.md`
   lineage) so the agent reading the hint knows the tool exists.

**Measurement**: define the verbose cohort inside the existing harvested set
(same ≥25-token heuristic); report code-search MRR on that cohort vs overall
in the finding doc. If the cohort gap is small (<0.05 MRR), DECIDE: routing
hint unnecessary — remove it. (The full head-to-head vs the agent runs in
code-graph's harness, not here.)

**Effort**: ~half a day + measurement.

---

## P9 — Public benchmark anchor (Loc-Bench file-level)

**2026-07-26 bounded foundation:** the shared evaluation worker now emits
stable query-ID-keyed, unique document rankings and scores graded nDCG@k and
Recall@k. The manual CoIR workflow persists those scored runs. This removes a
worker-level prerequisite for public evaluation, but P9 remains **BLOCKED ON
MEASUREMENT** until the reachable Loc-Bench lock and current-main baseline
below are produced. See
`docs/findings/2026-07-26-p9-ranked-list-ndcg-foundation.md`.

**Evidence/motivation**: all our quality numbers live on private PSM fixtures;
nothing is externally comparable. Cursor/GitHub/Voyage publish on private
benches too — an anchor on a public set is a differentiator and keeps our
internal deltas honest. code-graph already ships Loc-Bench tooling
(`bench/research/eval_locbench_*.py`) and learned the hard lesson that
**58/200 instance base-commits had been GC'd by 2026-05-12** — pin the
reachable subset first.

**Scope**:
1. Port a thin harness `bench/eval/locbench_file_level.py`: for each reachable
   Loc-Bench instance — clone at base commit, `index_directory`, issue text →
   `search_code(k=10)`, score file-level Acc@10 against expected files.
2. Freeze the reachable-instance list (`locbench_reachable_v1.lock`).
3. Run once at current main → that number is the standing baseline; re-run
   only for index-changing default flips (P1/P6/P7 winners).

**Reference points** (file-level Acc@10, different setups — context not
competition): LocAgent+Claude-3.5 86.07; CodeRankEmbed retrieval baseline
80.89 (arXiv 2503.09089). A flat-search MCP harness landing near the
embedding baseline is the expected zone.

**Ship gate**: DONE = baseline number + lock file committed; this phase
produces a measurement capability, not a default flip.

**Effort**: ~2 days (indexing hundreds of repos is the wall-clock cost; run
a 50-instance subset first).

---

## Citations

- cAST: arXiv 2506.15655 (EMNLP 2025 Findings) — split-then-merge AST
  chunking; 2000–2500 NWS optimum; merge-ablation −16.8–19.0 nDCG.
- Pointwise judge ties: arXiv 2603.12520 (67% ties); fine-grained scales:
  arXiv 2505.19334; judgment-distribution scoring: arXiv 2503.03064.
- Listwise generalization: arXiv 2508.16757 / ACL 2025.findings-emnlp.305
  (8%/12%/15% novel-query degradation; contamination-controlled set).
- Fusion: Bruch, Gai & Ingber, arXiv 2210.11934, ACM TOIS 2023.
- Hybrid complementarity: CrossCodeEval arXiv 2310.11248 (BM25 > UniXcoder);
  CoIR arXiv 2407.02883 (dense ≈2× BM25 on code overall).
- Quantization: huggingface.co/blog/embedding-quantization (int8 raw
  90.8–100% by model; ~99% with rescoring).
- Cursor: cursor.com/blog/semsearch (+12.5% avg, +2.6% retention 1k+ files;
  trace-distilled embedder); cursor.com/blog/secure-codebase-indexing
  (Merkle + chunk-hash embedding cache + index reuse).
- GitHub: github.blog Copilot embedding model post, 2025-09-24 (+37.6%,
  2× throughput, 8× smaller).
- Sourcegraph: sourcegraph.com/blog/how-cody-understands-your-codebase
  (embeddings removed for Zoekt/BM25 at enterprise scale).
- Claude Code agentic-grep position: x.com/bcherny/status/2017824286489383315.
- LocAgent: arXiv 2503.09089 (ACL 2025) — Loc-Bench; SweRank: arXiv
  2505.07849 (ICLR 2026) — fn Acc@10 71.25 vs LocAgent 60.71, $0.011 vs
  $0.66/instance; SweRank+: arXiv 2512.20482 (agent re-added).
- Embedding benchmarks: arXiv 2508.21290 (Jina measurements incl.
  voyage-code-3 79.84 MTEB-Code); arXiv 2506.05176 (Qwen3-Embedding-8B 80.68).
- Anthropic contextual retrieval: anthropic.com/news/contextual-retrieval
  (−49%/−67%; vendor, includes code corpora, no independent code replication).
- Upstream: github.com/zilliztech/claude-context (~40% token reduction claim,
  vendor n=2 case studies).
