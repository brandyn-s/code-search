# External benchmark runbook — CoIR & CodeRAG-Bench

**Status (2026-05-30): BLOCKED ON MEASUREMENT.** code-search has never been
evaluated on a public code-retrieval benchmark. Every headline number (MRR 0.828
golden / 0.670 PSM-multitarget) is internal and not comparable to any
leaderboard. This runbook is the procedure to produce *external, comparable*
numbers so the embedding-provider and reranker choices have a defensible basis
beyond a possibly-label-biased internal golden set (see
`docs/findings/2026-05-24-r9-extension-session-synthesis.md` and
`docs/findings/2026-05-30-embedding-model-string-confirmation.md`).

## Why this is the right next lever

1. The internal golden labels may be engine-biased (a sub-0.06-MRR delta may be
   noise — quantify the floor first with the **out-of-engine** harness below).
2. "voyage-4-large beats voyage-code-3 on code retrieval" is a surprising claim
   that currently rests *only* on the internal corpus. CoIR is the standard
   place to check it.

## Step 0 — De-bias the internal metric first (cheap, no API key)

```bash
python bench/research/out_of_engine_sample.py        # 50-query blind sheet
# (independent labeler fills independent_expected_files by reading the repo)
python bench/research/out_of_engine_bias_score.py    # prints the bias floor
```

Treat any provider/reranker MRR delta below the reported bias floor as noise.

## Step 1 — Datasets

- **CoIR** (https://github.com/CoIR-team/coir) — 10 code-retrieval tasks; the
  headline metric is **nDCG@10**. Each task is `{queries, corpus, qrels}`.
- **CodeRAG-Bench** (https://github.com/code-rag-bench/code-rag-bench) — headline
  metric **Recall@k** (k=10, plus end-to-end pass@1 for the generation half,
  which is out of scope here — we measure retrieval only).

Public leaderboard anchors to compare against (as of early 2026; re-check):
voyage-code-002 ≈ 56.3 CoIR nDCG@10; OpenAI-text-embedding-3-large ≈ 65.2;
Qodo-Embed-1-1.5B ≈ 68.5; Qodo-Embed-1-7B ≈ 71.5. We have **no published
voyage-4-large vs voyage-code-3 head-to-head** — generating one is the point.

## Step 2 — Adapt to the harness shape

The existing worker, `benchmarks/_eval_worker.py`, is invoked as:

```
python benchmarks/_eval_worker.py <config_json> <corpus_dir> <golden_path> <storage_dir>
```

and consumes a corpus directory plus a golden file of
`[{"query": str, "expected_files": [str, ...]}]`. Adapt each external task into
that shape:

- `corpus_dir`: materialize the task's corpus as files (one file per corpus doc,
  path = doc id). The chunker indexes them like any repo.
- `golden_path`: for each query, `expected_files` = the doc ids with qrel > 0.

Keep the qrels' graded relevance alongside (CoIR has graded judgments) so Step 4
can compute true nDCG@10 rather than the binary HR@k the worker emits.

## Step 3 — Run both providers (needs `VOYAGE_API_KEY`)

```bash
for prov in "voyage:voyage-4-large" "voyage-code-3:voyage-code-3"; do
  P="${prov%%:*}"; M="${prov##*:}"
  cfg="{\"name\":\"$M\",\"provider\":\"$P\",\"model\":\"$M\",\"use_input_type\":true,\"use_reranker\":false,\"use_voyage_input_type\":true}"
  python benchmarks/_eval_worker.py "$cfg" /path/to/coir_task_corpus /path/to/coir_task_golden.json /tmp/coir_$M
done
```

Run rerank-off first (isolates the embedding model), then repeat with
`RERANKER=sonnet` to measure the reranker's external lift.

## Step 4 — Score with the comparable metric

The worker emits MRR/HR@k. For leaderboard comparability, capture the per-query
ranked doc-id lists and feed them to `bench/research/ndcg.py`:

```python
from bench.research.ndcg import ndcg_at_k, recall_at_k, aggregate
per_query = [{"ndcg@10": ndcg_at_k(ranked, qrels, 10),
              "recall@10": recall_at_k(ranked, [d for d,r in qrels.items() if r>0], 10)}
             for ranked, qrels in runs]
print(aggregate(per_query))
```

(`ndcg.py` is dependency-free and self-tested: `python bench/research/ndcg.py`.)

## Step 5 — Close per ship-discipline rule 10

Record, in a dated `docs/findings/` doc, one of:
- `"Measurement on CoIR nDCG@10 shows voyage-4-large {beats|ties|trails}
  voyage-code-3 by X (CI ...). DONE."`
- or `"Outcome unmeasured"` → stays BLOCKED.

Only after this does the "voyage-4-large is our default" claim have an external
leg to stand on. Until then it is internal-only.
