# External benchmark runbook — CoIR & CodeRAG-Bench

**Status (2026-08-12): ONE PUBLIC FILE-LOCALIZATION ENDPOINT MEASURED; EXTERNAL
PROVIDER COMPARISON STILL BLOCKED.** The released v0.3.5 server has now produced
a scored public result on a frozen, balanced LocBench n=80 endpoint: Acc@1
0.375, Acc@10 0.788, and MRR@10 0.503. Against the Sourcegraph public endpoint,
the paired Acc@1 comparison was 22 wins, 4 losses, and 54 ties (p=0.00053).
That establishes narrow superiority on this revision-pinned file-localization
endpoint only; it is not a general code-search or platform-superiority claim.
See the README's **Public release evidence** section for the complete scope.

The CoIR/CodeRAG-Bench provider experiment in this document remains unrun.
Internal provider numbers (including MRR 0.828 golden / 0.670
PSM-multitarget) are not comparable to a public leaderboard, and the LocBench
result does not compare voyage-4-large with voyage-code-3. The manual workflow
emits query-ID-stable document rankings plus graded nDCG@10 and Recall@10
artifacts. It still needs a credentialed run before any external
provider-quality conclusion is supportable (see
`docs/findings/2026-05-24-r9-extension-session-synthesis.md` and
`docs/findings/2026-05-30-embedding-model-string-confirmation.md`).

## Why this is the right next lever

1. The internal golden labels may be engine-biased (a sub-0.06-MRR delta may be
   noise — quantify the floor first with the **out-of-engine** harness below).
2. "voyage-4-large beats voyage-code-3 on code retrieval" is a surprising claim
   that still rests *only* on the internal corpus. The public LocBench result
   evaluates the released product, not this provider ablation; CoIR is the
   standard place to check it.

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
`[{"query_id": str, "query": str, "expected_files": [str, ...]}]`. Adapt each
external task into that shape:

- `corpus_dir`: materialize the task's **complete candidate corpus** as files
  (one file per corpus doc, path = doc id). Restricting the corpus to judged
  positives would make retrieval artificially easy and invalidate comparison.
- `golden_path`: for each query, `expected_files` = the doc ids with qrel > 0.

Fetch the complete qrels split before selecting the query subset, then keep
each selected query's full graded relevance map keyed by `query_id` alongside
the golden file. Truncated qrels change the ideal DCG denominator and invalidate
nDCG@10.

## Step 3 — Run both providers (needs `VOYAGE_API_KEY`)

```bash
for M in voyage-4-large voyage-code-3; do
  cfg="{\"name\":\"$M\",\"provider\":\"voyage\",\"model\":\"$M\",\"use_input_type\":true,\"use_reranker\":false,\"needs_reindex\":true}"
  EMBEDDING_PROVIDER=voyage EMBEDDING_MODEL="$M" VOYAGE_INPUT_TYPE=on RERANKER=off \
    python benchmarks/_eval_worker.py \
      "$cfg" \
      /path/to/coir_task/corpus \
      /path/to/coir_task/golden.json \
      "/tmp/coir_$M" \
      --k 10 \
      --qrels /path/to/coir_task/qrels_graded.json \
      --output "/tmp/results_$M.json" \
      --unique-documents
done
```

Run rerank-off first (isolates the embedding model), then repeat with
`RERANKER=sonnet` to measure the reranker's external lift.

## Step 4 — Inspect the comparable metric

Each `results_*.json` file contains:

- the unique document ranking for every stable `query_id`;
- per-query graded `ndcg@10` and binary `recall@10`; and
- aggregate `external_metrics`.

The manual `external-benchmarks` workflow runs the same command and uploads the
adapted golden/qrels files even when the credentialed evaluation is skipped.
Credentialed runs also include the complete JSON results and human-readable
logs in the `coir-metrics` artifact. The adapter retrieves complete qrels and
corpus splits from manifest- and row-count-validated Parquet exports rather
than scanning the paginated row API. Scoring fails closed on zero-query runs,
missing or non-positive qrels, duplicate document IDs, or fewer than `k` unique
ranked documents, so a green artifact cannot silently represent an incomplete
measurement.
`bench/research/ndcg.py` remains dependency-free and self-tested:
`python bench/research/ndcg.py`.

## Step 5 — Close the remaining provider question per ship-discipline rule 10

Record, in a dated `docs/findings/` doc, one of:
- `"Measurement on CoIR nDCG@10 shows voyage-4-large {beats|ties|trails}
  voyage-code-3 by X (CI ...). DONE."`
- or `"Outcome unmeasured"` → stays BLOCKED.

Only after this does the "voyage-4-large is our default" claim have an external
leg to stand on. Until then it is internal-only.
