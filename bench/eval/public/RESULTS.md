# Public evaluation results

One complete, reproducible run of both public gold sets with the fully local
configuration (no API keys). Anyone can re-run these commands and compare.

## Run of 2026-09-04

| Gold set | Corpus commit | n | MRR | HR@1 | Recall@10 | p50 latency | Wall time (index + 30 queries) |
|---|---|---|---|---|---|---|---|
| `golden_flask.json` | `7fff56f5172c48b6f3aedf17ee14ef5c2533dfd1` (pallets/flask 3.1.1) | 30 | 0.7494 | 0.6333 | 0.9667 | 95 ms | 19.6 s |
| `golden_requests.json` | `021dc729f0b71a3030cefdbec7fb57a0e80a6cfd` (psf/requests v2.32.4) | 30 | 0.8778 | 0.8000 | 0.9667 | 91 ms | 19.5 s |

Configuration in effect (recorded in each result file's `config` block):

| Setting | Value |
|---|---|
| code-search-mcp | 0.4.0 (source checkout, mcp SDK 2.1.1) |
| `EMBEDDING_PROVIDER` | `local` (`sentence-transformers/all-MiniLM-L6-v2`, `[local]` extra) |
| `RERANKER` | `off` |
| `k` | 10 |
| Python | 3.12 |
| Machine | Apple M5 Max, 128 GB, macOS 26.6.2 |

## Exact commands

```bash
mkdir -p /tmp/eval-corpora && cd /tmp/eval-corpora
git clone --depth 1 --branch 3.1.1 https://github.com/pallets/flask flask
git clone --depth 1 --branch v2.32.4 https://github.com/psf/requests requests
git -C flask rev-parse HEAD      # 7fff56f5172c48b6f3aedf17ee14ef5c2533dfd1
git -C requests rev-parse HEAD   # 021dc729f0b71a3030cefdbec7fb57a0e80a6cfd

cd <code-search checkout>
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev,local]'
mkdir -p results
RERANKER=off CODE_SEARCH_STARTUP_AUDIT=0 .venv/bin/python bench/eval/public/run_public_eval.py \
  --corpus /tmp/eval-corpora/flask --gold bench/eval/public/golden_flask.json \
  --output results/flask-local.json
RERANKER=off CODE_SEARCH_STARTUP_AUDIT=0 .venv/bin/python bench/eval/public/run_public_eval.py \
  --corpus /tmp/eval-corpora/requests --gold bench/eval/public/golden_requests.json \
  --output results/requests-local.json
```

Each runner prints one summary line, for example:

```text
golden_flask.json: n=30 MRR=0.7494 HR@1=0.6333 Recall@10=0.9667 p50=95ms provider=local reranker=off
```

## Reading these numbers

- MRR aggregates reciprocal rank across queries; it does not by itself give
  top-result accuracy. HR@1 is the share of queries whose first result is an
  expected file; Recall@10 is the share with an expected file anywhere in the
  top ten.
- These are the offline baseline. Voyage embeddings and Sonnet reranking are
  expected to score higher; run the same commands with `VOYAGE_API_KEY` and
  `ANTHROPIC_API_KEY` set to measure them on your account.
- Only compare runs with the same corpus commit, provider, model, and reranker.
  Use `compare.py` for the paired bootstrap interval; a change counts as an
  improvement only when the interval excludes zero on the right.
- The result files under `results/` are not committed. The numbers above come
  from the files this run produced; regenerate them rather than trusting the
  table when you change ranking code.
