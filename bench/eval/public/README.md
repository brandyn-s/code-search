# Public evaluation set

Two 30-query gold sets over public repositories, pinned by commit so anyone
can reproduce a number and compare a change against it. This is the
measurement instrument for ranking changes; the frozen fixture under
`bench/eval/fixtures/` is only a catastrophic-breakage gate.

| Gold set | Corpus | Pin | Queries | Distinct expected files |
|---|---|---|---|---|
| `golden_flask.json` | [pallets/flask](https://github.com/pallets/flask) | tag `3.1.1`, commit `7fff56f5172c48b6f3aedf17ee14ef5c2533dfd1` | 30 | 22 |
| `golden_requests.json` | [psf/requests](https://github.com/psf/requests) | tag `v2.32.4`, commit `021dc729f0b71a3030cefdbec7fb57a0e80a6cfd` | 30 | 17 |

Each entry is `{"query", "expected_files", "category"}`. Queries are written
as an engineer would ask an agent ("Session object cookie persistence and
adapter mounting"), not as keyword lookups, and every expected path exists at
the pinned commit.

## Fetch the corpora

```bash
mkdir -p /tmp/eval-corpora && cd /tmp/eval-corpora
git clone --depth 1 --branch 3.1.1 https://github.com/pallets/flask flask
git clone --depth 1 --branch v2.32.4 https://github.com/psf/requests requests
git -C flask rev-parse HEAD      # 7fff56f5172c48b6f3aedf17ee14ef5c2533dfd1
git -C requests rev-parse HEAD   # 021dc729f0b71a3030cefdbec7fb57a0e80a6cfd
```

## Run

```bash
# local provider (needs `pip install -e '.[local]'`), reranker off
python bench/eval/public/run_public_eval.py \
  --corpus /tmp/eval-corpora/flask --gold bench/eval/public/golden_flask.json \
  --output results/flask-baseline.json

# same corpus with a cloud provider
VOYAGE_API_KEY=... EMBEDDING_PROVIDER=voyage RERANKER=off \
python bench/eval/public/run_public_eval.py \
  --corpus /tmp/eval-corpora/flask --gold bench/eval/public/golden_flask.json \
  --output results/flask-voyage.json
```

The runner indexes into a temporary storage directory (or `--keep-storage`
to reuse one across runs), waits for `index_ready`, runs every query with
`k=10`, and writes the configuration in effect (provider, model, reranker,
corpus commit, index generation) next to the per-query rankings. Never commit
`results/` or index artifacts.

## Compare two runs

```bash
python bench/eval/public/compare.py results/flask-baseline.json results/flask-change.json
```

Prints MRR, HR@1 and Recall@10 for both runs plus a paired bootstrap
95% confidence interval on the per-query reciprocal-rank delta. A change is
only a measured improvement when the interval excludes zero on the right;
exit code 2 means it is significantly worse. Only compare runs with the same
corpus commit, provider, model and reranker.

## Adding queries

Keep the sets balanced across categories, write the query the way an agent
would phrase it, list every file that fully answers it, and confirm the paths
exist at the pinned commit. Re-pin both corpora together when you move them.
