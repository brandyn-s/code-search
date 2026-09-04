# Contributing

## Development setup

```bash
git clone https://github.com/brandyn-s/code-search.git
cd code-search
./scripts/install.sh          # creates .venv and installs this checkout
.venv/bin/python -m pytest tests/unit -q
```

`scripts/install.sh` only touches the checkout it runs from. Python 3.12 or
newer is required.

## Tests

- `tests/unit` must pass on every pull request; CI runs it on Linux, macOS,
  and Windows.
- `tests/acceptance` exercises the built wheel outside the source tree.
- `bench/eval` holds the frozen offline retrieval-floor fixture that CI runs
  as a regression gate. It is synthetic and safe to extend.

Retrieval-quality claims need measurement, not just green tests. If a change
is meant to improve ranking, include the eval you ran and its numbers in the
pull request description.

## Measuring a ranking change

Retrieval-quality claims need numbers on a corpus anyone can fetch. The public
evaluation set under [`bench/eval/public/`](bench/eval/public/README.md) has
30 queries each over pinned checkouts of Flask and Requests.

```bash
git clone --depth 1 --branch 3.1.1 https://github.com/pallets/flask /tmp/flask
python bench/eval/public/run_public_eval.py --corpus /tmp/flask \
  --gold bench/eval/public/golden_flask.json --output results/flask-before.json
# ... make your change ...
python bench/eval/public/run_public_eval.py --corpus /tmp/flask \
  --gold bench/eval/public/golden_flask.json --output results/flask-after.json
python bench/eval/public/compare.py results/flask-before.json results/flask-after.json
```

`compare.py` prints MRR, HR@1, Recall@10 and a paired bootstrap 95% CI on the
per-query delta. Put that output in the pull request. Only compare runs with
the same corpus commit, provider, model and reranker; the runner records all
four in the result file. Never commit `results/` or index artifacts.

## Extending code-search

Recipes for a new language chunker, embedding provider, reranker, or MCP
tool, each with the file and test to copy, are in
[docs/extending.md](docs/extending.md).

## Documentation contracts

`tests/unit/test_documentation_contract.py` pins README, CLAUDE.md, and
`docs/ENV_REFERENCE.md` to the code's actual defaults. When you change a
default, update all three and the test in the same change.

## Releasing

See [docs/RELEASING.md](docs/RELEASING.md).
