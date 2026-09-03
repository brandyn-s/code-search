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

## Documentation contracts

`tests/unit/test_documentation_contract.py` pins README, CLAUDE.md, and
`docs/ENV_REFERENCE.md` to the code's actual defaults. When you change a
default, update all three and the test in the same change.

## Releasing

See [docs/RELEASING.md](docs/RELEASING.md).
