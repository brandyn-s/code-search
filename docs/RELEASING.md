# Releasing

Releases are cut by the `Release` workflow (`.github/workflows/release.yml`)
from `main` via `workflow_dispatch`.

1. Bump `version` in `pyproject.toml`, move the `Unreleased` section of
   `CHANGELOG.md` under the new version, merge to `main`, and wait for the
   `Merge CI` run on that commit to succeed.
2. Run the `Release` workflow with the version (without the `v` prefix).
   It builds the wheel once, attests it with GitHub artifact attestations,
   publishes an immutable GitHub release with `code_search_mcp-<v>-py3-none-any.whl`,
   `SHA256SUMS`, and the provenance bundle, verifies every asset, and then
   publishes the same wheel to PyPI.
3. Confirm `uvx code-search-mcp@<v> --help` works from a clean machine.

## One-time PyPI setup

PyPI publishing uses trusted publishing (OIDC); no token is stored.

- Create the PyPI project `code-search-mcp` and add a pending trusted
  publisher: owner `brandyn-s`, repository `code-search`, workflow
  `release.yml`, environment `pypi`.
- Create a `pypi` environment in this repository's GitHub settings. Optionally
  require a reviewer on it.

Artifact attestations require the repository to be public (or GitHub
Enterprise Cloud).
