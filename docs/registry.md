# MCP registry listing

`server.json` at the repository root describes this server for the official
MCP registry (`io.github.brandyn-s/code-search`). The registry stores
metadata only; the artifact is the `code-search-mcp` package on PyPI.

## Ownership verification

The registry verifies PyPI ownership by looking for the string
`mcp-name: io.github.brandyn-s/code-search` in the package's PyPI description,
which is rendered from `README.md`. The README carries it as an HTML comment,
which PyPI preserves. Do not remove it.

## Publish or update a listing

Run this after the PyPI release exists (the registry checks the package
version is on PyPI):

```bash
brew install mcp-publisher          # or the release tarball from
                                    # github.com/modelcontextprotocol/registry
mcp-publisher validate              # checks server.json against the schema
mcp-publisher login github          # authenticates as brandyn-s
mcp-publisher publish               # publishes server.json
```

Keep `version` in `server.json` (top level and `packages[0].version`) equal to
the PyPI version being listed; bump both when you release.

## Keep it in sync

`tests/unit/test_registry_manifest.py` asserts that `server.json` names the
PyPI package, matches the `pyproject.toml` version, declares only optional
environment variables with secrets marked, and that the README contains the
`mcp-name` verification string.
