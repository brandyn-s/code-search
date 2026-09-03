# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0] - Unreleased

First release from the public primary repository `brandyn-s/code-search`.

### Changed
- Package renamed from `redacted-code-search` to `code-search-mcp`; the console
  script is `code-search-mcp`, so `uvx code-search-mcp` starts the server.
- `RERANKER` defaults to `auto`: Sonnet pointwise reranking when
  `ANTHROPIC_API_KEY` is set, otherwise `off`. Previously the default was
  `sonnet`, which degraded silently on installs without a key.
- `CODE_SYNONYM_PROFILE` defaults to `generic`; the domain-specific `corsair`
  profile remains available as an opt-in.
- The server prints one startup line naming the resolved embedding provider
  and reranker mode.
- `mcp` is pinned below 2.0; this code targets the 1.x `FastMCP` API.
- Release workflow publishes the attested wheel to PyPI via trusted publishing.

### Removed
- The research harness (`bench/research/`, `benchmarks/`), internal evaluation
  findings, plan documents, and internal process tooling were removed from the
  public tree. Repository history was rewritten to exclude evaluation data
  derived from internal codebases.

### Added
- `LICENSE` (GPL-3.0-only), `CHANGELOG.md`, `CONTRIBUTING.md`, issue and pull
  request templates, and `docs/clients.md` covering Claude Code, Claude
  Desktop, Cursor, Codex CLI, Windsurf, and generic stdio configuration.

## [0.3.6] and earlier

Internal releases from the originating organization. Their wheels and
attestations were built by that organization's release workflow and are not
reproducible from this repository's history.
