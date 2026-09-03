# code-search

Hybrid semantic and lexical code search as an MCP server, with persistent
per-project indexes and source-backed evidence for every claim an agent makes.

Most code-search tools return ranked snippets and leave the agent to assert
things about them. code-search fuses vector search and BM25 over
structure-aware chunks, optionally reranks with an LLM, and can return
evidence references bound to an exact source revision and index generation.
An agent can show you *where* an answer came from, and the server refuses to
emit evidence when the index no longer matches the checkout.

## Quick Start

Install and run with [uv](https://docs.astral.sh/uv/) (or `pipx run
code-search-mcp`, or `pip install code-search-mcp`):

```bash
claude mcp add code-search --scope user -- uvx code-search-mcp
```

Any MCP client works; see [docs/clients.md](docs/clients.md) for Claude
Desktop, Cursor, Codex CLI, Windsurf, and generic stdio configuration:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "uvx",
      "args": ["code-search-mcp"],
      "env": { "VOYAGE_API_KEY": "optional", "ANTHROPIC_API_KEY": "optional" }
    }
  }
}
```

No API keys are required. Without `VOYAGE_API_KEY` the server embeds with a
local model (downloaded on first index); without `ANTHROPIC_API_KEY` it skips
LLM reranking. With both set it uses Voyage `voyage-4-large` and Sonnet
reranking. The server prints one startup line naming the resolved mode.

Then, from the client:

```text
index_directory(directory_path="/absolute/repository")
get_index_status(project_path="/absolute/repository")   # wait for index_ready=true
search_code(query="where is request authentication enforced?", search_mode="auto")
search_code_evidence(query="where is request authentication enforced?")
```

Python 3.12 or newer is required.

## What It Provides

- Natural-language and exact-signal retrieval over source code and Markdown.
- Hybrid FAISS vector search plus SQLite FTS5 BM25, fused with weighted
  reciprocal-rank fusion (RRF).
- Structure-aware chunking for 17 language modes across 21 registered file
  extensions, with bounded adjacent-chunk merging.
- Provider-aware project indexes, Merkle-based incremental updates, background
  indexing progress, and source/index identity checks.
- Immutable generation publication with manifest verification and last-good
  fallback.
- Optional cloud embeddings and reranking, or a fully local path.
- Backend-issued, generation-bound evidence IDs for exact source lines.
- Project-balanced discovery across as many as 25 isolated indexes.

## Choose the Right Operation

| Need | Use |
|---|---|
| Exact literal, regex, or known symbol | `rg` or `search_code(search_mode="keyword")` |
| Conceptual discovery | `search_code` in `auto` or `hybrid` mode |
| Evidence for a claim | `search_code_evidence`; select an emitted `evidence_ref.id` |
| Similar implementations | `find_similar_code` |
| File-level issue localization | `code_localize` |
| Discovery across local projects | `search_all_projects`, then a project-bound follow-up |
| Callers, callees, inheritance, impact | [code-graph](https://github.com/brandyn-s/code-graph) |
| Vulnerability-grade variable taint | CodeQL; search and graph reachability do not substitute for it |

## How It Works

```mermaid
flowchart LR
    A[Source checkout] --> B[Language-aware chunks]
    B --> C[Contextual headers]
    C --> D[Embedding provider]
    D --> E[FAISS vectors]
    B --> F[SQLite FTS5]
    E --> G[Vector candidates]
    F --> H[BM25 candidates]
    G --> I[RRF + deterministic boosts]
    H --> I
    I --> J[Optional PPR]
    J --> K[Optional LLM rerank]
    K --> L[Ranked retrieval context]
    L --> M[Atomic evidence candidates]
```

Indexing splits files at semantic boundaries, merges small adjacent units
within the **400-2500 NWS budget** (a 2,500 non-whitespace character budget),
embeds them, and publishes the result as an immutable generation whose
manifest records provider, model, and source identity. A Merkle DAG drives
incremental updates. The ending checkout identity must match the starting
identity or the generation never becomes ready.

Retrieval runs vector and BM25 arms, fuses ranks with content-mode weights
(**code 65/35, docs 70/30, all 50/50 (vector/BM25)**), applies chunk-type
boosts and file diversification, then optionally graph PPR and a reranker.
`search/retrieval.py` owns candidate retrieval, `search/fusion.py` owns RRF and
boosts, `search/query_expansion.py` owns synonym expansion,
`search/result_models.py` owns response types, and `search/pipeline.py`
composes PPR and reranking. Any reranker failure preserves the hybrid order and
is reported in `_metadata.reranker`.

`search_code_evidence` wraps the same path and adds `atomic_source_line`
candidates, each carrying an immutable `evidence_ref` bound to repository,
source revision, index generation, path, and line. If the generation is stale,
evidence fails closed while ordinary retrieval continues. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/INDEX_IDENTITY.md](docs/INDEX_IDENTITY.md).

## MCP Tools

| Tool | Purpose |
|---|---|
| `search_code` | Ranked semantic, keyword, or hybrid retrieval in the active project |
| `search_code_evidence` | Normal retrieval plus atomic, immutable evidence candidates |
| `code_localize` | Aggregate chunk results into a file-level issue ranking |
| `find_similar_code` | Nearest chunks to a prior result |
| `get_file_context` | Inspect indexed chunks for a file or line window |
| `search_all_projects` | Project-balanced discovery across isolated indexes |
| `index_directory` | Start background full or incremental indexing |
| `get_indexing_progress` | Poll the active indexing job |
| `cancel_indexing` | Request bounded cancellation at a progress checkpoint |
| `get_index_status` | Read readiness, source/index identity, provider, and stats |
| `verify_index_integrity` | Check canonical chunk IDs, dependent stores, and manifests |
| `list_projects` | List indexed projects and active state |
| `switch_project` | Change the active project without modifying index data |
| `index_test_project` | Index the packaged demonstration fixture |
| `clear_index` | Destructively remove the active index |
| `delete_project` | Destructively remove one project and all of its artifacts |

Treat `clear_index` and `delete_project` as irreversible.

## Configuration

The complete table is in [`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md).

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | Voyage when its key exists; otherwise local | `voyage`, `voyage-code-3`, `voyage-context`, `openai`, `jina`, or `local` |
| `EMBEDDING_DIMENSION` | `unset` | Required positive output-dimension contract for custom remote embedding models; built-in models derive it automatically |
| `RERANKER` | `auto` | `auto` (Sonnet when `ANTHROPIC_API_KEY` is set, else off), `sonnet`, `listwise`, `cross-encoder`, or `off` |
| `CONTENT_MODE` | `code` | `code`, `docs`, or `all`; controls retrieval weights and boosts |
| `CONTEXTUAL_HEADERS` | `on` | Add file/type/name context before embedding |
| `QUERY_EXPANSION` | `on` | Expand BM25 terms with the selected synonym profile |
| `CODE_SYNONYM_PROFILE` | `generic` | `generic`, `corsair`, or `off` |
| `CODE_SYNONYMS_PATH` | `unset` | Optional JSON overlay for the selected synonym profile |
| `QUANTIZATION` | `int8` | FAISS `int8`, `float32`, or `binary` storage |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Root for project indexes and local models |
| `CODE_SEARCH_LOG_LEVEL` | `INFO` | Minimum server log level |
| `CODE_SEARCH_LOG_QUERY_TEXT` | `off` | Opt in to raw query text in logs |
| `CODE_SEARCH_QUERY_HISTORY` | `metadata` | `off`, `metadata`, or `full`; metadata excludes raw query text |
| `CODE_SEARCH_QUERY_RETENTION_DAYS` | `30` | Query-history retention window in days |

These settings are process-static: they are read once when the MCP server starts. Restart the MCP server after changing them.

Cloud providers receive query text and source-derived chunk text. Use the local
provider and `RERANKER=off` when that boundary is unacceptable.

## Measured Evidence

On a frozen balanced public LocBench n=80 endpoint, release `v0.3.5` measured
Acc@1 `0.375`, Acc@3 `0.613`, Acc@10 `0.788`, and MRR@10 `0.503`; a
Sourcegraph public endpoint measured `0.150/0.175/0.188/0.165` on the same
endpoint (paired Acc@1 sign test `p=0.00053`). This establishes narrow superiority for this frozen file-localization endpoint, not general platform superiority. The same server indexed a 39-million-line LLVM checkout into
183,663 chunks in about ten minutes with 3.65 GB peak RSS.

MRR aggregates reciprocal rank across queries; it does not by itself determine top-result accuracy, a typical rank, or the probability that any one query succeeds. These are historical evaluation results, not current production guarantees.

## Comparison to Alternatives

- Sourcegraph has the broader search language, history UX, organization ACLs,
  and a managed indexing fleet. This project is a focused MCP retrieval
  backend with evidence semantics.
- Grep-style tools (`rg`, Claude Code's built-in search) win on exact tokens.
  code-search is for conceptual queries and for evidence you can cite.
- Cross-project search does not federate scores and has no organization
  authorization model.
- The index is not updated on every keystroke; reindex or rely on the
  freshness path.
- Search does not prove call relationships or runtime behavior. Use
  code-graph, runtime instrumentation, or CodeQL for those.

## Troubleshooting

| Symptom | Check | Recovery |
|---|---|---|
| Index job started but queries are incomplete | `get_indexing_progress()` and `get_index_status()` | Wait for `index_ready=true` and a matching source/index identity. |
| Evidence is absent | Index identity or generation is stale | Reindex the unchanged checkout; evidence fails closed until identity is current. |
| Reranker is unavailable | `_metadata.reranker.reason` and the startup line | Results keep the hybrid order; set `ANTHROPIC_API_KEY` or `RERANKER=off`. |
| Changed environment is ignored | Server was already running | Restart the MCP process; configuration is process-static. |
| `ModuleNotFoundError: mcp.server.fastmcp` | `mcp` 2.x installed | Install `code-search-mcp` in its own environment; it pins `mcp<2`. |

## Verified Install

Every GitHub release ships the wheel, a `SHA256SUMS` file, and a build
provenance bundle. Install the verified v0.4.0 release when you want to check
provenance before running anything:

```bash
REPO="brandyn-s/code-search"
TAG="v0.4.0"
WHEEL="code_search_mcp-0.4.0-py3-none-any.whl"
BUNDLE="code_search_mcp-0.4.0-provenance.jsonl"

gh release download "$TAG" --repo "$REPO"
shasum -a 256 -c SHA256SUMS            # Linux: sha256sum --check SHA256SUMS

RELEASE_SHA="$(gh api "repos/$REPO/git/ref/tags/$TAG" --jq '.object.sha')"
gh attestation verify "$WHEEL" \
  --bundle "$BUNDLE" \
  --deny-self-hosted-runners \
  --repo "$REPO" \
  --signer-workflow "$REPO/.github/workflows/release.yml" \
  --source-ref "refs/heads/main" \
  --source-digest "$RELEASE_SHA"
gh release verify "$TAG" --repo "$REPO"
gh release verify-asset "$TAG" "$WHEEL" --repo "$REPO"

python3 -m venv .venv
.venv/bin/python -m pip install "$WHEEL"
```

## Development

```bash
git clone https://github.com/brandyn-s/code-search.git
cd code-search
./scripts/install.sh
.venv/bin/python -m pytest tests/unit -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [docs/RELEASING.md](docs/RELEASING.md),
and [CHANGELOG.md](CHANGELOG.md).

## License

GPL-3.0-only. See [LICENSE](LICENSE). This project is derived from
[FarhanAliRaza/claude-context-local](https://github.com/FarhanAliRaza/claude-context-local)
(GPL-3.0); modifications are copyright 2026 redacted Security and are released
under the same license.
