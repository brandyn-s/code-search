# code-search

Hybrid semantic and lexical code retrieval for MCP clients, with persistent
per-project indexes, explicit freshness, and backend-issued source evidence.

`code-search` is the discovery half of redacted's verifiable code-intelligence
stack. It answers “where is the code that does X?” Code relationships and
impact questions belong to
[code-graph](https://github.com/redacted-org/code-graph).

> **Current state (reviewed 2026-08-13):** implementation baseline
> `cbdb9bd` is exactly the published [`v0.3.6`](https://github.com/redacted-org/code-search/releases/tag/v0.3.6)
> tag. This statement describes source and release identity; it does not assert
> which version any MCP client currently has installed.

Read the [architecture and operating model](docs/ARCHITECTURE.md) for component
boundaries, failure behavior, and storage contracts. The combined
[`code-search` + `code-graph` HTML guide](https://github.com/redacted-org/code-graph/blob/main/docs/index.html)
is a self-contained page that can be downloaded and opened locally.

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
- Optional cloud embeddings/reranking or local embeddings.
- Backend-issued, generation-bound evidence IDs for exact source lines.
- Project-balanced discovery across as many as 25 isolated indexes without
  treating scores from different indexes as comparable.

## Choose the Right Operation

| Need | Use |
|---|---|
| Exact literal, regex, or known symbol | `rg` or `search_code(search_mode="keyword")` |
| Conceptual discovery | `search_code` in `auto` or `hybrid` mode |
| Evidence for a claim | `search_code_evidence`; select an emitted `evidence_ref.id` |
| Similar implementations | `find_similar_code` |
| File-level issue localization | `code_localize` |
| Discovery across local projects | `search_all_projects`, then a project-bound follow-up |
| Callers, callees, inheritance, impact, or graph paths | `code-graph` |
| Vulnerability-grade variable taint | CodeQL; neither search nor graph reachability substitutes for it |

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
    J --> K[Optional Sonnet rerank]
    K --> L[Ranked retrieval context]
    L --> M[Atomic evidence candidates]
```

### Indexing and Publication

1. `index_directory` captures the source identity and starts a background job.
2. Language-specific chunkers split files at semantic boundaries where
   supported; a merge pass combines small adjacent units within the configured
   **400-2500 NWS budget**: a 2,500 non-whitespace character budget. This
   preserves short gap code without turning a retrieval unit into a file-sized
   citation.
3. The configured embedder produces vectors. FAISS stores vector search state;
   SQLite stores FTS5 and chunk metadata.
4. A Merkle DAG classifies additions, changes, and deletions for incremental
   runs. Excess stale-vector churn escalates to compaction.
5. Candidate artifacts are validated and published as an immutable generation.
   `manifest/current.json` names the active generation; a verified prior
   generation provides fail-closed recovery.
6. The ending checkout identity must match the starting identity. A source
   change during indexing prevents the generation from becoming ready.

Project storage is rooted at `CODE_SEARCH_STORAGE` (default
`~/.claude_code_search`). Provider-specific indexes for the same checkout can
coexist; the project configuration and verified manifest bind the provider,
model, and vector dimension used by that generation.

### Retrieval

`search_code` supports `auto`, `hybrid`, `keyword`, and `semantic` modes. The
default hybrid path runs vector and BM25 retrieval, applies query-signal-aware
candidate widening and exact-signal promotion, fuses ranks, applies content
type boosts and file diversification, then optionally applies graph PPR and a
reranker. PPR is off by default. The default reranker mode is Sonnet, but any
missing key, timeout, rate limit, or other reranker error preserves the hybrid
order and is reported in `_metadata.reranker`.

The fusion policy is content-mode specific: **code 65/35, docs 70/30, all 50/50 (vector/BM25)**. `search/retrieval.py` owns vector/BM25 candidate
retrieval and deterministic signal promotion; `search/fusion.py` owns RRF and
chunk-type boosts; `search/query_expansion.py` owns synonym expansion;
`search/result_models.py` owns response types; and `search/pipeline.py`
composes optional PPR and reranking. `search/searcher.py` remains the
orchestration and compatibility boundary.

Each result includes retrieval coordinates and structured metadata for
freshness, generation manifest, provider/model, reranker outcome, and any
stale-index advisory. Retrieval coordinates are context, not automatically a
minimal proof.

### Verifiable Evidence

`search_code_evidence` wraps the same production retrieval path. It does not
fork ranking behavior.

- The broad result span is labeled `retrieval_context`.
- The backend reads the exact indexed chunk and offers each nonblank source
  line as an `atomic_source_line` candidate.
- Every candidate carries an immutable `evidence_ref` and `observation_ref`
  bound to repository, source revision, index generation, path, and line.
- A `symbol_ref` is emitted only when the result contains a canonical qualified
  name. A short name is never guessed into a cross-engine identity.
- Identity is checked before and after search. If the generation changes or is
  stale, evidence emission fails closed while ordinary retrieval remains
  available.

Consumers should select emitted IDs. They should not manufacture, widen, or
rewrite source ranges. `code-graph` implements the same canonical reference
schema so a downstream host can join evidence without relying on prose.

## Quick Start

### Install the verified v0.3.6 release

The release contains a Python wheel, SHA-256 manifest, and GitHub artifact
attestation bundle:

```bash
REPO="redacted-org/code-search"
TAG="v0.3.6"
WHEEL="redacted_code_search-0.3.6-py3-none-any.whl"
BUNDLE="redacted_code_search-0.3.6-provenance.jsonl"

mkdir code-search-v0.3.6
cd code-search-v0.3.6
gh release download "$TAG" --repo "$REPO"

# Linux; on macOS use: shasum -a 256 -c SHA256SUMS
sha256sum --check SHA256SUMS

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
gh release verify-asset "$TAG" SHA256SUMS --repo "$REPO"
gh release verify-asset "$TAG" "$BUNDLE" --repo "$REPO"

python3 -m venv .venv
.venv/bin/python -m pip install "$WHEEL"
```

Python 3.12 or newer and an authenticated GitHub CLI are required. For a
development checkout:

```bash
git clone https://github.com/redacted-org/code-search.git
cd code-search
./scripts/install.sh
```

`./scripts/install.sh` installs that checkout into its own `.venv` and does
not update another clone.

### Configure an MCP Client

```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "env": {
        "VOYAGE_API_KEY": "<secret>"
      }
    }
  }
}
```

With `VOYAGE_API_KEY`, code-mode indexing defaults to Voyage
`voyage-4-large`. Without it, provider resolution defaults to the local
sentence-transformer path. Set `EMBEDDING_PROVIDER=jina` for the larger local
Jina code model. `RERANKER=sonnet` is the search default; set `RERANKER=off`
for a fully local query path or when no Anthropic key is configured.

### Index, Verify, and Search

```text
index_directory(directory_path="/absolute/repository")
get_indexing_progress()
get_index_status(project_path="/absolute/repository")
search_code(query="where is request authentication enforced?", search_mode="auto")
search_code_evidence(query="where is request authentication enforced?", search_mode="auto")
```

Do not treat a background job start as a ready index. Wait for a terminal
progress result and require `index_ready=true` with a matching live identity.

## MCP Tools

The current server registers 16 tools: 15 direct server operations plus the
additive evidence adapter.

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

Tool annotations classify read-only and destructive operations. Treat
`clear_index` and `delete_project` as irreversible; list and resolve the exact
project first.

## Configuration

The common settings are below. The complete, current table is in
[`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md).

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | Voyage when its key exists; otherwise local | `voyage`, `voyage-code-3`, `voyage-context`, `openai`, `jina`, or `local` |
| `EMBEDDING_DIMENSION` | `unset` | Required positive output-dimension contract for custom remote embedding models; built-in models derive it automatically |
| `CONTENT_MODE` | `code` | `code`, `docs`, or `all`; controls retrieval weights and boosts |
| `CONTEXTUAL_HEADERS` | `on` | Add file/type/name context before embedding |
| `QUERY_EXPANSION` | `on` | Expand BM25 terms with the selected synonym profile |
| `CODE_SYNONYM_PROFILE` | `corsair` | `corsair`, `generic`, or `off` |
| `CODE_SYNONYMS_PATH` | `unset` | Optional JSON overlay for the selected synonym profile |
| `CODE_SEARCH_LOG_LEVEL` | `INFO` | Minimum server log level |
| `RERANKER` | `sonnet` | `sonnet`, `listwise`, `cross-encoder`, or `off` |
| `QUANTIZATION` | `int8` | FAISS `int8`, `float32`, or `binary` storage |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Root for project indexes and local models |
| `CODE_SEARCH_LOG_QUERY_TEXT` | `off` | Opt in to raw query text in logs |
| `CODE_SEARCH_QUERY_HISTORY` | `metadata` | `off`, `metadata`, or `full`; metadata excludes raw query text |
| `CODE_SEARCH_QUERY_RETENTION_DAYS` | `30` | Query-history retention window in days |

These settings are process-static: they are read once when the MCP server starts. Restart the MCP server after changing them.

## Operating Guidance

1. Index a stable checkout and wait for `index_ready=true`.
2. Use keyword mode for known tokens and hybrid mode for concepts.
3. Use ordinary results for discovery; use `search_code_evidence` when an
   answer will make a code claim.
4. Keep evidence IDs with the answer and report the source revision/index
   generation when the decision is consequential.
5. Use `search_all_projects` only to discover the owning project. Re-run a
   project-bound evidence query before asserting anything.
6. Escalate relationship questions to `code-graph`. Composition is owned by
   the client or agent; `code-search` does not silently query graph state.

## Measured Evidence

- On a frozen, balanced public LocBench `n=80` file-localization endpoint,
  released `v0.3.5` measured Acc@1 `0.375`, Acc@3 `0.613`, Acc@10 `0.788`,
  and MRR@10 `0.503`. A Sourcegraph public endpoint measured
  `0.150/0.175/0.188/0.165` under that same endpoint. The paired Acc@1 sign
  test was `p=0.00053`. This is narrow endpoint evidence, not general platform
  superiority.
- The `v0.3.6` source-role prior and file diversification improved a separate
  paired replay from Acc@1 `0.3625` to `0.3875` and MRR@10 `0.49147` to
  `0.51608`, with 13 cases improved and none regressed. See
  [`docs/findings/2026-08-12-public-n80-source-role-prior.md`](docs/findings/2026-08-12-public-n80-source-role-prior.md).
- The released server indexed a pinned 39,222,246-line LLVM checkout into
  183,663 chunks in 609.3 seconds with 3.65 GB peak RSS. The persisted search
  index was 4.98 GB; five warm queries measured 3.77 seconds p50 and 3.84
  seconds p95. This proves operation on one very large repository, not a
  distributed organizational fleet or class-leading efficiency.
- On APFS, `v0.3.6` uses copy-on-write clones for mutable compatibility mirrors
  when available; other filesystems retain byte-copy behavior. See
  [`docs/findings/2026-08-12-copy-on-write-publication.md`](docs/findings/2026-08-12-copy-on-write-publication.md).

Historical provider and reranker experiments remain in `docs/findings/` and
[`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md). Compare only measurements
with compatible corpus, revision, configuration, and metric definitions.
MRR aggregates reciprocal rank across queries; it does not by itself determine top-result accuracy, a typical rank, or the probability that any one query succeeds. These are historical evaluation results, not current production guarantees.

The frozen balanced public LocBench n=80 endpoint establishes a narrow comparison: “This establishes narrow superiority for this frozen file-localization endpoint, not general platform superiority.”

## Comparison to Alternatives

- Public superiority is established only for the bounded endpoint above.
  Cursor, Augment, and Greptile remain ungraded.
- Sourcegraph has the broader search language, history UX, organization ACLs,
  and managed indexing fleet. This project is a focused MCP retrieval backend.
- Cross-project search does not federate scores and does not implement an
  organization authorization model.
- The index is not updated on every keystroke. Reindex or allow the freshness
  path to refresh it.
- Cloud embedding and reranking providers send query/source-derived content to
  their APIs. Select a local provider and disable reranking when that boundary
  is unacceptable.
- Search does not prove call relationships, runtime behavior, or variable-level
  taint. Use the appropriate graph, runtime, or CodeQL evidence.
- Very-large-repository storage and warm latency are functional but remain
  optimization targets.

## Troubleshooting

| Symptom | Check | Recovery |
|---|---|---|
| Index job started but queries are incomplete | `get_indexing_progress()` and `get_index_status()` | Wait for `index_ready=true` and a matching source/index identity. |
| Evidence is absent | Index identity or generation is stale | Reindex the unchanged checkout; evidence intentionally fails closed until identity is current. |
| Reranker is unavailable | `_metadata.reranker.reason` | Results retain the hybrid order; configure the provider/key or keep `RERANKER=off`. |
| Changed environment is ignored | Server was already running | Restart the MCP process; configuration is process-static. |

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests/unit/ -v
```

Architecture-sensitive changes should also run the relevant integration and
acceptance suites. A quality, latency, or storage claim is not complete merely
because tests pass; follow [`docs/SHIP_DISCIPLINE.md`](docs/SHIP_DISCIPLINE.md)
and [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md).

## License

GPL-3.0 (inherited from the upstream fork).
