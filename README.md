# code-search

Semantic code search MCP server for Claude Code. Hybrid BM25 + vector search with multiple embedding providers.

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/redacted-org/code-search.git
cd code-search
python -m venv .venv

# Linux/Mac
.venv/bin/pip install -r requirements.txt

# Windows
.venv\Scripts\pip install -r requirements.txt
```

### 2. Choose an embedding provider

| Provider | Quality (MRR) | Data leaves machine? | Cost | Setup |
|----------|:---:|:---:|:---:|---|
| **`voyage-context`** | **0.723** | Yes | ~$0.06/1M tokens | Set `VOYAGE_API_KEY` |
| `jina` | 0.582-0.660 | **No** | **Free** | Nothing — downloads model on first run |
| `local` | ~0.35-0.45 | No | Free | Nothing |

MRR values from eval on 102 queries across Nix, Rust, and TypeScript. See [Model Comparison](#model-comparison) for details.

### 3. Configure Claude Code

Add to your MCP settings (`.claude/settings.local.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "/path/to/code-search/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/code-search",
      "env": {
        "EMBEDDING_PROVIDER": "voyage-context",
        "VOYAGE_API_KEY": "pa-..."
      }
    }
  }
}
```

For local-only (no API key):
```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "/path/to/code-search/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/code-search",
      "env": {
        "EMBEDDING_PROVIDER": "jina"
      }
    }
  }
}
```

On Windows, use `.venv\Scripts\pythonw.exe` instead of `.venv/bin/python`.

### 4. Index a repo

From Claude Code:
```
mcp__code-search__index_directory(directory_path="/path/to/your/repo")
```

Or if using the [codebase-search-plugin](https://github.com/redacted-org/codebase-search-plugin):
```
/index-repo /path/to/your/repo
```

### 5. Search

```
mcp__code-search__search_code(query="authentication middleware")
```

## What It Does

Natural language queries against indexed codebases. "Find authentication logic" returns actual auth functions ranked by relevance, not string matches.

- **Hybrid search**: Weighted RRF fusion of FAISS vector similarity + FTS5 BM25 keyword matching
- **18+ file types**: Python, JS/TS/JSX/TSX, Go, Rust, Java, C/C++/C#, Nix, Svelte, Markdown, TOML, YAML, HCL
- **Contextual chunk headers**: Prepends file path + type + name before embedding for better retrieval
- **Incremental indexing**: Merkle tree change detection — only re-embeds changed files
- **Per-project config**: Embedding provider and model stored per project. Server switches automatically.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic + keyword hybrid search |
| `find_similar_code` | Find chunks similar to a given result |
| `index_directory` | Background indexing with progress polling |
| `get_indexing_progress` | Poll index job status |
| `clear_index` | Delete current project's index |
| `switch_project` | Change active project context |
| `list_projects` | Show all indexed projects |
| `get_index_status` | Index stats and model info |

## Model Comparison

Measured on 102 queries across 4 language sub-projects from a production Rust/Nix/TypeScript monorepo:

| Provider | Model | Nix (n=44) | Rust svc (n=20) | Rust lib (n=18) | TypeScript (n=20) | Avg |
|----------|-------|:---:|:---:|:---:|:---:|:---:|
| **`voyage`** | **voyage-4-large** | **0.826** | **0.917** | 0.861 | **0.683** | **0.828** |
| `voyage-context` | voyage-context-3 | 0.792 | 0.783 | **0.861** | 0.662 | 0.775 |
| `voyage` | voyage-4 | 0.803 | 0.892 | 0.861 | 0.650 | 0.806 |
| **`jina`** (enriched) | jina-code-0.5b | 0.638 | 0.742 | ~0.86 | 0.660 | — |
| `local` | all-MiniLM-L6-v2 | ~0.35 | ~0.45 | ~0.50 | ~0.40 | — |

### What the numbers mean

- **MRR** (Mean Reciprocal Rank): How often the correct file appears at position #1 in results. 0.826 means the right answer is typically the top result. 0.662 means it's typically at position #2.
- Values measured using golden test sets with verified expected files.

### Key findings

- **`voyage-4-large` wins 3 of 4 languages** (+0.053 weighted avg MRR over voyage-context-3). Uses MoE architecture via standard `/embeddings` endpoint.
- **`jina-code-0.5b` with enriched context** runs fully on-device with no data exfiltration. Good fallback for air-gapped environments.
- **Reranking hurts quality.** Cross-encoder reranking was tested and disabled (-30% MRR).

### Indexing time (3,000 chunks)

| Provider | Time | Notes |
|----------|------|-------|
| `voyage` (voyage-4-large) | ~5-10 min | API rate-limited |
| `jina` | ~50 min (CPU) | First run downloads ~1GB model. GPU: ~5 min |
| `local` | ~2-5 min | Small model, fast |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `voyage` (if `VOYAGE_API_KEY` set), else `local` | Embedding provider (voyage uses voyage-4-large) |
| `VOYAGE_API_KEY` | - | Voyage AI API key ([get one](https://dash.voyageai.com)) |
| `LOCAL_EMBEDDING_MODEL` | `jinaai/jina-code-embeddings-0.5b` | Model for `jina` provider |
| `JINA_TRUNCATE_DIM` | - | Matryoshka dim truncation for Jina (0.5b: 64-896) |
| `CONTENT_MODE` | `code` | `code` or `docs` — affects search weights |
| `CONTEXTUAL_HEADERS` | `on` | Prepend context headers to embeddings |
| `ENRICHED_CONTEXT` | `on` (jina/local), `off` (voyage-context) | Add sibling chunk names to headers (+9.6% MRR on Nix) |
| `QUERY_EXPANSION` | `on` | Expand query terms with domain synonyms |
| `QUANTIZATION` | `int8` | FAISS index type: `int8` (4x smaller), `float32`, `binary` (32x smaller) |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Storage directory for indexes |

## Architecture

```
code-search/
├── chunking/                       # Multi-language AST chunking (18+ file types)
│   └── languages/                  # Per-language chunkers
├── embeddings/
│   ├── embedder.py                 # Provider routing, contextual headers
│   ├── voyage_context_embedder.py  # Voyage contextualized API
│   ├── jina_code_embedder.py       # Jina local code embeddings
│   ├── openai_embedder.py          # OpenAI + Voyage standard API
│   └── sentence_transformer.py     # Local sentence-transformers
├── search/
│   ├── indexer.py                  # FAISS index + SQLite metadata
│   ├── searcher.py                 # Hybrid BM25+vector with RRF fusion
│   └── incremental_indexer.py      # Merkle tree change detection
├── mcp_server/
│   └── code_search_server.py       # MCP server, per-project switching
├── benchmarks/
│   ├── golden_nix.json             # Eval: 44 Nix queries
│   ├── golden_rust_assetman.json   # Eval: 20 Rust service queries
│   ├── golden_rust_libnet.json     # Eval: 18 Rust library queries
│   ├── golden_typescript_mithrandir.json  # Eval: 20 TypeScript queries
│   └── run_multilang_eval.py       # Cross-language A/B eval harness
└── tests/
    └── unit/                       # 34+ unit tests
```

## Development

```bash
# Run all tests
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Run the multi-language eval
.venv/Scripts/python.exe benchmarks/run_multilang_eval.py --lang rust-assetman

# Run full eval (all languages, ~2 hours with Jina CPU)
.venv/Scripts/python.exe benchmarks/run_multilang_eval.py
```

## License

GPL-3.0 (inherited from upstream fork)
