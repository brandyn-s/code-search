# code-search

Semantic code search MCP server for Claude Code. Hybrid BM25 + vector search with per-project embedding models.

Originally forked from [FarhanAliRaza/claude-context-local](https://github.com/FarhanAliRaza/claude-context-local). Substantially rewritten - hybrid search, Voyage AI embeddings, per-project model switching, contextual chunk headers, config file chunkers, and multi-language AST chunking across 18+ file types.

## What It Does

Natural language queries against indexed codebases. "Find authentication logic" returns actual auth functions ranked by relevance, not string matches.

- **Hybrid search**: Weighted RRF fusion of FAISS vector similarity + FTS5 BM25 keyword matching
- **Per-project models**: `voyage-code-3` for code repos, `voyage-context-3` for documentation - auto-selected from `CONTENT_MODE`
- **Contextual chunk headers**: Prepends file path + type + name before embedding for better retrieval quality
- **18+ file types**: Python, JS/TS/JSX/TSX, Go, Rust, Java, C/C++/C#, Nix, Svelte, Markdown, TOML, YAML, HCL
- **Chunk overlap**: Config file chunkers (TOML/YAML/HCL) carry 100 chars from previous section to prevent boundary context loss
- **Incremental indexing**: Merkle tree change detection - only re-embeds changed files

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

## Embedding Providers

| Provider | Model | Dimensions | Use Case |
|----------|-------|:---:|----------|
| `voyage` (default) | `voyage-code-3` | 1024 | Code repos - trained on 300+ languages |
| `voyage-context` | `voyage-context-3` | 1024 | Documentation - sees full document context per chunk |
| `openai` | `text-embedding-3-small` | 1536 | Fallback |
| `local` | `all-MiniLM-L6-v2` | 384 | Fully offline, no API keys |

Auto-selection: `CONTENT_MODE=code` uses `voyage-code-3`, `CONTENT_MODE=docs` uses `voyage-context-3`. Override with `EMBEDDING_PROVIDER` env var. Per-project config stored in `project_info.json` - server switches automatically on project change.

## Setup

```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "C:/path/to/.venv/Scripts/pythonw.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/path/to/code-search",
      "env": {
        "EMBEDDING_PROVIDER": "voyage",
        "VOYAGE_API_KEY": "pa-...",
        "CONTENT_MODE": "code",
        "CONTEXTUAL_HEADERS": "on"
      }
    }
  }
}
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Run MCP server
.venv/Scripts/python.exe -m mcp_server.server
```

## Architecture

```
code-search/
├── chunking/                    # Multi-language AST chunking (18+ file types)
│   └── languages/               # Per-language chunkers (Python, Nix, TOML, YAML, HCL, etc.)
├── embeddings/
│   ├── embedder.py              # CodeEmbedder - provider routing, contextual headers
│   ├── openai_embedder.py       # OpenAI + Voyage standard API (voyage-code-3)
│   └── voyage_context_embedder.py  # Voyage contextualized API (voyage-context-3)
├── search/
│   ├── indexer.py               # FAISS index + SQLite metadata
│   ├── searcher.py              # Hybrid BM25+vector search with RRF fusion
│   ├── incremental_indexer.py   # Merkle tree change detection
│   └── reranker.py              # Cross-encoder reranker (optional, off by default)
├── mcp_server/
│   └── code_search_server.py    # MCP server with per-project model switching
└── tests/
    └── unit/                    # 34+ unit tests
```

## License

GPL-3.0 (inherited from upstream fork)
