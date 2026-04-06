# CLAUDE.md

redacted fork of claude-context-local. Hybrid semantic + keyword code search MCP server.

## Key Commands

```bash
# Run tests
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Run MCP server
.venv/Scripts/python.exe -m mcp_server.server

# Index a repo (from MCP client)
# mcp__code-search__index_directory(directory_path="...")
```

## Architecture

- **Embedding providers**: Voyage AI (`voyage-context-3` default — +24% MRR over `voyage-code-3` in eval), `voyage-code-3` legacy, OpenAI, local sentence-transformers
- **Search**: Weighted RRF fusion of FAISS vector + FTS5 BM25. Content mode boosts (code: function/method 1.3x, docs: section 1.3x)
- **Chunking**: Tree-sitter AST for 12+ languages, regex-based for TOML/YAML/HCL/Markdown/Nix
- **Per-project config**: `project_info.json` stores embedding provider, model, content mode. Server creates correct embedder on project switch.
- **Contextual headers**: `# From <path> - <type> <name>` prepended before embedding (controlled by `CONTEXTUAL_HEADERS=on`)

## Testing

```bash
# All unit tests (34+)
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Specific test files
.venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py -v
.venv/Scripts/python.exe -m pytest tests/unit/test_voyage_context_embedder.py -v
.venv/Scripts/python.exe -m pytest tests/unit/test_hybrid_search.py -v
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `voyage-context` (if `VOYAGE_API_KEY` set) | Provider: `voyage-context` (recommended), `voyage`, `openai`, `jina` (local, code-optimized), `local` |
| `JINA_TRUNCATE_DIM` | - | Matryoshka dim truncation for jina provider (0.5b: 64-896, 1.5b: 128-1536) |
| `VOYAGE_API_KEY` | - | Voyage AI API key |
| `CONTENT_MODE` | `code` | `code` or `docs` - affects search weights and provider auto-select |
| `CONTEXTUAL_HEADERS` | `on` | Prepend context headers to embeddings |
| `QUERY_EXPANSION` | `on` | Expand query terms with domain synonyms |
| `RERANKER` | `off` | Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). **Disabled** — golden eval (2026-03-22) showed reranking degrades quality: HR 0.700→1.000, MRR 0.527→0.783 with reranker off. The cross-encoder reshuffles well-ranked RRF output into a worse order. Code preserved for future evaluation. |
| `QUANTIZATION` | `int8` | Index type: `int8` (QT_8bit trained, 4x smaller, default), `float32` (legacy), `binary` (32x smaller, opt-in for 100K+ chunks). **Note**: QT_8bit requires a training step (learns value range). Indexes built before 2026-04-05 used QT_8bit_direct which silently returned 0.0 similarities — must reindex. |
| `VOYAGE_BATCH_API` | `off` | `on` to use Batch API for full reindex (33% cheaper, 1000+ chunk threshold) |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Storage directory |

## Voyage AI Integration

code-search uses [Voyage AI](https://voyageai.com) embedding models to convert code chunks into vectors for semantic similarity search. When you search "where is the firewall config?", your query and every indexed chunk are compared as vectors — chunks whose vectors point in similar directions are returned as results.

**Model**: `voyage-context-3` (contextualized chunk embeddings). Chunks from the same source file are sent to Voyage together, so each chunk is embedded with awareness of its sibling chunks. A line like `allowedTCPPorts = [ 80 443 ]` gets embedded knowing it's inside a NixOS firewall block — not just a random list. This gave +24% MRR improvement over the code-specific `voyage-code-3` model in A/B eval (n=44 queries).

**Search pipeline**:
1. **Indexing**: Tree-sitter parses code into AST chunks → contextual headers prepended → chunks grouped by file → sent to Voyage `/v1/contextualizedembeddings` → vectors stored in FAISS (int8 quantized, 4x smaller)
2. **Search**: Query embedded via Voyage → FAISS cosine similarity → BM25 keyword search → Weighted RRF fusion (50/50) → chunk-type boosts → results

**Optimizations**:
- **int8 quantization** (default): QT_8bit (trained), 4x smaller FAISS indexes. Must use `QT_8bit`, NOT `QT_8bit_direct` (which silently returns 0.0 similarities on normalized vectors — discovered 2026-04-05).
- **Binary + rescore** (opt-in `QUANTIZATION=binary`): 32x smaller, hamming search → float rescore top-k. For 100K+ chunk repos.
- **Token pre-count**: Batches split by estimated token budget before API calls, preventing 400 errors
- **Batch API** (opt-in `VOYAGE_BATCH_API=on`): 33% cheaper async embedding for full reindexes (1000+ chunks)

**Multi-language eval results** (n=102, 4 language sub-projects):

| Provider | Model | Nix (n=44) | Rust svc (n=20) | Rust lib (n=18) | TypeScript (n=20) |
|----------|-------|-----------|----------------|----------------|------------------|
| `voyage-context` | voyage-context-3 | **0.723** | **0.783** | **0.861** | **0.677** |
| `voyage` | voyage-code-3 | 0.584 | 0.742 | 0.861 | 0.642 |
| `jina` (enriched) | jina-code-0.5b | **0.638** | 0.742 | ~0.86 | 0.660 |

Key: voyage-context-3 wins all languages (+24% Nix, +5.5% Rust/TS, 0% Rust lib). Jina with enriched context headers closes 40% of the Nix gap (0.582→0.638) and beats voyage-code-3 on Nix (+9.2%) and TypeScript (+2.8%). Enriched context is on by default for jina/local providers. Reranking (rerank-2.5) degrades quality (-30% MRR) — disabled.

## Protected Repo

PR required to merge to main. Use `--repo redacted-org/code-search` with `gh` CLI (bare `gh` resolves to upstream).
