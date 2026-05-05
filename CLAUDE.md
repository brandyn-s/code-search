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

- **Embedding providers**: Voyage AI (`voyage-4-large` default — +0.053 weighted avg MRR over `voyage-context-3` across 4 langs), `voyage-context-3` legacy, OpenAI, local sentence-transformers
- **Search**: Weighted RRF fusion of FAISS vector + FTS5 BM25. Content mode boosts (code: function/method 1.3x, docs: section 1.3x)
- **Chunking**: Tree-sitter AST for 12+ languages, regex-based for TOML/YAML/HCL/Markdown/Nix. Post-processing merge step (cAST-style) greedily combines small adjacent chunks to 1500 NWS char budget, capturing gap code (imports, constants) between semantic units.
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
| `EMBEDDING_PROVIDER` | `voyage` (if `VOYAGE_API_KEY` set) | Provider: `voyage` (recommended, uses voyage-4-large), `voyage-context` (legacy contextualized), `openai`, `jina` (local, code-optimized), `local` |
| `JINA_TRUNCATE_DIM` | - | Matryoshka dim truncation for jina provider (0.5b: 64-896, 1.5b: 128-1536) |
| `VOYAGE_API_KEY` | - | Voyage AI API key |
| `CONTENT_MODE` | `code` | `code` or `docs` - affects search weights and provider auto-select |
| `CONTEXTUAL_HEADERS` | `on` | Prepend context headers to embeddings |
| `QUERY_EXPANSION` | `on` | Expand query terms with domain synonyms |
| `RERANKER` | `sonnet` | Reranker mode. `sonnet` (default, 2026-05-03+): Sonnet 4.6 query-time reranker, validated +0.087 MRR / +0.137 HR@1 on n=183 multi-target real_session (D4b, PR #93+). Reranks top-15 hybrid candidates via Anthropic API. **Always-on graceful fallback**: missing `ANTHROPIC_API_KEY`, timeout (>8s), or any error → silently returns hybrid order. Cost ~$0.005/query, latency +1-2s. `cross-encoder`: legacy MiniLM cross-encoder (off-by-default since 2026-03-22 A/B showed quality regression). `off`: skip reranking, return RRF+boost order. |
| `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD` | `6` | Hybrid-prior fallback threshold (added 2026-05-03, tuned 2026-05-04). When the max Sonnet score across the candidate pool is below this value, the reranker is uncertain — score ties get arbitrary tie-breaking that favors keyword-dense chunks over canonical files. Falls back to hybrid order in that case. Set to `0` to disable. Default tuned 7→6 (PR #96): n=183 multi-target eval showed threshold=6 wins MRR (0.838 vs 0.830 at 7) and HR@1 (0.803 vs 0.787 at 7). |
| `QUANTIZATION` | `int8` | Index type: `int8` (QT_8bit trained, 4x smaller, default), `float32` (legacy), `binary` (32x smaller, opt-in for 100K+ chunks). **Note**: QT_8bit requires a training step (learns value range). Indexes built before 2026-04-05 used QT_8bit_direct which silently returned 0.0 similarities — must reindex. |
| `VOYAGE_BATCH_API` | `off` | `on` to use Batch API for full reindex (33% cheaper, 1000+ chunk threshold) |
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Storage directory |
| `CODE_SEARCH_DISABLE_AUTO_REINDEX` | unset | Set to `1`/`true`/`yes`/`on` to make `auto_reindex_if_needed` a no-op. Useful for large projects (10K+ chunks, 2000+ files) where `detect_changes` is multi-minute. Refresh on demand via `index_directory(incremental=false)` instead. Logs `[REINDEX_PROGRESS] auto_reindex_if_needed: SKIPPED` to `~/.claude/logs/code-search-mcp.log` when active. |

## Voyage AI Integration

code-search uses [Voyage AI](https://voyageai.com) embedding models to convert code chunks into vectors for semantic similarity search. When you search "where is the firewall config?", your query and every indexed chunk are compared as vectors — chunks whose vectors point in similar directions are returned as results.

**Model**: `voyage-4-large` (MoE architecture, SOTA retrieval). Uses the standard `/v1/embeddings` endpoint. Eval (n=102 queries, 4 languages) showed +0.053 weighted avg MRR over `voyage-context-3`. Wins on Nix (+0.034), Rust service (+0.134), TypeScript (+0.021), ties on Rust lib. The `voyage-context` provider (contextualized embeddings via `/v1/contextualizedembeddings`) is preserved as legacy.

**Search pipeline**:
1. **Indexing**: Tree-sitter parses code into AST chunks → contextual headers prepended → sent to Voyage `/v1/embeddings` → vectors stored in FAISS (int8 quantized, 4x smaller)
2. **Search**: Query embedded via Voyage → FAISS cosine similarity → BM25 keyword search → Weighted RRF fusion (50/50) → chunk-type boosts → results

**Optimizations**:
- **int8 quantization** (default): QT_8bit (trained), 4x smaller FAISS indexes. Must use `QT_8bit`, NOT `QT_8bit_direct` (which silently returns 0.0 similarities on normalized vectors — discovered 2026-04-05).
- **Binary + rescore** (opt-in `QUANTIZATION=binary`): 32x smaller, hamming search → float rescore top-k. For 100K+ chunk repos.
- **Token pre-count**: Batches split by estimated token budget before API calls, preventing 400 errors
- **Batch API** (opt-in `VOYAGE_BATCH_API=on`): 33% cheaper async embedding for full reindexes (1000+ chunks)

**Multi-language eval results** (n=102, 4 language sub-projects, MRR):

| Provider | Model | Nix (n=44) | Rust svc (n=20) | Rust lib (n=18) | TypeScript (n=20) | Avg |
|----------|-------|-----------|----------------|----------------|------------------|-----|
| `voyage` | **voyage-4-large** | **0.826** | **0.917** | 0.861 | **0.683** | **0.828** |
| `voyage-context` | voyage-context-3 | 0.792 | 0.783 | **0.861** | 0.662 | 0.775 |
| `voyage` | voyage-4 | 0.803 | 0.892 | 0.861 | 0.650 | 0.806 |
| `voyage` | voyage-4-lite | 0.785 | 0.892 | 0.880 | 0.650 | 0.798 |
| `jina` (enriched) | jina-code-0.5b | 0.638 | 0.742 | ~0.86 | 0.660 | — |

Key: voyage-4-large wins 3 of 4 languages, +0.053 weighted avg MRR over voyage-context-3. Uses standard `/embeddings` endpoint (simpler than contextualized). Reranking (rerank-2.5) degrades quality (-30% MRR) — disabled.

## Protected Repo

PR required to merge to main. Use `--repo redacted-org/code-search` with `gh` CLI (bare `gh` resolves to upstream).
