# Environment Variable Reference

Every setting code-search reads from the environment, with its default and what
it does. `README.md` and `CLAUDE.md` list the subset most deployments touch.

These settings are process-static: they are read once when the MCP server starts. Restart the MCP server after changing them.

The server prints one line at startup naming the resolved embedding provider and
reranker mode, for example
`code-search: embeddings=local(sentence-transformers/all-MiniLM-L6-v2) reranker=off`.

## Providers and credentials

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `voyage` (if `VOYAGE_API_KEY` set), else `local` | Embedding backend: `voyage` (voyage-4-large), `voyage-code-3`, `voyage-context` (contextualized embeddings), `openai`, `jina` (local code model), `gemma` (local), or `local` (small sentence-transformers model). Cloud providers receive chunk text and queries. |
| `EMBEDDING_MODEL` | per provider | Model for remote providers: `voyage-4-large`, `voyage-code-3`, `voyage-context-3`, `text-embedding-3-small`, or any model your `OPENAI_BASE_URL` server offers (set `EMBEDDING_DIMENSION` for models not in the built-in table). |
| `LOCAL_EMBEDDING_MODEL` | per provider | Model for local providers: `sentence-transformers/all-MiniLM-L6-v2` (`local`), `jinaai/jina-code-embeddings-0.5b` (`jina`), `google/embeddinggemma-300m` (`gemma`). Downloaded on first index and cached under `CODE_SEARCH_STORAGE`. |
| `EMBEDDING_DIMENSION` | `unset` | Required positive output-dimension contract for custom remote embedding models; known built-in models derive this automatically. Stored project metadata reuses the value only when its provider and model both match. |
| `JINA_TRUNCATE_DIM` | unset | Matryoshka truncation for Jina models (0.5b: 64, 128, 256, 512, 896). |
| `VOYAGE_API_KEY` | unset | Enables the Voyage providers. |
| `VOYAGE_INPUT_TYPE` | `off` | `on` sends Voyage `input_type=document`/`query` hints. Always on for `voyage-context`. Recorded in the index identity, so changing it requires a full reindex. |
| `VOYAGE_BATCH_API` | `off` | `on` uses Voyage's asynchronous batch endpoint for full reindexes, which is cheaper but slower. |
| `VOYAGE_BATCH_THRESHOLD` | `1000` | Minimum chunk count before the batch endpoint is used. |
| `OPENAI_API_KEY` | unset | Key for `EMBEDDING_PROVIDER=openai`. Required when `OPENAI_BASE_URL` is api.openai.com; optional for self-hosted servers. Never auto-selects the provider. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Endpoint root for the `openai` provider, including the version path. Point it at any OpenAI-compatible embeddings server: Ollama (`http://localhost:11434/v1`), vLLM, LM Studio, Azure OpenAI, OpenRouter, or a gateway in front of Gemini or Bedrock. Custom models need `EMBEDDING_DIMENSION`. See `docs/providers.md`. |
| `OPENAI_AUTH_HEADER` | `bearer` | How the key is sent: `bearer` (`Authorization: Bearer`) or `api-key` (Azure OpenAI API keys). |
| `ANTHROPIC_API_KEY` | unset | Enables LLM reranking and query rewriting. With `RERANKER=auto`, its presence selects Sonnet reranking. |

## Indexing

| Variable | Default | Purpose |
|----------|---------|---------|
| `CODE_SEARCH_STORAGE` | `~/.claude_code_search` | Root directory for project indexes, manifests, query history, and cached local models. |
| `CODE_SEARCH_ALLOWED_ROOTS` | unset | Comma-separated directories that `index_directory` may index. Unset allows any path outside a small system denylist. |
| `CONTENT_MODE` | `code` | `code`, `docs`, or `all`. Sets fusion weights (code 65/35, docs 70/30, all 50/50 (vector/BM25)) and, when no provider is explicit, picks `voyage-context` for `docs`. |
| `CONTEXTUAL_HEADERS` | `on` | Prepend a `# From <path>` header with file, type, and name before embedding each chunk. |
| `ENRICHED_CONTEXT` | `on` (`off` for `voyage-context`) | Add sibling chunk names from the same file to the header for providers that embed chunks independently. |
| `LLM_CONTEXT_PATH` | unset | Path to a JSON map `{chunk_id: paragraph}`; a matching paragraph replaces the simple header at embedding time. Requires reindexing. |
| `QUANTIZATION` | `int8` | FAISS storage: `int8` (trained scalar quantizer, 4x smaller), `float32`, or `binary` (32x smaller with float rescoring; for very large indexes). |
| `CODE_SEARCH_NIX_OPTION_CHUNKING` | `off` | `on` chunks files under `nix/modules/` per option declaration instead of per module. |
| `CODE_SEARCH_DISABLE_AUTO_REINDEX` | unset | `1`/`true`/`yes`/`on` stops searches from triggering incremental reindexes; refresh with `index_directory` instead. Useful for very large projects. |
| `CODE_SEARCH_NONBLOCKING_SEARCH` | unset | `1`/`true`/`yes`/`on` makes `search_code` return last-good results immediately and run a needed reindex in a background thread; results carry `_metadata.freshness="stale_reindex_in_progress"` until it finishes. Default blocks until the index is fresh. |
| `CODE_SEARCH_STARTUP_AUDIT` | `1` | `0` skips the background integrity audit of existing indexes at startup. |

## Retrieval and fusion

| Variable | Default | Purpose |
|----------|---------|---------|
| `SEARCH_MODE` | `hybrid` | Fallback mode when a `search_code` call does not specify one: `auto`, `hybrid`, `keyword`, or `semantic`. |
| `FUSION_K` | `20` | Reciprocal-rank-fusion smoothing constant. |
| `VECTOR_WEIGHT` | `0.0` | Override for the vector arm weight; `0.0` uses the content-mode default. |
| `BM25_WEIGHT` | `0.0` | Override for the BM25 arm weight; `0.0` uses the content-mode default. |
| `CHUNK_TYPE_BOOST_OVERRIDE` | unset | JSON object mapping chunk type to a score multiplier, layered over the built-in boosts, e.g. `{"function": 1.2}`. |
| `QUERY_EXPANSION` | `on` | Expand BM25 query terms with the selected synonym profile. |
| `CODE_SYNONYM_PROFILE` | `generic` | Built-in synonym profile: `generic` or `off`. The active profile is reported in search metadata. |
| `CODE_SYNONYMS_PATH` | `unset` | JSON object overlaying the selected profile: a list extends or replaces a key, `null` removes it. A load failure logs a warning and uses the profile unchanged. |
| `BM25_REWRITE` | `off` | `on` asks an LLM (`BM25_REWRITE_MODEL`) to rewrite the BM25 query. Needs `ANTHROPIC_API_KEY`. |
| `SHORT_QUERY_REWRITE` | `off` | `on` expands very short queries with an LLM before retrieval. Needs `ANTHROPIC_API_KEY`. |
| `BM25_REWRITE_MODEL` | `claude-haiku-4-5-20251001` | Model for the two rewrite features above. |
| `AGENTIC_SEARCH` | `off` | `on` adds an LLM ordering pass over formatted results using `LLM_MODEL`. Experimental. |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Model for `AGENTIC_SEARCH`. |
| `CODE_SEARCH_PPR_ENABLED` | `off` | `on` applies personalized PageRank from a compatible code-graph sidecar when one is present. |
| `CODE_SEARCH_PPR_ALPHA` | `0.5` | Blend weight for the PageRank signal. |

## Reranking

| Variable | Default | Purpose |
|----------|---------|---------|
| `RERANKER` | `auto` | `auto` resolves to `sonnet` when `ANTHROPIC_API_KEY` is set and `off` otherwise (it never selects `openai`). `sonnet` scores the top hybrid candidates with one Anthropic call each. `openai` does the same through any OpenAI-compatible chat endpoint (`RERANKER_LLM_*`). `listwise` ranks them in a single Anthropic call. `cross-encoder` uses a local MiniLM cross-encoder. `off` returns fused order. Any reranker failure keeps the hybrid order and records the reason in `_metadata.reranker`. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model for `RERANKER=cross-encoder`. Requires the `[local]` extra. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model for both the pointwise (`sonnet`) and `listwise` rerankers. |
| `RERANKER_LLM_MODEL` | unset | Model name for `RERANKER=openai` (required; the server warns at startup and keeps hybrid order when unset). |
| `RERANKER_LLM_BASE_URL` | `OPENAI_BASE_URL`, else `https://api.openai.com/v1` | Chat-completions endpoint root for `RERANKER=openai`, including the version path (Ollama `http://localhost:11434/v1`, vLLM, LM Studio, Azure, OpenRouter, gateways). |
| `RERANKER_LLM_API_KEY` | `OPENAI_API_KEY` | Key for the reranker endpoint; required only for api.openai.com. Sent per `OPENAI_AUTH_HEADER`. |
| `RERANKER_LLM_TIMEOUT_S` | `12.0` | Per-request timeout for `RERANKER=openai`, capped by `SONNET_RERANKER_TIMEOUT`. |
| `SONNET_RERANKER_TIMEOUT` | `8.0` | Overall deadline in seconds for the pointwise rerank of one query (`sonnet` and `openai`). |
| `SONNET_LISTWISE_TIMEOUT` | `12.0` | Deadline in seconds for the single listwise call. |
| `ANTHROPIC_PER_CALL_TIMEOUT_S` | `12.0` | Per-request timeout inside the overall deadline. |
| `ANTHROPIC_MAX_RETRIES` | `1` | Anthropic SDK retry count for reranker calls. `0` disables retries. |
| `ANTHROPIC_CONCURRENCY_LIMIT` | unset | Caps concurrent pointwise scoring calls. Unset scores all candidates in parallel. |
| `SONNET_RERANKER_POOL_SIZE` | `0` | Score only the top N hybrid candidates; the rest keep their order. `0` scores all. Applies to `sonnet` and `openai`. |
| `SONNET_RERANKER_SKIP_THRESHOLD` | unset | Skip reranking when the top hybrid `similarity_score` is at least this value; reason `skipped_high_confidence`. |
| `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD` | `6` | If no candidate scores at least this (0-10 scale), the reranker is treated as uncertain and hybrid order is kept; reason `hybrid_prior_fallback`. `0` disables. |
| `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES` | unset | JSON object mapping path prefix to a higher threshold, e.g. `{"billing/": 11}`. The strictest matching override applies; overrides can only tighten. |
| `SONNET_RERANKER_PROMPT_CLAUSE_OVERRIDES` | unset | JSON object mapping path prefix to extra judging guidance injected only when scoring candidates under that prefix. |
| `SONNET_RERANKER_LOG_PER_CANDIDATE_SCORE` | unset | Any value logs one line per scored candidate. Diagnostic only. |
| `SONNET_RERANKER_LOG_OVERRIDE_TRIGGERS` | unset | Any value logs a `[PATH_OVERRIDE_TRIGGER]` line whenever a per-path hybrid-prior override raises the effective threshold. Diagnostic only. |

## Logging and privacy

| Variable | Default | Purpose |
|----------|---------|---------|
| `CODE_SEARCH_LOG_LEVEL` | `INFO` | Minimum log level, using Python logging level names. |
| `CODE_SEARCH_LOG_QUERY_TEXT` | `off` | `on` includes raw query text in logs. Leave off unless the log sink is approved for query data. |
| `CODE_SEARCH_QUERY_HISTORY` | `metadata` | Query-history mode: `off`, `metadata`, or `full`. `metadata` stores timing and outcome without query text; `full` stores query text and must be treated as sensitive. |
| `CODE_SEARCH_QUERY_RETENTION_DAYS` | `30` | Query-history retention window in days. |

## Reranker `reason` vocabulary

`_metadata.reranker.reason` is a stable string:

| Reason | applied | Meaning |
|--------|---------|---------|
| `ok` | true | Rerank applied |
| `empty_input` | false | No candidates passed to the reranker |
| `api_key_missing` | false | `ANTHROPIC_API_KEY` not set (`sonnet`, `listwise`) or no key for an api.openai.com reranker endpoint (`openai`) |
| `model_not_configured` | false | `RERANKER=openai` without `RERANKER_LLM_MODEL` |
| `package_not_installed` | false | `anthropic` SDK not importable |
| `timeout` | false | Deadline exceeded |
| `rate_limit` | false | Rate-limit responses dominated |
| `too_many_failures` | false | More than 30% of per-candidate calls failed |
| `hybrid_prior_fallback` | false | Max score below `SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD` |
| `skipped_high_confidence` | false | Top hybrid score met `SONNET_RERANKER_SKIP_THRESHOLD` |
| `disabled_by_env` | false | `RERANKER=off` (including `auto` without a key) |
| `not_invoked_keyword_mode` | false | `search_mode=keyword` |
| `not_invoked_semantic_mode` | false | `search_mode=semantic` |
| `not_invoked_cross_encoder_mode` | false | `RERANKER=cross-encoder` path |
| `not_invoked_no_candidates` | false | Retrieval returned nothing |
| `not_invoked_insufficient_candidates` | false | Fewer candidates than `k` |
| `async_context` | false | Called from an async context |
| `unexpected_error` | false | Catch-all; details are logged |

## Provider comparison

Measured on four private sub-projects (Nix, Rust service, Rust library,
TypeScript; 102 queries) at the time each provider was added. Treat the numbers
as relative guidance, not as a guarantee for your corpus.

| Provider | Model | Mean MRR |
|----------|-------|----------|
| `voyage` | voyage-4-large | 0.828 |
| `voyage` | voyage-4 | 0.806 |
| `voyage` | voyage-4-lite | 0.798 |
| `voyage-context` | voyage-context-3 | 0.775 |
| `jina` | jina-code-embeddings-0.5b | about 0.72 |

Voyage `rerank-2.5` was evaluated as a reranker and reduced MRR by roughly 30%,
so it is not offered. The local `cross-encoder` mode also regressed quality in
an early comparison and is kept only for offline experiments.
