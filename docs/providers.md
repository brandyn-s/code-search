# Providers: bring your own embedder and LLM

code-search talks to embedding and chat models through two wire protocols:
the OpenAI embeddings API (`POST {base}/embeddings`) and the OpenAI chat
completions API (`POST {base}/chat/completions`), plus native Anthropic and
Voyage clients. Anything that implements those two OpenAI surfaces works
without code changes.

| Setting | Embeddings | Reranking (`RERANKER=openai`) |
|---|---|---|
| Provider selector | `EMBEDDING_PROVIDER=openai` | `RERANKER=openai` |
| Endpoint root (include the version path) | `OPENAI_BASE_URL` | `RERANKER_LLM_BASE_URL` (defaults to `OPENAI_BASE_URL`) |
| Model | `EMBEDDING_MODEL` (+ `EMBEDDING_DIMENSION` for models not in the built-in table) | `RERANKER_LLM_MODEL` (required) |
| Key | `OPENAI_API_KEY` (required only for api.openai.com) | `RERANKER_LLM_API_KEY` (defaults to `OPENAI_API_KEY`) |
| Header style | `OPENAI_AUTH_HEADER=bearer` (default) or `api-key` | same variable |
| Timeouts | n/a (300 s client timeout) | `RERANKER_LLM_TIMEOUT_S` per call, `SONNET_RERANKER_TIMEOUT` overall |

Native engines: `EMBEDDING_PROVIDER=voyage*` with `VOYAGE_API_KEY`;
`RERANKER=sonnet` or `listwise` with `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`.
Local engines: `EMBEDDING_PROVIDER=local|jina|gemma` and `RERANKER=cross-encoder`
with the `[local]` extra; nothing leaves the machine.

## Endpoints

Status legend: **tested** means exercised in this repository's test suite
against a fake server that asserts the exact URL, headers, and body;
**expected** means the vendor documents an OpenAI-compatible surface and the
same request shape applies, but nobody has run code-search against it yet.
Report results either way in an issue.

| Endpoint | `*_BASE_URL` | Key | Header | Embeddings | Reranking | Status |
|---|---|---|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` (default) | `OPENAI_API_KEY` required | bearer | `text-embedding-3-small`, `-large` | any chat model | tested |
| Ollama | `http://localhost:11434/v1` | none | – | `nomic-embed-text` (768), `mxbai-embed-large` (1024) | `qwen2.5-coder:7b`, `llama3.1`, … | tested (fake); local server expected |
| vLLM | `http://localhost:8000/v1` | optional (`--api-key`) | bearer | any served embedding model | any served chat model | expected |
| LM Studio | `http://localhost:1234/v1` | none | – | any loaded embedding model | any loaded chat model | expected |
| OpenRouter | `https://openrouter.ai/api/v1` | required | bearer | `openai/text-embedding-3-small` | any routed model | tested (fake) |
| Azure OpenAI (v1 surface) | `https://<resource>.openai.azure.com/openai/v1` | required | `OPENAI_AUTH_HEADER=api-key` for API keys; `bearer` for Entra ID tokens | deployment name as model | deployment name as model | tested (fake) |
| Google Gemini (OpenAI-compatible) | `https://generativelanguage.googleapis.com/v1beta/openai` | required (Gemini API key) | bearer | `gemini-embedding-001` (3072; or set `EMBEDDING_DIMENSION` for a truncated size) | `gemini-2.5-flash`, … | expected |
| Amazon Bedrock | via a gateway: LiteLLM proxy (`http://localhost:4000/v1`) or the Bedrock Access Gateway | gateway key | bearer | `amazon.titan-embed-text-v2:0` (1024) | `anthropic.claude-*`, `amazon.nova-*` | expected |
| Any LiteLLM / Portkey / Kong AI gateway | gateway URL | gateway key | bearer | whatever it routes | whatever it routes | expected |

Notes:

- Set `EMBEDDING_DIMENSION` whenever the model is not in the built-in table;
  the server refuses to start indexing without it, because a wrong size would
  corrupt the FAISS index silently.
- Changing provider, model, or dimension changes the index identity. Reindex
  after switching; queries against an index built with a different embedder
  fail closed.
- `RERANKER=auto` only ever resolves to `sonnet` or `off`. Selecting the
  OpenAI-compatible reranker is always explicit.
- Small local chat models are slow judges. Start with
  `SONNET_RERANKER_POOL_SIZE=5` and raise `SONNET_RERANKER_TIMEOUT` if
  `_metadata.reranker.reason` reports `timeout`.
- Azure: the `/openai/v1` surface accepts `api-key` for key-based auth and
  `Authorization: Bearer` for Entra ID tokens. This project sends exactly one
  of the two based on `OPENAI_AUTH_HEADER`; the header choice is exercised by
  tests, the Azure endpoint itself is not.

## What each provider receives

| Data | Embedding provider | Reranker |
|---|---|---|
| Source chunk text (with contextual headers) | yes, at index time and for `find_similar_code` | yes, the top hybrid candidates for each query (up to `SONNET_RERANKER_POOL_SIZE`, default all 15) |
| Query text | yes, per query | yes, per query |
| File paths | inside contextual headers | inside the judge prompt |
| Repository name, revision, index metadata | no | no |

With `EMBEDDING_PROVIDER=local` (or `jina`/`gemma`) and `RERANKER=off` or
`cross-encoder`, no source or query text leaves the machine. Query text is
never logged unless `CODE_SEARCH_LOG_QUERY_TEXT=on`.
