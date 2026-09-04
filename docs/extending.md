# Extending code-search

Four things people add most often, each with the file to copy and the test to
copy. Every recipe ends the same way: run `tests/unit`, update
`docs/ENV_REFERENCE.md` if you read a new variable (the inventory test will
tell you), and add a line under `Unreleased` in `CHANGELOG.md`.

## 1. A language chunker

Chunkers turn one file into semantic units (functions, classes, sections).

1. Copy the smallest existing chunker, `chunking/languages/go_chunker.py`
   (48 lines), to `chunking/languages/<lang>_chunker.py`. Subclass
   `LanguageChunker` from `chunking/base_chunker.py`; implement the node-type
   mapping and the tree-sitter language import.
2. Register the file extensions in `LANGUAGE_MAP` in
   `chunking/languages/__init__.py`. `MultiLanguageChunker` derives
   `SUPPORTED_EXTENSIONS` from that map, so nothing else needs to change.
3. Add the tree-sitter grammar to `pyproject.toml` dependencies and
   `requirements.txt` (pin like the others).
4. Copy `tests/unit/test_toml_chunker.py` and assert on chunk names, kinds, and
   line ranges for a small fixture string.
5. Update the language count in README ("17 language modes across 21
   registered file extensions") and `CLAUDE.md`; the documentation-contract
   test pins those numbers.

## 2. An embedding provider

Providers are factories registered by name; resolution happens in
`embeddings/embedder.py`.

1. Implement the model class. Cloud providers follow
   `embeddings/openai_embedder.py` (HTTP client, batching, dimension
   contract); local providers follow `embeddings/sentence_transformer.py` and
   must call `require_local_extra()` from `embeddings/local_extra.py` before
   importing anything heavy.
2. Register a factory in `embeddings/embedder.py`:

   ```python
   @register_provider("myprovider")
   def _factory_myprovider(model_name: str, cache_dir: str, device: str) -> Any:
       from embeddings.myprovider import MyProviderEmbedder
       return MyProviderEmbedder(model_name=model_name or env_get("MYPROVIDER_MODEL", "default-model"))
   ```

   Add the model to `_KNOWN_MODEL_PROVIDERS` and its dimension to the
   dimension table next to it so `EMBEDDING_DIMENSION` is derived automatically.
3. Read configuration only through `search.env.env_get`; document every new
   variable in `docs/ENV_REFERENCE.md`. Keys are auto-detected only when
   the environment names a credential the user clearly intended for us
   (`VOYAGE_API_KEY`); anything else stays an explicit `EMBEDDING_PROVIDER`
   opt-in so source code is never sent to a service by accident.
4. Copy `tests/unit/test_openai_embedder.py` for the client and
   `tests/unit/test_embedder_registry.py` for registration and resolution.

## 3. A reranker

Rerankers are functions registered in `search/reranker_registry.py`; the
pipeline looks them up by the `RERANKER` value. Use `search/openai_reranker.py`
as the template for a new LLM engine: it reuses the shared judge prompt and
score parser in `search/llm_judge.py`, so a new transport needs only the HTTP
call and error classification. `tests/unit/test_openai_reranker.py` shows how
to fake the endpoint.

1. Add a function with the shared signature next to the existing ones:

   ```python
   @register_reranker("my-mode")
   def rerank_my_mode(searcher, query, *, k, config, candidates, metadata_lookup):
       ...
       searcher.last_reranker_metadata = {"applied": True, "reason": "ok", "latency_ms": ms}
       return ranked[:k]
   ```

   Contract: set `searcher.last_reranker_metadata` with at least `applied`,
   `reason`, `latency_ms`; on any failure preserve the hybrid order and
   report the reason instead of raising. Use `_not_invoked(searcher, reason)`
   for the skip cases.
2. `RERANKER_MODES` in `search/config.py` derives from the registry, so the
   new name is immediately a valid `RERANKER` value. Document it in the
   `RERANKER` row of `docs/ENV_REFERENCE.md`.
3. Copy `tests/unit/test_searcher_listwise_dispatch.py` for dispatch and
   `tests/unit/test_reranker_metadata_propagation.py` for the metadata
   contract. Add the mode to the `reason` vocabulary section of the reference.

## 4. An MCP tool

Tools are methods on `CodeSearchServer` that return a JSON string; the MCP
layer registers them from `mcp_server/strings.yaml`.

1. Add the method to `mcp_server/code_search_server.py` (or, for anything
   that owns state, to a module under `search/` that the server delegates
   to, as `search/index_jobs.py` does). Return `json.dumps(...)`; never
   raise across the tool boundary.
2. Add the tool's description under `tools:` in `mcp_server/strings.yaml`
   and its annotations (read-only, destructive, idempotent) where
   `code_search_mcp.py` builds them.
3. Add a row to the README "MCP Tools" table and update the tool count in
   the README and `docs/ARCHITECTURE.md` State of Record; `tests/unit/
   test_mcp_tool_descriptions.py` and the documentation-contract tests will
   fail until the table, the count, and `strings.yaml` agree.
4. Copy `tests/unit/test_evidence_search_tool.py` as the model for a tool
   test: build a `CodeSearchServer` against a temp storage dir, call the
   method, and assert on the parsed JSON.

## Where not to add things

- Do not read `os.environ` outside `search/env.py`; the inventory test fails.
- Do not add a second place that decides the provider or reranker; both have
  one registry each.
- Do not put research harnesses or eval outputs in the tree; use
  `bench/eval/public/` and keep results out of Git.
