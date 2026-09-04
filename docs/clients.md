# Using code-search from MCP clients

code-search is a stdio MCP server. Any client that can launch a command can use
it. The examples below start the published package with `uvx`; substitute
`pipx run code-search-mcp`, or the absolute path to a venv's `code-search-mcp`
script, if you prefer.

Both API keys are optional. Without `VOYAGE_API_KEY` the server embeds with a
local model, which requires the `[local]` extra (`uvx --from
'code-search-mcp[local]' code-search-mcp`; adds PyTorch, about 1 GB). Without
`ANTHROPIC_API_KEY` it skips LLM reranking. The server prints one line at
startup saying which mode it resolved to.

## Claude Code

```bash
# cloud embeddings
claude mcp add code-search --scope user \
  -e VOYAGE_API_KEY=... -e ANTHROPIC_API_KEY=... -- uvx code-search-mcp
# fully offline
claude mcp add code-search --scope user -- uvx --from 'code-search-mcp[local]' code-search-mcp
```

Or install the [code-intelligence plugin](https://github.com/brandyn-s/codebase-search-plugin),
which bundles code-search with code-graph and adds `/index-repo` and
`/code-intel` skills.

## Claude Desktop

`claude_desktop_config.json`:

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

## Cursor

`.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "code-search": {
      "command": "uvx",
      "args": ["code-search-mcp"],
      "env": { "VOYAGE_API_KEY": "optional" }
    }
  }
}
```

## Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.code-search]
command = "uvx"
args = ["code-search-mcp"]
# env = { VOYAGE_API_KEY = "optional", ANTHROPIC_API_KEY = "optional" }
```

## Windsurf

`~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "uvx",
      "args": ["code-search-mcp"]
    }
  }
}
```

## Generic stdio configuration

```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "uvx",
      "args": ["code-search-mcp"],
      "env": {}
    }
  }
}
```

## Self-hosted embeddings (Ollama / vLLM / LM Studio)

The `openai` provider speaks the OpenAI embeddings API, so any server that
implements it works. Set the endpoint root (including `/v1`), the model name
the server exposes, and the vector size for models not in the built-in table.
No API key is needed for local servers.

```bash
ollama pull nomic-embed-text
claude mcp add code-search --scope user \
  -e EMBEDDING_PROVIDER=openai \
  -e OPENAI_BASE_URL=http://localhost:11434/v1 \
  -e EMBEDDING_MODEL=nomic-embed-text \
  -e EMBEDDING_DIMENSION=768 \
  -- uvx code-search-mcp
```

To rerank with a local chat model as well:

```bash
ollama pull qwen2.5-coder:7b
claude mcp add code-search --scope user \
  -e EMBEDDING_PROVIDER=openai -e OPENAI_BASE_URL=http://localhost:11434/v1 \
  -e EMBEDDING_MODEL=nomic-embed-text -e EMBEDDING_DIMENSION=768 \
  -e RERANKER=openai -e RERANKER_LLM_MODEL=qwen2.5-coder:7b \
  -- uvx code-search-mcp
```

`RERANKER_LLM_BASE_URL` defaults to `OPENAI_BASE_URL`, so one endpoint serves
both. Small local models rerank slowly; raise `SONNET_RERANKER_TIMEOUT` or set
`SONNET_RERANKER_POOL_SIZE=5` if searches time out.

vLLM and LM Studio use the same four variables with their own port and model
name. Hosted OpenAI-compatible endpoints (Azure OpenAI, OpenRouter, Gemini's
OpenAI surface, Bedrock gateways) are listed in [providers.md](providers.md).
Changing provider, model, or dimension changes the index identity, so reindex
after switching.

## First run

The first call to `index_directory` downloads the local embedding model when
no `VOYAGE_API_KEY` is set (`all-MiniLM-L6-v2`, about 90 MB, cached under
`CODE_SEARCH_STORAGE`, default `~/.claude_code_search`). Indexes are
per-project and persist across sessions; restart the server after changing any
environment variable, because settings are read once at startup.
