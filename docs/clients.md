# Using code-search from MCP clients

code-search is a stdio MCP server. Any client that can launch a command can use
it. The examples below start the published package with `uvx`; substitute
`pipx run code-search-mcp`, or the absolute path to a venv's `code-search-mcp`
script, if you prefer.

Both API keys are optional. Without `VOYAGE_API_KEY` the server embeds with a
local model; without `ANTHROPIC_API_KEY` it skips LLM reranking. The server
prints one line at startup saying which mode it resolved to.

## Claude Code

```bash
claude mcp add code-search --scope user -- uvx code-search-mcp
# with cloud providers:
claude mcp add code-search --scope user \
  -e VOYAGE_API_KEY=... -e ANTHROPIC_API_KEY=... -- uvx code-search-mcp
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

## First run

The first call to `index_directory` downloads the local embedding model when
no `VOYAGE_API_KEY` is set (`all-MiniLM-L6-v2`, about 90 MB, cached under
`CODE_SEARCH_STORAGE`, default `~/.claude_code_search`). Indexes are
per-project and persist across sessions; restart the server after changing any
environment variable, because settings are read once at startup.
