# Security

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
<https://github.com/brandyn-s/code-search/security/advisories/new>.
Do not open a public issue for a suspected vulnerability.

You will get an acknowledgement within 7 days. We aim to ship a fix and
publish an advisory within 90 days of the report, sooner for anything that
exposes source code or credentials. Please include the `code-search-mcp`
version (`code-search-mcp doctor` prints it), the MCP client, and a minimal
reproduction.

## Threat model

code-search is a local MCP server that reads source trees you point it at and
builds a searchable index. It reads every file under the directories you index
(subject to `CODE_SEARCH_ALLOWED_ROOTS` when set, and to its built-in ignore
rules), and writes indexes, manifests, and optional query history under
`CODE_SEARCH_STORAGE` (default `~/.claude_code_search`). Anyone who can read
that directory can read the indexed code. Query text is not logged unless
`CODE_SEARCH_LOG_QUERY_TEXT` is on, and query history stores metadata only
unless `CODE_SEARCH_QUERY_HISTORY=full`. Destructive tools (`clear_index`,
`delete_project`) act only on code-search's own storage, never on your source
tree.

What leaves the machine depends entirely on configuration. With the local
embedding provider and `RERANKER=off`, nothing leaves the machine. With
`VOYAGE_API_KEY` set, chunk text and query text are sent to Voyage AI for
embedding. With `ANTHROPIC_API_KEY` set and reranking enabled, the query and
the candidate snippets being scored are sent to Anthropic. OpenAI embeddings
are used only when `EMBEDDING_PROVIDER=openai` is selected explicitly; an
ambient `OPENAI_API_KEY` alone never causes egress. The `sse`, `http`, and
`streamable-http` transports carry no authentication and no TLS: they are
intended for `localhost` only. Do not bind them to a public interface or put
them behind a reverse proxy without adding authentication yourself.
