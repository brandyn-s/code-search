# Search Orchestration Skill

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code skill that automatically routes code questions to the right search tool (code-search for conceptual, codebase-memory-mcp for structural) and chains them when a single tool can't fully answer.

**Architecture:** A single skill file (`~/.claude/skills/code-search-router/SKILL.md`) that teaches Claude the routing rules and chaining patterns. No new code - just a prompt-based skill that leverages existing MCP tools. The skill fires on code exploration queries and provides a decision tree for tool selection plus automatic follow-up patterns.

**Tech Stack:** Claude Code skill (SKILL.md), existing MCP tools: `code-search` (7 tools) and `codebase-memory-mcp` (14 tools).

---

### Task 1: Create the routing skill

**Files:**
- Create: `~/.claude/skills/code-search-router/SKILL.md`

**Step 1: Create the skill directory**

```bash
mkdir -p ~/.claude/skills/code-search-router
```

**Step 2: Write the skill**

Create `~/.claude/skills/code-search-router/SKILL.md` with this content:

```markdown
---
name: code-search-router
description: Route code exploration queries to the right search tool - code-search (semantic) for "where is X" conceptual queries, codebase-memory-mcp (graph) for "what connects to X" structural queries. Auto-chains when one tool's result needs the other. Trigger phrases - "find code", "where is", "how does", "what calls", "blast radius", "dead code", "callers of", "who uses", "show me the", "trace", "understand this codebase". Do NOT use for file reading (use Read), simple grep (use Grep), or non-code questions.
---

# Code Search Router

Route code exploration queries to the right tool and chain automatically.

## Tool Inventory

### code-search (semantic + keyword)
- `mcp__code-search__search_code` - find code by meaning/keywords
- `mcp__code-search__index_directory` - index a repo
- `mcp__code-search__find_similar_code` - find similar chunks
- `mcp__code-search__get_index_status` - check index state
- `mcp__code-search__list_projects` - list indexed projects

### codebase-memory-mcp (graph)
- `mcp__codebase-memory-mcp__search_graph` - find nodes by name/pattern
- `mcp__codebase-memory-mcp__query_graph` - Cypher graph queries
- `mcp__codebase-memory-mcp__trace_call_path` - trace call chains
- `mcp__codebase-memory-mcp__get_code_snippet` - get source + metadata
- `mcp__codebase-memory-mcp__get_architecture` - codebase overview
- `mcp__codebase-memory-mcp__detect_changes` - blast radius analysis
- `mcp__codebase-memory-mcp__query_security_surfaces` - security audit

## Routing Decision Tree

### Step 1: Classify the query

| Query pattern | Type | Primary tool |
|--------------|------|-------------|
| "Where is the X code?" | Conceptual | code-search |
| "Find the X implementation" | Conceptual | code-search |
| "How does X work?" | Conceptual | code-search |
| "Show me X patterns" | Conceptual | code-search |
| "What calls X?" | Structural | graph (query_graph) |
| "Who uses X?" | Structural | graph (search_graph + trace) |
| "Blast radius of changing X" | Structural | graph (detect_changes) |
| "Find dead code" | Structural | graph (search_graph max_degree=0) |
| "Show all routes/endpoints" | Structural | graph (get_architecture routes) |
| "Trace from X to Y" | Structural | graph (trace_call_path) |
| "What depends on X?" | Structural | graph (query_graph CALLS inbound) |
| "Understand this codebase" | Overview | graph (get_architecture) then code-search for details |

### Step 2: Execute primary tool

Run the primary tool from Step 1.

### Step 3: Auto-chain if the answer is incomplete

| Primary result | Follow-up | Tool |
|---------------|-----------|------|
| code-search found a function | Need to see who calls it | graph: `query_graph` with CALLS inbound |
| code-search found a function | Need to see what it calls | graph: `get_code_snippet` with include_neighbors=true |
| code-search found a function | Need full source (preview is truncated) | graph: `get_code_snippet` by qualified name |
| graph found callers/callees | Need to understand what a caller does | code-search: `search_code` with the function name |
| graph found a node | Need the actual implementation | graph: `get_code_snippet` or Read tool with file:line |
| "How does X work?" partially answered | Need related code in same domain | code-search: `find_similar_code` with chunk_id from first result |

### Step 4: Present combined results

Format the answer with:
1. Direct answer to the question
2. The primary result (file, function, line numbers)
3. Related context from the chained tool (callers, dependencies, similar code)

## Pre-flight: Ensure Both Indexes Exist

Before routing, check that the target repo is indexed in both tools:

1. `mcp__code-search__get_index_status` - if no index, run `index_directory`
2. `mcp__codebase-memory-mcp__index_status` - if no index, run `index_repository`

Both indexes are required for chaining. If only one exists, route to that tool only and note the limitation.

## Examples

### "Where's the rate limiting code?"
1. Classify: Conceptual -> code-search
2. `search_code(query="rate limiting code")` -> `check_rate_limit` in claude-proxy
3. Auto-chain: `query_graph("MATCH (f)-[:CALLS]->(g) WHERE g.name = 'check_rate_limit' RETURN f.name, f.file LIMIT 10")` -> shows all callers
4. Present: "Rate limiting is implemented in `check_rate_limit()` at claude-proxy/claude_proxy.py:902. It's called by `proxy_messages()` during request processing."

### "What's the blast radius of changing shared/mcp_http.py?"
1. Classify: Structural -> graph
2. `detect_changes(scope="branch")` or `query_graph("MATCH (m:Module)-[:IMPORTS]->(t:Module) WHERE t.file CONTAINS 'mcp_http' RETURN m.name, m.file LIMIT 20")`
3. Auto-chain: `search_code(query="configure_http_transport")` -> shows the main entry point function
4. Present: "15 services import mcp_http.py. The main entry point is `configure_http_transport()`. Here are the importers: ..."

### "Understand the authentication system"
1. Classify: Overview -> both tools
2. `get_architecture(aspects=["routes", "services"])` -> shows service boundaries
3. `search_code(query="authentication logic")` -> finds `_build_oauth`, `_authorize_tool_call`
4. `trace_call_path(function_name="_authorize_tool_call")` -> shows the auth call chain
5. Present: combined narrative of auth architecture
```

**Step 3: Verify the skill loads**

Run in a new terminal: `claude --print-skills 2>&1 | grep code-search-router`
Or just check the file exists: `cat ~/.claude/skills/code-search-router/SKILL.md | head -5`

**Step 4: Commit**

```bash
cd ~/.claude && git add skills/code-search-router/SKILL.md
git commit -m "feat: add code-search-router skill for semantic+graph orchestration"
```

---

### Task 2: Add routing hints to the code-search MCP tool descriptions

The code-search MCP server's tool descriptions should hint at when to use graph search instead. This helps Claude route correctly even without the skill loaded.

**Files:**
- Modify: `C:~/Documents/GitHub/claude-context-local/mcp_server/strings.yaml`

**Step 1: Read the current strings.yaml**

Read `mcp_server/strings.yaml` to see current tool descriptions.

**Step 2: Update the search_code description**

Add a routing hint to the `search_code` tool description. Append to the existing description:

```
Best for conceptual queries: "where is X", "how does X work", "find X patterns".
For structural queries ("what calls X", "blast radius", "dead code"), use codebase-memory-mcp graph tools instead.
```

**Step 3: Verify the server still starts**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -c "from mcp_server.code_search_mcp import CodeSearchMCP; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
cd C:~/Documents/GitHub/claude-context-local
git add mcp_server/strings.yaml
git commit -m "docs: add routing hints to search_code tool description"
git push origin main
```

---

### Task 3: Update agent memory with routing rules

Add the routing rules to the worker agent's topic files so subagents also know how to route.

**Files:**
- Modify: `~/.claude/agent-memory/topics/architecture.md`

**Step 1: Add code search routing section**

Append to `~/.claude/agent-memory/topics/architecture.md`:

```markdown
## Code Search Routing (code-search + codebase-memory-mcp)
- **Conceptual** ("where is X", "find X code", "how does X work"): use `code-search` MCP (semantic+BM25 hybrid)
- **Structural** ("what calls X", "blast radius", "dead code", "all routes"): use `codebase-memory-mcp` (graph)
- **Chain**: code-search finds the function, then graph traces callers/dependencies
- Both must be indexed on the target repo. code-search stores at `~/.claude_code_search/`, graph stores `.db` files per project.
- code-search uses OpenAI embeddings (OPENAI_API_KEY required). Graph is fully local.
```

**Step 2: Commit**

```bash
cd ~/.claude && git add agent-memory/topics/architecture.md
git commit -m "docs: add code search routing rules to agent memory"
```

---

## Execution order and dependencies

```
Task 1 (routing skill) -----> Task 3 (agent memory)
                          |
Task 2 (tool description) -+
```

Tasks 1 and 2 are independent. Task 3 depends on the routing rules being finalized in Task 1.
