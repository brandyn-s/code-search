# MCP Server Concurrency Audit (Plan-2 F1)

**Date**: 2026-05-05
**Source**: `~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md` Phase F1
**Question (the architectural pivot point per the roundtable)**: does the MCP server (`mcp_server/server.py` + FastMCP framework) dispatch tool calls concurrently or serialize them? If serial, F2/F3 cost rises 2-3 weeks → 4-6 weeks (server threading work).

## Conclusion

**Mixed**. The MCP framework dispatches incoming MESSAGES concurrently via `anyio.create_task_group()`, but FastMCP runs synchronous tool functions DIRECTLY in the event loop without `to_thread()` offload. Long-running sync tools (like our `search_code` blocking on `auto_reindex_if_needed`) therefore serialize in practice.

**F-stream cost estimate revision**: F2/F3 cost stays in the 2-3 week band, NOT the 4-6 week worst case. The fix is small (1-2 days of work): make `search_code` either async-with-`to_thread`-for-blocking-parts, or have FastMCP's tool dispatch wrap sync tools in `anyio.to_thread.run_sync`. Either change unblocks concurrent dispatch.

## Evidence (file:line citations)

### Layer 1 — anyio task group (concurrent ✅)

`.venv/Lib/site-packages/mcp/server/lowlevel/server.py:673-683`:

```python
async with anyio.create_task_group() as tg:
    async for message in session.incoming_messages:
        logger.debug("Received message: %s", message)
        tg.start_soon(
            self._handle_message,
            message,
            session,
            lifespan_context,
            raise_exceptions,
        )
```

`tg.start_soon` schedules `_handle_message` to run concurrently with subsequent message dispatches. **The protocol-level dispatch is NOT serial.**

### Layer 2 — FastMCP tool dispatch (concurrent for async, serial for sync ⚠️)

`.venv/Lib/site-packages/mcp/server/fastmcp/utilities/func_metadata.py:74-95`:

```python
async def call_fn_with_arg_validation(
    self,
    fn: Callable[..., Any | Awaitable[Any]],
    fn_is_async: bool,
    ...
) -> Any:
    ...
    if fn_is_async:
        return await fn(**arguments_parsed_dict)
    else:
        return fn(**arguments_parsed_dict)
```

When `fn_is_async=False`, the function runs **directly in the event-loop coroutine** without `anyio.to_thread.run_sync()` or `asyncio.to_thread()`. It blocks the loop until completion.

`.venv/Lib/site-packages/mcp/server/fastmcp/tools/base.py:120-126` confirms the sync detection logic:

```python
def _is_async_callable(obj: Any) -> bool:
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj) or (
        callable(obj) and inspect.iscoroutinefunction(getattr(obj, "__call__", None))
    )
```

Our `CodeSearchServer.search_code` is `def`, not `async def`. So `is_async=False` → direct sync call inside the loop.

### What this means in practice

A long `search_code` call (e.g., one that triggers `auto_reindex_if_needed` on a 10K-chunk project) blocks the event loop for the duration of the reindex. Other incoming MCP messages get queued at Layer 1 (because of the task-group concurrent dispatch) but they can't progress past Layer 2 because the sync tool is occupying the loop.

So:
- Two simultaneous fast `search_code` calls: serial (one blocks the other)
- One long `search_code` + one fast `get_indexing_progress` from the same client: serial — the second waits for the first
- One long `search_code` + one tool call from a SECOND client connection: depends on whether stdio session boundaries map to separate event loops; the audit didn't probe this case

## Implication for Phase F2 (search_code returns last-good-index immediately + freshness metadata)

**Original plan estimate**: 2-3 weeks if MCP can dispatch concurrently; 4-6 weeks if not.

**Revised estimate**: 2-3 weeks. The fix is straightforward — pick one of:

### Option A: make `search_code` async (preferred)

```python
async def search_code(self, ...) -> str:
    ...
    if auto_reindex and self._current_project:
        # offload the blocking reindex check to a thread
        reindex_result = await anyio.to_thread.run_sync(
            lambda: incremental_indexer.auto_reindex_if_needed(...)
        )
    ...
```

This requires changing the method signature, which has ripple effects through any sync caller. Manageable.

### Option B: make the MCP wrapper async-aware for sync tools (no method-signature change)

Override the tool dispatch in `code_search_mcp.py` so sync methods are wrapped in `anyio.to_thread.run_sync`:

```python
def _setup(self):
    for tool_name, description in self._strings["tools"].items():
        server_method = getattr(self.server, tool_name)
        if not asyncio.iscoroutinefunction(server_method):
            # Wrap sync methods to offload to a thread, preserving async semantics
            sync_method = server_method
            async def async_wrapper(*args, _sm=sync_method, **kwargs):
                return await anyio.to_thread.run_sync(lambda: _sm(*args, **kwargs))
            self.tool(description=description, ...)(async_wrapper)
        else:
            self.tool(description=description, ...)(server_method)
```

Option B is less invasive. Option A is cleaner architecturally and lets `search_code` make explicit decisions about which parts to offload.

## Implication for Phase F3 (get_indexing_progress)

If F2 makes `search_code` non-blocking by returning last-good-epoch immediately and dispatching reindex in a background task, then `get_indexing_progress` needs to read shared in-memory state (the active indexing job's progress). With Layer 1 concurrent dispatch + Option A/B unblocking sync tools, this works without server threading work.

The shared state needs a lock (asyncio.Lock or threading.Lock depending on which option chosen) to prevent torn reads — small additional scope, ~½ day.

## Recommendation

Proceed with F2 using **Option A** (make `search_code` async). Estimate: 2-3 weeks for F2+F3 combined, matching the plan's optimistic case.

**This audit unblocks the F-stream cost estimate.** No architectural pivot needed.

## Out of Scope

- Multi-client stdio session isolation: not investigated. The current code-search MCP is local single-user; multi-client semantics are uncharted but not in F's scope.
- HTTP/SSE transport concurrency: docs/mcp_server_concurrency.md focuses on the stdio path (the production transport per `mcp_server/server.py:32`). HTTP transport uses uvicorn which has its own concurrency model.
