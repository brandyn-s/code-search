# Async Indexing with Progress Polling

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `index_directory` non-blocking with a `get_indexing_progress` poll tool so large repos show progress instead of timing out.

**Architecture:** `index_directory` spawns a background thread and returns immediately with a job ID. A shared `_indexing_job` dict on the `CodeSearchServer` tracks state (phase, chunks done, total, errors). New `get_indexing_progress` tool reads that dict. The `IncrementalIndexer` accepts an optional progress callback that the server wires in.

**Tech Stack:** Python threading, existing CodeSearchServer, IncrementalIndexer, strings.yaml tool registration.

---

### Task 1: Add progress callback to IncrementalIndexer

The incremental indexer already logs progress (`Embedded N/M chunks`). Add an optional callback so the server can capture this state.

**Files:**
- Modify: `search/incremental_indexer.py:175-230` (_do_full_index method)
- Test: `tests/unit/test_indexing_progress.py`

**Step 1: Write the failing test**

Create `tests/unit/test_indexing_progress.py`:

```python
"""Tests for indexing progress callback."""
import tempfile
import numpy as np
from unittest.mock import MagicMock
from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult


def test_progress_callback_receives_updates():
    """IncrementalIndexer should call progress_fn with phase and counts."""
    from search.incremental_indexer import IncrementalIndexer

    callback_calls = []

    def track_progress(phase, current, total):
        callback_calls.append((phase, current, total))

    with tempfile.TemporaryDirectory() as tmpdir:
        index_mgr = CodeIndexManager(tmpdir)

        # Mock embedder that returns dummy embeddings
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [
            EmbeddingResult(
                embedding=np.random.randn(384).astype(np.float32),
                chunk_id=f"test:{i}:func:f{i}",
                metadata={
                    "file_path": "test.py", "relative_path": "test.py",
                    "content_preview": "x", "full_content": "x",
                    "chunk_type": "function", "start_line": i, "end_line": i+5,
                    "name": f"f{i}", "parent_name": None, "docstring": None,
                    "decorators": [], "imports": [], "complexity_score": 1,
                    "tags": [], "folder_structure": [],
                },
            )
            for i in range(5)
        ]

        # Mock chunker that returns 5 chunks
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.content = "def f(): pass"
        mock_chunk.relative_path = "test.py"
        mock_chunk.file_path = "test.py"
        mock_chunk.start_line = 1
        mock_chunk.end_line = 5
        mock_chunk.chunk_type = "function"
        mock_chunk.name = "f"
        mock_chunk.parent_name = None
        mock_chunk.docstring = None
        mock_chunk.decorators = []
        mock_chunk.imports = []
        mock_chunk.complexity_score = 1
        mock_chunk.tags = []
        mock_chunk.folder_structure = []
        mock_chunker.chunk_file.return_value = [mock_chunk] * 5
        mock_chunker.is_supported.return_value = True

        indexer = IncrementalIndexer(
            indexer=index_mgr, embedder=mock_embedder, chunker=mock_chunker,
            progress_fn=track_progress,
        )

        # Create a minimal project dir with a file
        import os
        proj = os.path.join(tmpdir, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "test.py"), "w") as f:
            f.write("def f(): pass\n")

        indexer.incremental_index(proj, "test", force_full=True)

        # Should have received chunking and embedding progress
        phases = [c[0] for c in callback_calls]
        assert "chunking" in phases
        assert "embedding" in phases

        # Embedding progress should report current/total
        embed_calls = [(c, t) for p, c, t in callback_calls if p == "embedding"]
        assert len(embed_calls) >= 1
        assert embed_calls[-1][0] > 0  # current > 0
        assert embed_calls[-1][1] > 0  # total > 0

        # Cleanup
        if index_mgr._metadata_db:
            index_mgr._metadata_db.close()
            index_mgr._metadata_db = None
        if hasattr(index_mgr, "_fts_conn") and index_mgr._fts_conn:
            index_mgr._fts_conn.close()
            index_mgr._fts_conn = None
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_indexing_progress.py -v`
Expected: FAIL - `IncrementalIndexer.__init__` doesn't accept `progress_fn`

**Step 3: Add progress_fn to IncrementalIndexer**

In `search/incremental_indexer.py`, modify `__init__` to accept `progress_fn`:

```python
def __init__(self, indexer, embedder, chunker, progress_fn=None):
    self.indexer = indexer
    self.embedder = embedder
    self.chunker = chunker
    self._progress_fn = progress_fn or (lambda phase, current, total: None)
```

Then in `_do_full_index`, add callbacks at key points:

After line 196 (supported files filtered):
```python
self._progress_fn("chunking", 0, len(supported_files))
```

Inside the chunking loop, after `all_chunks.extend(chunks)`:
```python
self._progress_fn("chunking", idx + 1, len(supported_files))
```
(Change the loop to `for idx, file_path in enumerate(supported_files):`)

After line 211 (before embedding loop):
```python
self._progress_fn("embedding", 0, len(all_chunks))
```

Inside the embedding loop, after `all_embedding_results.extend(batch_results)` (line 220):
```python
self._progress_fn("embedding", len(all_embedding_results), len(all_chunks))
```

After line 228 (add_embeddings complete):
```python
self._progress_fn("saving", 0, 0)
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_indexing_progress.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add search/incremental_indexer.py tests/unit/test_indexing_progress.py
git commit -m "feat: add progress callback to IncrementalIndexer"
```

---

### Task 2: Add async job tracking to CodeSearchServer

Add a `_indexing_job` dict to the server that tracks the current indexing job's state. Make `index_directory` run in a background thread and return immediately.

**Files:**
- Modify: `mcp_server/code_search_server.py:29-34` (constructor)
- Modify: `mcp_server/code_search_server.py:257-320` (index_directory method)
- Create: new `get_indexing_progress` method

**Step 1: Write the failing test**

Add to `tests/unit/test_indexing_progress.py`:

```python
def test_server_index_returns_immediately_with_job_id():
    """index_directory should return a job_id immediately, not block."""
    import os, json, time
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")

    from mcp_server.code_search_server import CodeSearchServer
    server = CodeSearchServer()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal project
        proj = os.path.join(tmpdir, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "test.py"), "w") as f:
            f.write("def hello(): return 42\n")

        result = json.loads(server.index_directory(proj))

        # Should return immediately with status "indexing" or "completed"
        assert "status" in result
        assert result["status"] in ("indexing", "completed")

        if result["status"] == "indexing":
            assert "job_id" in result

            # Poll progress
            progress = json.loads(server.get_indexing_progress())
            assert "status" in progress
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_indexing_progress.py::test_server_index_returns_immediately_with_job_id -v`
Expected: FAIL - current `index_directory` blocks

**Step 3: Implement async indexing in CodeSearchServer**

Add to `__init__`:
```python
self._indexing_job = None  # {job_id, status, phase, current, total, errors, result}
self._indexing_thread = None
```

Replace `index_directory` method:

```python
def index_directory(
    self,
    directory_path: str,
    project_name: str = None,
    file_patterns: List[str] = None,
    incremental: bool = True
) -> str:
    """Start indexing a directory. Returns immediately with job status."""
    import threading, uuid

    # If already indexing, return current status
    if self._indexing_job and self._indexing_job["status"] == "indexing":
        return json.dumps({
            "status": "indexing",
            "message": "Indexing already in progress",
            "job_id": self._indexing_job["job_id"],
            "phase": self._indexing_job.get("phase", "unknown"),
            "chunks_done": self._indexing_job.get("current", 0),
            "chunks_total": self._indexing_job.get("total", 0),
        })

    directory_path_obj = Path(directory_path).resolve()
    if not directory_path_obj.exists():
        return json.dumps({"error": f"Directory does not exist: {directory_path_obj}"})
    if not directory_path_obj.is_dir():
        return json.dumps({"error": f"Path is not a directory: {directory_path_obj}"})

    project_name = project_name or directory_path_obj.name
    job_id = uuid.uuid4().hex[:8]

    self._indexing_job = {
        "job_id": job_id,
        "status": "indexing",
        "phase": "starting",
        "current": 0,
        "total": 0,
        "errors": [],
        "directory": str(directory_path_obj),
        "project_name": project_name,
        "result": None,
    }

    def _progress_callback(phase, current, total):
        if self._indexing_job and self._indexing_job["job_id"] == job_id:
            self._indexing_job["phase"] = phase
            self._indexing_job["current"] = current
            self._indexing_job["total"] = total

    def _run_indexing():
        try:
            from search.incremental_indexer import IncrementalIndexer

            index_manager = self.get_index_manager(str(directory_path_obj))
            embedder = self.embedder()
            chunker = MultiLanguageChunker(str(directory_path_obj))

            incremental_indexer = IncrementalIndexer(
                indexer=index_manager,
                embedder=embedder,
                chunker=chunker,
                progress_fn=_progress_callback,
            )

            result = incremental_indexer.incremental_index(
                str(directory_path_obj),
                project_name,
                force_full=not incremental
            )

            stats = incremental_indexer.get_indexing_stats(str(directory_path_obj))

            self._indexing_job["status"] = "completed"
            self._indexing_job["phase"] = "done"
            self._indexing_job["result"] = {
                "success": result.success,
                "files_added": result.files_added,
                "chunks_added": result.chunks_added,
                "time_taken": round(result.time_taken, 2),
                "index_stats": stats,
                "error": result.error,
            }
        except Exception as e:
            logger.error(f"Background indexing failed: {e}", exc_info=True)
            self._indexing_job["status"] = "failed"
            self._indexing_job["phase"] = "error"
            self._indexing_job["result"] = {"error": str(e)}

    self._indexing_thread = threading.Thread(target=_run_indexing, daemon=True)
    self._indexing_thread.start()

    return json.dumps({
        "status": "indexing",
        "job_id": job_id,
        "directory": str(directory_path_obj),
        "project_name": project_name,
        "message": "Indexing started in background. Use get_indexing_progress to check status.",
    })
```

Add `get_indexing_progress` method:

```python
def get_indexing_progress(self) -> str:
    """Get current indexing job progress."""
    if not self._indexing_job:
        return json.dumps({"status": "idle", "message": "No indexing job running"})

    job = self._indexing_job
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "phase": job["phase"],
        "directory": job.get("directory", ""),
        "project_name": job.get("project_name", ""),
    }

    if job["total"] > 0:
        response["chunks_done"] = job["current"]
        response["chunks_total"] = job["total"]
        response["percent"] = round(100 * job["current"] / job["total"], 1)

    if job["status"] in ("completed", "failed") and job.get("result"):
        response["result"] = job["result"]

    return json.dumps(response)
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_indexing_progress.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mcp_server/code_search_server.py tests/unit/test_indexing_progress.py
git commit -m "feat: async index_directory with background thread and progress tracking"
```

---

### Task 3: Register get_indexing_progress as MCP tool

Add the tool description and annotation so Claude Code can call it.

**Files:**
- Modify: `mcp_server/strings.yaml` (add tool description)
- Modify: `mcp_server/code_search_mcp.py:16-25` (add annotation)

**Step 1: Add tool description to strings.yaml**

Add after the `index_directory` entry:

```yaml
  get_indexing_progress: |
    Check the progress of a running index_directory operation. Returns JSON with job_id, status (indexing/completed/failed/idle), current phase (chunking/embedding/saving/done), chunks_done, chunks_total, and percent complete. Call after index_directory returns a job_id. When status is "completed", the result field contains the final indexing stats. Poll every 15-30 seconds for updates.
```

**Step 2: Add annotation to code_search_mcp.py**

In `TOOL_ANNOTATIONS` dict, add:

```python
"get_indexing_progress": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
```

**Step 3: Update index_directory description**

Change the `index_directory` description in strings.yaml to mention async behavior:

```yaml
  index_directory: |
    Start indexing a codebase directory in the background. Returns immediately with a job_id and status. Parses source files into AST chunks (functions, classes, modules across Python, JS, TS, Go, Java, Rust), generates embeddings via OpenAI, and stores in FAISS with SQLite metadata. Use get_indexing_progress to poll progress (chunks_done/chunks_total/percent). Supports incremental mode via Merkle tree change detection.
```

**Step 4: Verify server loads**

Run: `.venv/Scripts/python.exe -c "from mcp_server.code_search_mcp import CodeSearchMCP; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add mcp_server/strings.yaml mcp_server/code_search_mcp.py
git commit -m "feat: register get_indexing_progress MCP tool"
```

---

### Task 4: Update search_code to handle in-progress indexing

When indexing is in progress, `search_code` should return a helpful message instead of empty results or errors.

**Files:**
- Modify: `mcp_server/code_search_server.py` (search_code method, around line 155)

**Step 1: Add check at start of search_code**

At the top of `search_code`, before any search logic:

```python
# If indexing is in progress, report that instead of returning empty
if self._indexing_job and self._indexing_job["status"] == "indexing":
    job = self._indexing_job
    pct = round(100 * job["current"] / job["total"], 1) if job["total"] > 0 else 0
    return json.dumps({
        "query": query,
        "results": [],
        "indexing_in_progress": True,
        "message": f"Indexing in progress ({job['phase']}: {pct}% - {job['current']}/{job['total']} chunks). Results will be available when complete.",
    })
```

**Step 2: Verify with a manual test**

Run the MCP server, start indexing, immediately search - should get the progress message.

**Step 3: Commit**

```bash
git add mcp_server/code_search_server.py
git commit -m "feat: search_code returns progress message during indexing"
```

---

## Execution order

```
Task 1 (progress callback in IncrementalIndexer)
    |
    v
Task 2 (async job tracking in CodeSearchServer) -- depends on Task 1
    |
    v
Task 3 (register MCP tool) -- depends on Task 2
    |
    v
Task 4 (search_code handles in-progress) -- depends on Task 2
```

Tasks 3 and 4 are independent of each other but both depend on Task 2.
