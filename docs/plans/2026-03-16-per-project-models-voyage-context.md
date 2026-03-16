# Per-Project Embedding Models + voyage-context-3

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable per-project embedding model selection so Corsair uses `voyage-code-3` (code-optimized) and knowledge-base uses `voyage-context-3` (document-context-aware), with model config stored in project metadata.

**Architecture:** Store `embedding_provider` and `embedding_model` in each project's `project_info.json` at index time. When searching, load the project's model config and create the right embedder. Add a new `VoyageContextEmbedder` class that uses the `/v1/contextualizedembeddings` endpoint (groups chunks by source file for document-aware embeddings). The standard `OpenAIEmbeddingModel` continues to handle `voyage-code-3` and OpenAI models.

**Tech Stack:** Python 3.12, httpx (existing), Voyage AI REST API, FAISS (existing).

---

### Task 1: Store embedding model config in project_info.json

When indexing a project, record which provider and model were used. When searching, read the stored config to create the correct embedder.

**Files:**
- Modify: `C:~/Documents/GitHub/claude-context-local/mcp_server/code_search_server.py:38-62`
- Test: `C:~/Documents/GitHub/claude-context-local/tests/unit/test_project_model_config.py`

**Step 1: Write the failing test**

Create `C:~/Documents/GitHub/claude-context-local/tests/unit/test_project_model_config.py`:

```python
"""Tests for per-project embedding model config."""
import json
import os
import tempfile
from pathlib import Path

def test_project_info_stores_embedding_config():
    """project_info.json should include embedding_provider and embedding_model."""
    from mcp_server.code_search_server import CodeSearchServer

    os.environ.setdefault("EMBEDDING_PROVIDER", "voyage")
    os.environ.setdefault("VOYAGE_API_KEY", "test-key")

    server = CodeSearchServer()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake project path
        project_path = os.path.join(tmpdir, "test-project")
        os.makedirs(project_path)

        project_dir = server.get_project_storage_dir(project_path)
        info_file = project_dir / "project_info.json"

        with open(info_file, "r") as f:
            info = json.load(f)

        assert "embedding_provider" in info
        assert "embedding_model" in info

def test_project_info_preserves_existing_config():
    """Re-indexing should not overwrite stored model config if project_info.json exists."""
    from mcp_server.code_search_server import CodeSearchServer

    os.environ.setdefault("EMBEDDING_PROVIDER", "voyage")
    os.environ.setdefault("VOYAGE_API_KEY", "test-key")

    server = CodeSearchServer()
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = os.path.join(tmpdir, "test-project")
        os.makedirs(project_path)

        project_dir = server.get_project_storage_dir(project_path)
        info_file = project_dir / "project_info.json"

        # Verify it was created
        assert info_file.exists()
        with open(info_file, "r") as f:
            original = json.load(f)

        # Call again - should NOT overwrite
        project_dir2 = server.get_project_storage_dir(project_path)
        with open(info_file, "r") as f:
            second = json.load(f)

        assert original == second
```

**Step 2: Run test to verify it fails**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_project_model_config.py::test_project_info_stores_embedding_config -v`
Expected: FAIL - `embedding_provider` not in project_info.json

**Step 3: Add embedding config to project_info.json creation**

In `C:~/Documents/GitHub/claude-context-local/mcp_server/code_search_server.py`, modify `get_project_storage_dir` (line 50-60). Add embedding config to the `project_info` dict:

```python
        # Store project info
        project_info_file = project_dir / "project_info.json"
        if not project_info_file.exists():
            project_info = {
                "project_name": project_name,
                "project_path": str(project_path_obj),
                "project_hash": project_hash,
                "created_at": datetime.now().isoformat(),
                "embedding_provider": os.environ.get("EMBEDDING_PROVIDER", "voyage"),
                "embedding_model": os.environ.get("EMBEDDING_MODEL", ""),
            }
            with open(project_info_file, 'w') as f:
                json.dump(project_info, f, indent=2)
```

**Step 4: Run tests**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_project_model_config.py -v`
Expected: All pass

**Step 5: Commit**

```bash
cd C:~/Documents/GitHub/claude-context-local
git add mcp_server/code_search_server.py tests/unit/test_project_model_config.py
git commit -m "feat: store embedding provider/model in project_info.json"
```

---

### Task 2: VoyageContextEmbedder for /v1/contextualizedembeddings

Create a new embedder class that calls the Voyage contextualized embeddings endpoint. This endpoint takes chunks grouped by source document and returns per-chunk embeddings that capture document-level context.

Key API difference from standard embeddings:
- Endpoint: `POST /v1/contextualizedembeddings` (not `/v1/embeddings`)
- Input format: `{"inputs": [["chunk1_of_file1", "chunk2_of_file1"], ["chunk1_of_file2"]], "input_type": "document"}`
- Query format: `{"inputs": [["the query"]], "input_type": "query"}`
- Response: `{"data": [{"embeddings": [[...], [...]]}, {"embeddings": [[...]]}]}` - embeddings grouped by document

**Files:**
- Create: `C:~/Documents/GitHub/claude-context-local/embeddings/voyage_context_embedder.py`
- Test: `C:~/Documents/GitHub/claude-context-local/tests/unit/test_voyage_context_embedder.py`

**Step 1: Write the failing test**

Create `C:~/Documents/GitHub/claude-context-local/tests/unit/test_voyage_context_embedder.py`:

```python
"""Tests for Voyage contextualized chunk embedder."""
import numpy as np
from unittest.mock import patch, MagicMock

def test_voyage_context_encode_flat():
    """Standard encode() should work like OpenAI embedder for queries."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embeddings": [[0.1] * 1024]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.voyage_context_embedder import VoyageContextEmbedder

        model = VoyageContextEmbedder(api_key="test-key")
        result = model.encode(["test query"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)

def test_voyage_context_encode_grouped():
    """encode_grouped() should send chunks grouped by document."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embeddings": [[0.1] * 1024, [0.2] * 1024]},  # 2 chunks from file1
            {"embeddings": [[0.3] * 1024]},  # 1 chunk from file2
        ],
        "usage": {"prompt_tokens": 20, "total_tokens": 20},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.voyage_context_embedder import VoyageContextEmbedder

        model = VoyageContextEmbedder(api_key="test-key")
        grouped = [
            ["chunk1_of_file1", "chunk2_of_file1"],
            ["chunk1_of_file2"],
        ]
        result = model.encode_grouped(grouped)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 1024)  # 3 total chunks flattened

def test_voyage_context_dimension():
    """Should report 1024 dimensions for voyage-context-3."""
    from embeddings.voyage_context_embedder import VoyageContextEmbedder

    model = VoyageContextEmbedder(api_key="test-key")
    assert model.get_embedding_dimension() == 1024
```

**Step 2: Run test to verify it fails**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_voyage_context_embedder.py::test_voyage_context_encode_flat -v`
Expected: FAIL - module not found

**Step 3: Implement VoyageContextEmbedder**

Create `C:~/Documents/GitHub/claude-context-local/embeddings/voyage_context_embedder.py`:

```python
"""Voyage AI contextualized chunk embedder using /v1/contextualizedembeddings."""

import os
import logging
import time
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class VoyageContextEmbedder(EmbeddingModel):
    """Voyage AI contextualized chunk embedding model.

    Uses /v1/contextualizedembeddings which produces per-chunk embeddings
    that capture the full document context. Chunks from the same file are
    grouped together so the model can see cross-section relationships.
    """

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "voyage-context-3",
        batch_delay: float = 1.0,
        **kwargs,
    ):
        self._device = "api"
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required. Set it as an environment variable "
                "or pass api_key to the constructor."
            )
        self._model_name = model_name
        self._base_url = "https://api.voyageai.com/v1"
        self._client = httpx.Client(timeout=300.0)
        self._dimension = 1024
        self._batch_delay = batch_delay

        logger.info(
            f"Voyage context embedder initialized: model={model_name}, dim={self._dimension}"
        )

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts as individual items (for queries).

        Wraps each text in its own document group, since queries
        don't have multi-chunk document context.
        """
        input_type = kwargs.get("input_type", "query")
        grouped = [[text] for text in texts]
        return self._call_api(grouped, input_type)

    def encode_grouped(
        self,
        grouped_texts: List[List[str]],
        input_type: str = "document",
    ) -> np.ndarray:
        """Encode chunks grouped by source document.

        Args:
            grouped_texts: List of documents, each a list of chunk texts.
                Example: [["chunk1_file1", "chunk2_file1"], ["chunk1_file2"]]
            input_type: "document" for indexing, "query" for searching.

        Returns:
            Flat array of embeddings, one per chunk across all documents.
        """
        return self._call_api(grouped_texts, input_type)

    def _call_api(
        self,
        grouped_texts: List[List[str]],
        input_type: str,
    ) -> np.ndarray:
        """Call the contextualized embeddings API with retry."""
        all_embeddings = []

        # Process in batches of documents (not chunks) to respect rate limits
        # Each "document" can have many chunks, so batch by document count
        batch_size = 4  # 4 documents per request
        for i in range(0, len(grouped_texts), batch_size):
            batch = grouped_texts[i : i + batch_size]

            if i > 0 and self._batch_delay > 0:
                time.sleep(self._batch_delay)

            for attempt in range(4):
                try:
                    response = self._client.post(
                        f"{self._base_url}/contextualizedembeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "inputs": batch,
                            "model": self._model_name,
                            "input_type": input_type,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Flatten: each item in data has "embeddings" (list of chunk embeddings)
                    for doc_result in data["data"]:
                        all_embeddings.extend(doc_result["embeddings"])
                    break
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", 0)
                    if attempt < 3 and status in (429, 500, 502, 503, 529):
                        wait = (15 * (attempt + 1)) if status == 429 else 2**attempt
                        logger.warning(
                            f"Context embed batch {i} error {status}, "
                            f"retrying in {wait}s (attempt {attempt + 1}/3)..."
                        )
                        time.sleep(wait)
                        continue
                    raise

        return np.array(all_embeddings, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "embedding_dimension": self._dimension,
            "provider": "voyage-context",
            "device": "api",
            "status": "loaded",
        }

    def cleanup(self):
        if hasattr(self, "_client"):
            self._client.close()
```

**Step 4: Run tests**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_voyage_context_embedder.py -v`
Expected: All 3 pass

**Step 5: Commit**

```bash
cd C:~/Documents/GitHub/claude-context-local
git add embeddings/voyage_context_embedder.py tests/unit/test_voyage_context_embedder.py
git commit -m "feat: VoyageContextEmbedder for contextualized chunk embeddings"
```

---

### Task 3: Add voyage-context provider to CodeEmbedder

Wire the new `VoyageContextEmbedder` into `CodeEmbedder` as the `voyage-context` provider. Add `encode_grouped` passthrough for the indexing pipeline.

**Files:**
- Modify: `C:~/Documents/GitHub/claude-context-local/embeddings/embedder.py:49-60`
- Test: `C:~/Documents/GitHub/claude-context-local/tests/unit/test_openai_embedder.py`

**Step 1: Write the failing test**

Add to `C:~/Documents/GitHub/claude-context-local/tests/unit/test_openai_embedder.py`:

```python
def test_voyage_context_provider_creates_embedder():
    """EMBEDDING_PROVIDER=voyage-context should create VoyageContextEmbedder."""
    import os

    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "voyage-context",
        "VOYAGE_API_KEY": "test-key",
    }):
        from embeddings.embedder import CodeEmbedder
        embedder = CodeEmbedder()
        assert embedder._model._model_name == "voyage-context-3"
        assert hasattr(embedder._model, "encode_grouped")
```

**Step 2: Run test to verify it fails**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py::test_voyage_context_provider_creates_embedder -v`
Expected: FAIL - unknown provider

**Step 3: Add voyage-context provider block**

In `C:~/Documents/GitHub/claude-context-local/embeddings/embedder.py`, after the `voyage` provider block (line 60), add:

```python
        elif provider == "voyage-context":
            from embeddings.voyage_context_embedder import VoyageContextEmbedder

            model_name = model_name or os.environ.get(
                "EMBEDDING_MODEL", "voyage-context-3"
            )
            api_key = os.environ.get("VOYAGE_API_KEY", "")
            self._model = VoyageContextEmbedder(
                api_key=api_key,
                model_name=model_name,
            )
```

Also update the error message on the `else` branch to include the new providers:
```python
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {provider}. "
                f"Use 'openai', 'voyage', 'voyage-context', 'local', or 'gemma'."
            )
```

**Step 4: Run tests**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py -v`
Expected: All 7 pass

**Step 5: Commit**

```bash
cd C:~/Documents/GitHub/claude-context-local
git add embeddings/embedder.py tests/unit/test_openai_embedder.py
git commit -m "feat: wire voyage-context provider into CodeEmbedder"
```

---

### Task 4: Group-by-file embedding in IncrementalIndexer

When the embedder supports `encode_grouped`, group chunks by source file before embedding. This sends all chunks from the same file together so `voyage-context-3` can see document-level context. Falls back to flat embedding for models without `encode_grouped`.

**Files:**
- Modify: `C:~/Documents/GitHub/claude-context-local/search/incremental_indexer.py:173-250`
- Modify: `C:~/Documents/GitHub/claude-context-local/embeddings/embedder.py:219-283`

**Step 1: Add embed_chunks_grouped to CodeEmbedder**

In `C:~/Documents/GitHub/claude-context-local/embeddings/embedder.py`, add a new method after `embed_chunks` (around line 283):

```python
    def embed_chunks_grouped(
        self, chunks: List[CodeChunk], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings with chunks grouped by source file.

        Uses encode_grouped() if the model supports it (voyage-context-3),
        otherwise falls back to flat embed_chunks().
        """
        if not hasattr(self._model, "encode_grouped"):
            return self.embed_chunks(chunks, batch_size)

        from collections import defaultdict

        # Group chunks by source file
        file_groups = defaultdict(list)
        for chunk in chunks:
            file_groups[chunk.relative_path].append(chunk)

        self._logger.info(
            f"Grouped {len(chunks)} chunks into {len(file_groups)} files for contextualized embedding"
        )

        results = []
        file_items = list(file_groups.items())

        for batch_start in range(0, len(file_items), batch_size):
            batch_files = file_items[batch_start : batch_start + batch_size]

            # Build grouped texts and track chunk ordering
            grouped_texts = []
            batch_chunks = []
            for _file_path, file_chunks in batch_files:
                texts = [self.create_embedding_content(c) for c in file_chunks]
                grouped_texts.append(texts)
                batch_chunks.extend(file_chunks)

            # Get grouped embeddings (flattened)
            batch_embeddings = self._model.encode_grouped(
                grouped_texts, input_type="document"
            )

            # Create results
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
                if chunk.name:
                    chunk_id += f":{chunk.name}"

                metadata = {
                    "file_path": chunk.file_path,
                    "relative_path": chunk.relative_path,
                    "folder_structure": chunk.folder_structure,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.name,
                    "parent_name": chunk.parent_name,
                    "docstring": chunk.docstring,
                    "decorators": chunk.decorators,
                    "imports": chunk.imports,
                    "complexity_score": chunk.complexity_score,
                    "tags": chunk.tags,
                    "content_preview": chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    "full_content": chunk.content,
                }

                results.append(
                    EmbeddingResult(
                        embedding=embedding, chunk_id=chunk_id, metadata=metadata
                    )
                )

            if batch_start + batch_size < len(file_items):
                self._logger.info(
                    f"Processed {batch_start + batch_size}/{len(file_items)} files"
                )

        self._logger.info("Grouped embedding generation completed")
        return results
```

**Step 2: Use embed_chunks_grouped in IncrementalIndexer**

In `C:~/Documents/GitHub/claude-context-local/search/incremental_indexer.py`, find the `_full_index` method (line 173). Find where it calls `self.embedder.embed_chunks(all_chunks)` and replace with:

```python
            embedding_results = self.embedder.embed_chunks_grouped(all_chunks)
```

This is a one-line change. Models without `encode_grouped` fall back transparently.

**Step 3: Run all tests**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/ -v`
Expected: All pass

**Step 4: Commit**

```bash
cd C:~/Documents/GitHub/claude-context-local
git add embeddings/embedder.py search/incremental_indexer.py
git commit -m "feat: group-by-file embedding for voyage-context-3 document context"
```

---

### Task 5: Validate voyage-context-3 on knowledge base

Index the knowledge base with `EMBEDDING_PROVIDER=voyage-context` and run semantic queries to compare against the voyage-code-3 baseline.

**Files:**
- No code changes. Configuration + MCP tool calls.

**Step 1: Update MCP config for knowledge-base indexing**

Write a Python script to temporarily set `EMBEDDING_PROVIDER=voyage-context` in `~/.claude.json`:

```python
import json
from pathlib import Path

config_path = Path.home() / ".claude.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

env = config["mcpServers"]["code-search"]["env"]
env["EMBEDDING_PROVIDER"] = "voyage-context"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
```

**Step 2: Restart Claude Code, then re-index knowledge base**

```
mcp__code-search__switch_project("C:~/Documents/knowledge-base")
mcp__code-search__clear_index()
mcp__code-search__index_directory("C:~/Documents/knowledge-base", project_name="knowledge-base", incremental=false)
```

Wait for indexing to complete.

**Step 3: Run the 3 knowledge base queries**

```
search_code("what did we decide about Redis caching")
search_code("OBO authentication token flow")
search_code("hook design patterns windows console")
```

Compare top-1 results and scores against the voyage-code-3 baseline:
- "Redis caching" -> baseline: `message-bus.md:Redis Infrastructure` (score -6.82)
- "OBO auth" -> baseline: `obo-authentication.md:OBO Authentication` (score 4.38)
- "Hook patterns" -> baseline: `hook-design-patterns.md` (score 5.07)

**Step 4: Switch EMBEDDING_PROVIDER back to voyage for Corsair**

Update `~/.claude.json` to set `EMBEDDING_PROVIDER=voyage` (default for code repos). Note: per-project model config (Task 1) means future indexes remember their provider - you won't need to swap env vars once both repos are indexed with their optimal models.

**Step 5: Document results**

Record comparison in this plan file or a new experiment doc.

---

## Execution order

```
Task 1 (project model config)  --independent--
Task 2 (VoyageContextEmbedder) --independent--
Task 3 (wire provider)         --depends on Task 2--
Task 4 (grouped embedding)     --depends on Task 2, Task 3--
Task 5 (validation)            --depends on all above + restart--
```

Tasks 1 and 2 are independent. Task 3 depends on Task 2. Task 4 depends on 2+3. Task 5 is last.
