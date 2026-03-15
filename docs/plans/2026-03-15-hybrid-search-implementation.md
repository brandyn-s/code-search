# Hybrid Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add hybrid BM25+vector search with OpenAI embeddings to claude-context-local, matching claude-context's 9/10 query quality while keeping source code local.

**Architecture:** Add OpenAI embedding provider behind the existing `EmbeddingModel` interface, add FTS5 BM25 index alongside existing FAISS vector index, fuse results with Reciprocal Rank Fusion. All changes are additive - existing code paths preserved for `EMBEDDING_PROVIDER=local` and `SEARCH_MODE=semantic`.

**Tech Stack:** Python 3.12, httpx (OpenAI API), SQLite FTS5 (BM25), FAISS (vectors), existing tree-sitter chunking unchanged.

**Design doc:** `docs/plans/2026-03-15-hybrid-search-design.md`

---

### Task 1: OpenAI Embedding Provider

**Files:**
- Create: `embeddings/openai_embedder.py`
- Test: `tests/unit/test_openai_embedder.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_openai_embedder.py
"""Tests for OpenAI embedding provider."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_openai_embedder_encode_returns_correct_shape():
    """OpenAI embedder should return numpy array with correct dimensions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1536, "index": 0},
            {"embedding": [0.2] * 1536, "index": 1},
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from embeddings.openai_embedder import OpenAIEmbeddingModel
        model = OpenAIEmbeddingModel(api_key="test-key")
        result = model.encode(["hello world", "test query"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1536)


def test_openai_embedder_get_dimension():
    """OpenAI embedder should report correct dimension for text-embedding-3-small."""
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    model = OpenAIEmbeddingModel(api_key="test-key", model_name="text-embedding-3-small")
    assert model.get_embedding_dimension() == 1536


def test_openai_embedder_missing_api_key():
    """OpenAI embedder should raise if no API key provided."""
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingModel(api_key="")
```

**Step 2: Run test to verify it fails**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'embeddings.openai_embedder'`

**Step 3: Write minimal implementation**

```python
# embeddings/openai_embedder.py
"""OpenAI embedding model implementation."""

import os
import logging
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

# Known dimensions for OpenAI models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI API embedding model."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 2048,
        **kwargs,
    ):
        # Skip device resolution - not needed for API model
        self._device = "api"
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it as an environment variable "
                "or pass api_key to the constructor."
            )
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._client = httpx.Client(timeout=60.0)
        self._dimension = MODEL_DIMENSIONS.get(model_name, 1536)
        logger.info(f"OpenAI embedder initialized: model={model_name}, dim={self._dimension}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts via OpenAI embeddings API. Ignores kwargs like prompt_name."""
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": batch, "model": self._model_name},
            )
            response.raise_for_status()
            data = response.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "embedding_dimension": self._dimension,
            "provider": "openai",
            "device": "api",
            "status": "loaded",
        }

    def cleanup(self):
        if hasattr(self, "_client"):
            self._client.close()
```

**Step 4: Run test to verify it passes**

Run: `cd C:~/Documents/GitHub/claude-context-local && .venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add embeddings/openai_embedder.py tests/unit/test_openai_embedder.py
git commit -m "feat: add OpenAI embedding provider"
```

---

### Task 2: Register providers and update CodeEmbedder

**Files:**
- Modify: `embeddings/embedding_models_register.py`
- Modify: `embeddings/embedder.py:21-43` (constructor)
- Modify: `embeddings/embedder.py:121-124` (embed_chunk encode call)
- Modify: `embeddings/embedder.py:176-180` (embed_chunks encode call)
- Modify: `embeddings/embedder.py:226-230` (embed_query encode call)
- Modify: `pyproject.toml:28-53` (dependencies)
- Test: `tests/unit/test_embedding_provider_selection.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_embedding_provider_selection.py
"""Tests for embedding provider selection via env vars."""
import os
import pytest
from unittest.mock import patch


def test_openai_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=openai should create OpenAI model."""
    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
    }):
        from embeddings.embedder import CodeEmbedder
        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["provider"] == "openai"


def test_local_provider_selected_when_env_set():
    """EMBEDDING_PROVIDER=local should create local model."""
    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "local",
    }, clear=False):
        from embeddings.embedder import CodeEmbedder
        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        assert info["status"] == "loaded"
        assert "MiniLM" in info.get("model_name", "") or info.get("embedding_dimension") == 384


def test_default_provider_is_openai_when_key_present():
    """When no EMBEDDING_PROVIDER set but OPENAI_API_KEY exists, default to openai."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
    }, clear=False):
        env = os.environ.copy()
        env.pop("EMBEDDING_PROVIDER", None)
        with patch.dict(os.environ, env, clear=True):
            from embeddings.embedder import CodeEmbedder
            embedder = CodeEmbedder()
            info = embedder.get_model_info()
            assert info["provider"] == "openai"
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_embedding_provider_selection.py -v`
Expected: FAIL - current CodeEmbedder doesn't read `EMBEDDING_PROVIDER`

**Step 3: Update the registry**

Replace the full content of `embeddings/embedding_models_register.py`:

```python
"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel
from embeddings.sentence_transformer import SentenceTransformerModel
from embeddings.openai_embedder import OpenAIEmbeddingModel

AVAILIABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
}

# Provider -> (model_class, default_model_name, requires_api_key)
PROVIDERS = {
    "openai": (OpenAIEmbeddingModel, "text-embedding-3-small", True),
    "local": (SentenceTransformerModel, "sentence-transformers/all-MiniLM-L6-v2", False),
    "gemma": (GemmaEmbeddingModel, "google/embeddinggemma-300m", False),
}
```

**Step 4: Update CodeEmbedder constructor**

Replace `embeddings/embedder.py` lines 21-46 (the `__init__` method) with:

```python
class CodeEmbedder:
    """Wrapper for embedding code chunks."""

    def __init__(
        self,
        model_name: str = "",
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        if not cache_dir:
            cache_dir = str(get_storage_dir() / "models")
        self.device = device
        self._logger = logging.getLogger(__name__)

        # Determine provider from env
        provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
        if not provider:
            # Default: openai if key exists, else local
            provider = "openai" if os.environ.get("OPENAI_API_KEY") else "local"

        if provider == "openai":
            from embeddings.openai_embedder import OpenAIEmbeddingModel
            model_name = model_name or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
            self._model = OpenAIEmbeddingModel(model_name=model_name)
        elif provider == "local":
            from embeddings.sentence_transformer import SentenceTransformerModel
            model_name = model_name or os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self._model = SentenceTransformerModel(model_name=model_name, cache_dir=cache_dir, device=device)
        elif provider == "gemma":
            model_name = model_name or "google/embeddinggemma-300m"
            model_class = AVAILIABLE_MODELS[model_name]
            self._model = model_class(cache_dir=cache_dir, device=device)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}. Use 'openai', 'local', or 'gemma'.")

        self._logger.info(f"Embedding provider: {provider}, model: {model_name}")
```

Add `import os` to the top of `embeddings/embedder.py`.

**Step 5: Strip `prompt_name` from encode calls**

The `embed_chunk`, `embed_chunks`, and `embed_query` methods pass `prompt_name="Retrieval-document"` and `prompt_name="InstructionRetrieval"` to `self._model.encode()`. The OpenAI embedder ignores these via `**kwargs`, but for clarity, make the calls conditional:

In `embeddings/embedder.py`, change the three encode calls (lines 121-124, 176-180, 226-230) to not pass `prompt_name` when the model doesn't support it. The simplest approach: the OpenAI embedder's `encode()` already ignores unknown kwargs, so no change needed. The existing code works as-is.

**Step 6: Add httpx to pyproject.toml**

Add `"httpx>=0.27.0",` to the `dependencies` list in `pyproject.toml` (after line 33).

**Step 7: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_openai_embedder.py tests/unit/test_embedding_provider_selection.py -v`
Expected: All pass

**Step 8: Commit**

```bash
git add embeddings/embedding_models_register.py embeddings/embedder.py pyproject.toml tests/unit/test_embedding_provider_selection.py
git commit -m "feat: configurable embedding provider via EMBEDDING_PROVIDER env var"
```

---

### Task 3: FTS5 BM25 Index

**Files:**
- Modify: `search/indexer.py:17-34` (add FTS5 init to `__init__`)
- Modify: `search/indexer.py:89-134` (add FTS5 insert in `add_embeddings`)
- Modify: `search/indexer.py:410-426` (add FTS5 cleanup in `clear_index`)
- Test: `tests/unit/test_fts5_index.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fts5_index.py
"""Tests for FTS5 BM25 index in CodeIndexManager."""
import tempfile
import numpy as np
import pytest
from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult


def _make_result(chunk_id: str, content: str, file_path: str = "test.py") -> EmbeddingResult:
    """Helper to create an EmbeddingResult with FTS-relevant metadata."""
    return EmbeddingResult(
        embedding=np.random.randn(384).astype(np.float32),
        chunk_id=chunk_id,
        metadata={
            "file_path": file_path,
            "relative_path": file_path,
            "content_preview": content,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 10,
            "name": chunk_id.split(":")[-1] if ":" in chunk_id else None,
            "parent_name": None,
            "docstring": None,
            "decorators": [],
            "imports": [],
            "complexity_score": 1,
            "tags": [],
            "folder_structure": [],
        },
    )


def test_fts5_search_finds_keyword_match():
    """FTS5 should find chunks containing query keywords."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:get_redis", "def get_redis(): return redis.Redis(host='localhost')"),
            _make_result("b.py:1-10:func:get_db", "def get_db(): return sqlite3.connect('data.db')"),
            _make_result("c.py:1-10:func:health_check", "def health_check(): return {'status': 'ok'}"),
        ])

        results = mgr.search_bm25("redis", k=5)
        assert len(results) >= 1
        assert any("redis" in r[0].lower() or "redis" in r[2].get("content_preview", "").lower() for r in results)


def test_fts5_search_returns_empty_on_no_match():
    """FTS5 should return empty list when no keywords match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:foo", "def foo(): return 42"),
        ])

        results = mgr.search_bm25("nonexistent_keyword_xyz", k=5)
        assert results == []


def test_fts5_cleared_on_clear_index():
    """clear_index should also clear FTS5 data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CodeIndexManager(tmpdir)
        mgr.add_embeddings([
            _make_result("a.py:1-10:func:get_redis", "def get_redis(): return redis.Redis()"),
        ])
        assert len(mgr.search_bm25("redis", k=5)) >= 1

        mgr.clear_index()

        # Re-initialize after clear
        mgr2 = CodeIndexManager(tmpdir)
        assert mgr2.search_bm25("redis", k=5) == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fts5_index.py -v`
Expected: FAIL with `AttributeError: 'CodeIndexManager' object has no attribute 'search_bm25'`

**Step 3: Add FTS5 to CodeIndexManager**

Add these methods to `search/indexer.py` in the `CodeIndexManager` class:

After `__init__` (around line 34), add FTS5 initialization:

```python
    def _init_fts5(self):
        """Initialize FTS5 full-text search table."""
        import sqlite3
        self._fts_db_path = self.storage_dir / "fts5.db"
        self._fts_conn = sqlite3.connect(
            str(self._fts_db_path),
            check_same_thread=False,
        )
        self._fts_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id,
                content,
                file_path,
                name,
                tokenize='porter unicode61'
            )
        """)
        self._fts_conn.commit()
```

Call `self._init_fts5()` at the end of `__init__`.

In `add_embeddings` (after the metadata_db commit around line 134), add FTS5 inserts:

```python
        # Add to FTS5 index
        for result in embedding_results:
            content = result.metadata.get("content_preview", "")
            file_path = result.metadata.get("relative_path", result.metadata.get("file_path", ""))
            name = result.metadata.get("name", "") or ""
            self._fts_conn.execute(
                "INSERT INTO chunk_fts (chunk_id, content, file_path, name) VALUES (?, ?, ?, ?)",
                (result.chunk_id, content, file_path, name),
            )
        self._fts_conn.commit()
```

Add the BM25 search method:

```python
    def search_bm25(self, query: str, k: int = 50) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using BM25 full-text search. Returns (chunk_id, rank, metadata)."""
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return []

        try:
            cursor = self._fts_conn.execute(
                "SELECT chunk_id, rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, k),
            )
            results = []
            for chunk_id, rank in cursor.fetchall():
                metadata_entry = self.metadata_db.get(chunk_id)
                if metadata_entry:
                    results.append((chunk_id, float(rank), metadata_entry["metadata"]))
            return results
        except Exception as e:
            self._logger.warning(f"FTS5 search failed: {e}")
            return []
```

In `clear_index` (line 410), add FTS5 cleanup before the file removal loop:

```python
        # Close and remove FTS5 database
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
            self._fts_conn = None
        fts_path = self.storage_dir / "fts5.db"
        if fts_path.exists():
            fts_path.unlink()
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fts5_index.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add search/indexer.py tests/unit/test_fts5_index.py
git commit -m "feat: add FTS5 BM25 index for keyword search"
```

---

### Task 4: RRF Fusion and Hybrid Search

**Files:**
- Modify: `search/searcher.py:67-91` (update `search` method)
- Test: `tests/unit/test_hybrid_search.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_hybrid_search.py
"""Tests for RRF fusion and hybrid search."""
import pytest


def test_rrf_fusion_boosts_documents_in_both_lists():
    """Documents appearing in both vector and BM25 results should rank higher."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]
    bm25_results = [("doc_b", -1.0), ("doc_d", -2.0), ("doc_a", -3.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    fused_ids = [chunk_id for chunk_id, score in fused]

    # doc_a and doc_b appear in both lists, should rank top
    assert fused_ids[0] in ("doc_a", "doc_b")
    assert fused_ids[1] in ("doc_a", "doc_b")
    # doc_c and doc_d appear in only one list, should rank lower
    assert "doc_c" in fused_ids
    assert "doc_d" in fused_ids


def test_rrf_fusion_handles_no_overlap():
    """Fusion with zero overlap should interleave by rank."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = [("doc_c", -1.0), ("doc_d", -2.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 4


def test_rrf_fusion_handles_empty_bm25():
    """If BM25 returns nothing, fusion should return vector results only."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = []

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 2
    assert fused[0][0] == "doc_a"
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_hybrid_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'reciprocal_rank_fusion'`

**Step 3: Add RRF function and hybrid search**

Add the RRF function as a module-level function in `search/searcher.py` (before the `IntelligentSearcher` class):

```python
def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        vector_results: List of (chunk_id, score) from vector search, ordered by relevance.
        bm25_results: List of (chunk_id, score) from BM25 search, ordered by relevance.
        k: Smoothing parameter (default 60, industry standard).

    Returns:
        List of (chunk_id, rrf_score) sorted by fused relevance.
    """
    scores: Dict[str, float] = {}
    for rank, (chunk_id, _score) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _score) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Then update `IntelligentSearcher.search()` (lines 67-91) to support hybrid mode:

```python
    def search(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "",
        context_depth: int = 1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        import os
        mode = search_mode or os.environ.get("SEARCH_MODE", "hybrid")

        if mode == "keyword":
            return self._keyword_search(query, k)
        elif mode == "semantic":
            return self._semantic_search(query, k, context_depth, filters)
        else:  # hybrid
            return self._hybrid_search(query, k, context_depth, filters)
```

Add the hybrid search and keyword search methods:

```python
    def _keyword_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Pure BM25 keyword search."""
        raw_results = self.index_manager.search_bm25(query, k=k)
        return [
            self._create_search_result(chunk_id, abs(rank), metadata, 0)
            for chunk_id, rank, metadata in raw_results
        ][:k]

    def _hybrid_search(
        self,
        query: str,
        k: int = 5,
        context_depth: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid BM25 + vector search with RRF fusion."""
        import os
        fusion_k = int(os.environ.get("FUSION_K", "60"))
        candidate_k = 50  # Retrieve 50 from each source

        # Vector search
        optimized_query = self._optimize_query(query)
        query_embedding = self.embedder.embed_query(optimized_query)
        vector_raw = self.index_manager.search(query_embedding, candidate_k, filters)
        vector_pairs = [(chunk_id, sim) for chunk_id, sim, _meta in vector_raw]

        # BM25 search
        bm25_raw = self.index_manager.search_bm25(query, k=candidate_k)
        bm25_pairs = [(chunk_id, rank) for chunk_id, rank, _meta in bm25_raw]

        # Fuse
        fused = reciprocal_rank_fusion(vector_pairs, bm25_pairs, k=fusion_k)

        # Build SearchResult objects for top-k fused results
        # Need metadata for each chunk_id - build lookup from both result sets
        metadata_lookup = {}
        for chunk_id, _sim, metadata in vector_raw:
            metadata_lookup[chunk_id] = metadata
        for chunk_id, _rank, metadata in bm25_raw:
            if chunk_id not in metadata_lookup:
                metadata_lookup[chunk_id] = metadata

        results = []
        for chunk_id, rrf_score in fused[:k]:
            metadata = metadata_lookup.get(chunk_id)
            if metadata:
                result = self._create_search_result(chunk_id, rrf_score, metadata, context_depth)
                results.append(result)

        return results
```

**Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_hybrid_search.py tests/unit/test_fts5_index.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add search/searcher.py tests/unit/test_hybrid_search.py
git commit -m "feat: add RRF fusion and hybrid search mode"
```

---

### Task 5: Run existing tests to verify no regressions

**Files:** None (verification only)

**Step 1: Run the full existing test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v --ignore=tests/integration -x`
Expected: All existing unit tests pass. If any fail, fix the regression before proceeding.

**Step 2: Commit any fixes if needed**

```bash
git add -A && git commit -m "fix: resolve regressions from hybrid search changes"
```

---

### Task 6: EXP-008 validation - run the 10-query battery

**Files:**
- Create: `tests/exp008_validation.py` (temporary test script, delete after validation)

**Step 1: Write the validation script**

This is the same 10-query battery from EXP-008, adapted to run against the fork's API. Index mcp-servers, run queries in all 3 modes (semantic, keyword, hybrid), print side-by-side results for human scoring.

The script should:
1. Set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` from env
2. Index `C:~/Documents/GitHub/mcp-servers`
3. Run each of the 10 queries in all 3 modes
4. Print top-3 results per query per mode

**Step 2: Run Phase 1 (component isolation)**

Run with `SEARCH_MODE=semantic` (OpenAI vectors only), then `SEARCH_MODE=keyword` (FTS5 only), then `SEARCH_MODE=hybrid`.

**Step 3: Score results and compare to claude-context baseline**

Human judges top-1 result per query against 8/10 threshold from design doc.

**Step 4: Clean up**

Delete `tests/exp008_validation.py`. Record results in the knowledge base.

**Step 5: Commit final state**

```bash
git add -A && git commit -m "feat: hybrid search implementation complete"
git push origin main
```

---

## Execution order and dependencies

```
Task 1 (OpenAI embedder) -----> Task 2 (provider selection) -----> Task 5 (regression tests)
                                                                        |
Task 3 (FTS5 BM25 index) -----> Task 4 (RRF + hybrid search) -----> Task 5
                                                                        |
                                                                        v
                                                                   Task 6 (EXP-008 validation)
```

Tasks 1 and 3 are independent and can be implemented in parallel. Task 2 depends on Task 1. Task 4 depends on Task 3. Task 5 depends on both Task 2 and Task 4. Task 6 depends on Task 5.
