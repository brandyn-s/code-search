"""Tests for indexing progress callback."""
import json
import os
import tempfile
import time
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
        index_dir = os.path.join(tmpdir, "index")
        os.makedirs(index_dir)
        index_mgr = CodeIndexManager(index_dir)

        # Mock embedder that returns dummy embeddings
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [
            EmbeddingResult(
                embedding=np.random.randn(384).astype(np.float32),
                chunk_id=f"test:{i}:func:f{i}",
                metadata={
                    "file_path": "test.py", "relative_path": "test.py",
                    "content_preview": "x", "full_content": "x",
                    "chunk_type": "function", "start_line": i, "end_line": i + 5,
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


def test_server_index_returns_immediately_with_job_id():
    """index_directory should return a job_id immediately, not block."""
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
