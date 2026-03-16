"""Tests for per-project embedding model config."""

import json
import os
import tempfile


def test_project_info_stores_embedding_config():
    """project_info.json should include embedding_provider and embedding_model."""
    from mcp_server.code_search_server import CodeSearchServer

    os.environ.setdefault("EMBEDDING_PROVIDER", "voyage")
    os.environ.setdefault("VOYAGE_API_KEY", "test-key")

    server = CodeSearchServer()
    with tempfile.TemporaryDirectory() as tmpdir:
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

        assert info_file.exists()
        with open(info_file, "r") as f:
            original = json.load(f)

        # Call again - should NOT overwrite
        server.get_project_storage_dir(project_path)
        with open(info_file, "r") as f:
            second = json.load(f)

        assert original == second
