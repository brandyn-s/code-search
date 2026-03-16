"""Unit tests for the main TreeSitterChunker class."""

import tempfile
import shutil
from pathlib import Path

import pytest

from chunking.tree_sitter import TreeSitterChunker


@pytest.mark.unit
class TestTreeSitterChunker:
    """Test main TreeSitterChunker class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.chunker = TreeSitterChunker()
        self.temp_dir = tempfile.mkdtemp()

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after test."""
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_language_detection(self):
        """Test language detection from file extension."""
        from chunking.base_chunker import AVAILABLE_LANGUAGES

        # Python should be supported if tree-sitter-python is installed
        if "python" not in AVAILABLE_LANGUAGES:
            pytest.skip("tree-sitter-python not installed")

        assert self.chunker.is_supported("test.py"), "Should support .py files"

    def test_chunk_python_file(self):
        """Test chunking a Python file."""
        from chunking.base_chunker import AVAILABLE_LANGUAGES

        if "python" not in AVAILABLE_LANGUAGES:
            pytest.skip("tree-sitter-python not installed")

        file_path = Path(self.temp_dir) / "test.py"
        code = """
def test_function():
    return "test"

class TestClass:
    pass
"""
        file_path.write_text(code)

        chunks = self.chunker.chunk_file(str(file_path))

        assert len(chunks) >= 2, "Should chunk both function and class"
        assert all(c.language == "python" for c in chunks), (
            "All chunks should be marked as Python"
        )

    def test_unsupported_file(self):
        """Test handling of unsupported file types."""
        chunks = self.chunker.chunk_file("test.unsupported", "some content")
        assert len(chunks) == 0, "Unsupported files should produce no chunks"
