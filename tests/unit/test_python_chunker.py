"""Unit tests for Python tree-sitter chunking."""

import pytest

from chunking.languages import PythonChunker


@pytest.mark.unit
class TestPythonChunker:
    """Test Python-specific tree-sitter chunking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        try:
            self.chunker = PythonChunker()
        except ValueError:
            pytest.skip("tree-sitter-python not installed")

    def test_function_chunking(self):
        """Test chunking of Python functions."""
        code = '''
def simple_function():
    """A simple function."""
    return 42

def async_function():
    """An async function."""
    import asyncio
    await asyncio.sleep(1)

class MyClass:
    def method(self):
        return "method"
'''
        chunks = self.chunker.chunk_code(code)

        # Should find 2 functions and 1 class
        assert len(chunks) >= 2, "Should find at least functions and class"
        assert isinstance(chunks, list), "chunk_code should return a list"

        # Check function names in metadata
        func_names = [c.metadata.get('name') for c in chunks if 'name' in c.metadata]
        has_function = 'simple_function' in func_names or any(
            'simple_function' in c.content for c in chunks
        )
        assert has_function, "Should find simple_function in chunks"

        # Check chunk structure
        for chunk in chunks:
            assert hasattr(chunk, 'content'), "Each chunk should have content"
            assert hasattr(chunk, 'metadata'), "Each chunk should have metadata"
            assert hasattr(chunk, 'node_type'), "Each chunk should have node_type"
            assert isinstance(chunk.content, str), "Chunk content should be a string"

    def test_class_chunking(self):
        """Test chunking of Python classes."""
        code = '''
class SimpleClass:
    """A simple class."""

    def __init__(self):
        self.value = 0

    def method(self):
        return self.value

@dataclass
class DataClass:
    field1: str
    field2: int
'''
        chunks = self.chunker.chunk_code(code)

        # Should find at least 1 class (SimpleClass)
        assert len(chunks) >= 1, "Should find at least one class"

        # Check for class in node types
        class_chunks = [
            c for c in chunks if 'class' in c.node_type or c.node_type == 'decorated_definition'
        ]
        assert len(class_chunks) > 0, "Should find class chunks"

    def test_decorated_definition(self):
        """Test chunking of decorated definitions."""
        code = '''
@decorator1
@decorator2
def decorated_function():
    return "decorated"

@property
def my_property(self):
    return self._value
'''
        chunks = self.chunker.chunk_code(code)

        # Should find decorated definitions
        assert len(chunks) >= 1, "Should find at least one chunk"

        # Check for decorators in metadata or content
        has_decorator = any(
            'decorator' in str(c.metadata) or '@' in c.content for c in chunks
        )
        assert has_decorator, "Should find decorators in chunks"

    def test_empty_file(self):
        """Test chunking of empty file."""
        code = ''
        chunks = self.chunker.chunk_code(code)
        assert len(chunks) == 0, "Empty file should produce no chunks"

    def test_module_only(self):
        """Test file with only module-level code."""
        code = '''
import os
import sys

CONSTANT = 42
variable = "test"

print("Module level code")
'''
        chunks = self.chunker.chunk_code(code)

        # Should create a module chunk since no functions/classes
        assert len(chunks) == 1, "Module-only file should produce one chunk"
        assert chunks[0].node_type == 'module', "Chunk should be of type module"

    def test_module_chunk_does_not_count_terminal_newline_as_source_line(self):
        """Evidence coordinates must not extend one line past EOF."""
        code = 'import os\n\nSETTING = os.environ.get("SETTING")\n'

        chunks = self.chunker.chunk_code(code)

        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3
