"""Global pytest configuration and fixtures.

Note: Some tests may show DeprecationWarnings from tree-sitter's SWIG-generated C bindings
(SwigPyPacked, SwigPyObject, swigvarlink have no __module__ attribute). These are from the
tree-sitter-python package and cannot be fixed in our code. They appear in Python 3.11+
due to how SWIG generates type information. See:
https://github.com/tree-sitter/py-tree-sitter/issues/
"""

import logging
import os
import sys

if sys.platform == "darwin":
    # faiss-cpu and torch each bundle their own libomp on macOS. When a test
    # process imports faiss first (the indexer does, in almost every test
    # module) and then anything pulls in torch (the local sentence-transformers
    # embedder path), torch's OpenMP runtime aborts the process at
    # __kmp_register_library_startup — pytest dies with SIGABRT (exit 134)
    # partway through collection-ordered execution and the remaining tests
    # never run (observed 2026-06-11; Linux CI is unaffected because torch
    # there links the system runtime). KMP_DUPLICATE_LIB_OK is the standard
    # faiss+torch coexistence setting; it must be in the environment BEFORE
    # the first OpenMP runtime loads, which is why it lives at the very top
    # of the root conftest rather than in a fixture. setdefault so an
    # explicit operator value always wins. Scoped to the test suite on
    # purpose: the production server only hits the pairing when the
    # non-default local embedding provider is selected.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # KMP_DUPLICATE_LIB_OK only downgrades the duplicate-registration abort —
    # the two coexisting runtimes still race in threaded parallel regions.
    # The e2e battery (torch-before-faiss import order, real model encode +
    # QT_8bit train) died SIGSEGV (exit 139) inside
    # test_known_item_hits_and_quantization_parity until OpenMP was pinned to
    # one thread; the unit suite's faiss-before-torch order happens not to
    # hit the racing region (observed 2026-06-12, Python 3.14 arm64).
    # Test corpora are small enough that single-threaded OpenMP costs
    # nothing measurable here. Same setdefault contract as above.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict
try:
    import numpy as np
except ImportError:
    np = None

# Add the package to Python path for testing
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from chunking.multi_language_chunker import MultiLanguageChunker
    from embeddings.embedder import CodeEmbedder
except ImportError:
    MultiLanguageChunker = None
    CodeEmbedder = None

try:
    from tests.fixtures.sample_code import (
        SAMPLE_AUTH_MODULE,
        SAMPLE_DATABASE_MODULE, 
        SAMPLE_API_MODULE,
        SAMPLE_UTILS_MODULE
    )
except ImportError:
    SAMPLE_AUTH_MODULE = SAMPLE_DATABASE_MODULE = SAMPLE_API_MODULE = SAMPLE_UTILS_MODULE = None

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on file path and location
        path_str = str(item.fspath)

        # First, determine if it's unit or integration
        if "tests/unit/" in path_str or "test_system.py" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration/" in path_str:
            item.add_marker(pytest.mark.integration)

        # Then add specific markers based on test file name
        if "test_chunking" in path_str:
            item.add_marker(pytest.mark.chunking)
        elif "test_embeddings" in path_str:
            item.add_marker(pytest.mark.embeddings)
        elif "test_indexing" in path_str:
            item.add_marker(pytest.mark.search)
        elif "test_mcp_server" in path_str:
            item.add_marker(pytest.mark.mcp)


# Mock embedding model for testing
class EmbeddingModelMock:
    """Mock embedding model for testing."""

    def __init__(self, cache_dir=None, device="cpu"):
        """Initialize mock model."""
        self.cache_dir = cache_dir
        self.device = device
        self.model_name = "google/embeddinggemma-300m"
        if np is not None:
            self.rng = np.random.RandomState(42)  # Deterministic seed for reproducible tests

    def encode(self, texts, **kwargs):
        """Return mock embeddings."""
        # Return 768-dimensional embeddings (matching EmbeddingGemma)
        # Use deterministic seed for reproducible tests
        if np is not None:
            return self.rng.randn(len(texts), 768).astype(np.float32)
        return [[0.0] * 768 for _ in texts]

    def get_embedding_dimension(self):
        """Return embedding dimension."""
        return 768

    def get_model_info(self):
        """Return mock model info."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": 768,
            "max_seq_length": 512,
            "device": self.device,
            "status": "loaded"
        }

    def cleanup(self):
        """Cleanup mock resources."""
        pass


# Patch AVAILABLE_MODELS with mock for integration tests
def pytest_configure(config):
    """Configure pytest with custom markers and mock embedding models."""
    # Add filter to suppress SWIG deprecation warnings from tree-sitter
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")

    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "mcp: MCP server related tests")
    config.addinivalue_line("markers", "embeddings: Embedding generation tests")
    config.addinivalue_line("markers", "chunking: Code chunking tests")
    config.addinivalue_line("markers", "search: Search functionality tests")

    # Patch AVAILABLE_MODELS with mock for integration tests
    try:
        from embeddings import embedding_models_register
        embedding_models_register.AVAILIABLE_MODELS["google/embeddinggemma-300m"] = EmbeddingModelMock
    except ImportError:
        pass

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_projects():
    """Remove test-project artifacts from shared storage after test session."""
    yield
    storage_dir = Path.home() / ".claude_code_search" / "projects"
    if storage_dir.exists():
        for d in storage_dir.iterdir():
            if d.is_dir() and (d.name.startswith("proj_") or d.name.startswith("test-project_")):
                shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset MCP server global state
    try:
        import mcp_server.server as server_module
        server_module._embedder = None
        server_module._index_manager = None
        server_module._searcher = None
        server_module._storage_dir = None
    except ImportError:
        pass  # Module might not be available in some tests

    # R11: SearchConfig is memoized via lru_cache for production efficiency,
    # but tests routinely monkey-patch env vars between calls. Clear the
    # cache before every test so each test sees a fresh env-var read.
    try:
        from search.config import get_search_config
        get_search_config.cache_clear()
    except ImportError:
        pass

    yield

    # Cleanup after test if needed
    pass


# Test fixtures
@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir(parents=True)
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_codebase(temp_project_dir: Path) -> Dict[str, Path]:
    """Create a sample codebase with various Python modules."""
    if not SAMPLE_AUTH_MODULE:
        pytest.skip("Sample code not available")
    
    # Create directory structure
    src_dir = temp_project_dir / "src"
    src_dir.mkdir()
    
    auth_dir = src_dir / "auth"
    auth_dir.mkdir()
    
    database_dir = src_dir / "database" 
    database_dir.mkdir()
    
    api_dir = src_dir / "api"
    api_dir.mkdir()
    
    utils_dir = src_dir / "utils"
    utils_dir.mkdir()
    
    # Create Python files with sample code
    files = {}
    
    # Authentication module
    auth_file = auth_dir / "authenticator.py"
    auth_file.write_text(SAMPLE_AUTH_MODULE)
    files['auth'] = auth_file
    
    # Database module
    db_file = database_dir / "manager.py"
    db_file.write_text(SAMPLE_DATABASE_MODULE)
    files['database'] = db_file
    
    # API module
    api_file = api_dir / "endpoints.py"
    api_file.write_text(SAMPLE_API_MODULE)
    files['api'] = api_file
    
    # Utils module
    utils_file = utils_dir / "helpers.py"
    utils_file.write_text(SAMPLE_UTILS_MODULE)
    files['utils'] = utils_file
    
    # Add __init__.py files
    for directory in [src_dir, auth_dir, database_dir, api_dir, utils_dir]:
        init_file = directory / "__init__.py"
        init_file.write_text("# Package init file")
        
    return files


@pytest.fixture
def chunker(temp_project_dir: Path) -> 'MultiLanguageChunker':
    """Create a MultiLanguageChunker instance."""
    if not MultiLanguageChunker:
        pytest.skip("MultiLanguageChunker not available")
    return MultiLanguageChunker(str(temp_project_dir))


@pytest.fixture  
def mock_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory for tests."""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir(parents=True)
    
    # Create subdirectories
    (storage_dir / "models").mkdir()
    (storage_dir / "index").mkdir()
    (storage_dir / "cache").mkdir()
    
    return storage_dir


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'embedding_model': 'google/embeddinggemma-300m',
        'test_batch_size': 2,  # Small batch size for tests
        'test_timeout': 30,    # Timeout for tests
        'mock_embeddings': False,  # Use real embeddings if available
        'embedding_dimension': 768,
        'max_chunks_for_test': 10  # Limit chunks in tests
    }


@pytest.fixture(scope="session")
def ensure_model_downloaded(test_config):
    """Ensure the embedding model is downloaded before running tests."""
    import os
    import subprocess
    from pathlib import Path
    
    # Check if we should use mocks instead
    if os.environ.get('PYTEST_USE_MOCKS', '').lower() in ('1', 'true', 'yes'):
        pytest.skip("Using mocks instead of real model")
    
    # Try to download model
    script_path = Path(__file__).parent / "scripts" / "download_model.py"
    if script_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--model", test_config['embedding_model']], 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            if result.returncode != 0:
                pytest.skip(f"Could not download model: {result.stderr}")
        except subprocess.TimeoutExpired:
            pytest.skip("Model download timed out")
        except Exception as e:
            pytest.skip(f"Error downloading model: {e}")
    else:
        pytest.skip("Download script not found")
    
    return True


@pytest.fixture
def embedder_with_cleanup(mock_storage_dir):
    """Create a CodeEmbedder with proper GPU memory cleanup."""
    if not CodeEmbedder:
        pytest.skip("CodeEmbedder not available")
    
    # Create embedder with CPU device to avoid GPU memory issues
    embedder = CodeEmbedder(
        cache_dir=str(mock_storage_dir / "models"),
        device="cpu"  # Force CPU for tests to avoid VRAM issues
    )
    
    yield embedder
    
    # Cleanup after test
    try:
        embedder.cleanup()
    except Exception:
        logging.getLogger(__name__).debug("embedder cleanup failed", exc_info=True)
