"""Tests for contextual chunk headers in embedding content."""

from unittest.mock import MagicMock


def test_embedding_content_includes_context_header():
    """Embedding content should start with a context header line."""
    from embeddings.embedder import CodeEmbedder
    import os

    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    embedder = CodeEmbedder()

    chunk = MagicMock()
    chunk.relative_path = "claude-proxy/claude_proxy.py"
    chunk.chunk_type = "function"
    chunk.name = "check_rate_limit"
    chunk.parent_name = None
    chunk.docstring = "Check if user has exceeded their rate limit."
    chunk.content = (
        "async def check_rate_limit(key_suffix: str) -> Optional[dict]:\n    pass"
    )
    chunk.folder_structure = ["claude-proxy"]

    content = embedder.create_embedding_content(chunk)

    # Should start with a context header
    first_line = content.split("\n")[0]
    assert "claude-proxy/claude_proxy.py" in first_line
    assert "function" in first_line
    assert "check_rate_limit" in first_line


def test_embedding_content_header_includes_parent():
    """Context header should include parent name for methods."""
    from embeddings.embedder import CodeEmbedder
    import os

    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    embedder = CodeEmbedder()

    chunk = MagicMock()
    chunk.relative_path = "shared/opa_middleware.py"
    chunk.chunk_type = "method"
    chunk.name = "_authorize_tool_call"
    chunk.parent_name = "OPAMiddleware"
    chunk.docstring = None
    chunk.content = (
        "async def _authorize_tool_call(self, scope, rpc_message):\n    pass"
    )
    chunk.folder_structure = ["shared"]

    content = embedder.create_embedding_content(chunk)

    first_line = content.split("\n")[0]
    assert "OPAMiddleware" in first_line
    assert "_authorize_tool_call" in first_line


def test_embedding_content_header_nix_binding():
    """Context header should work for Nix binding chunks."""
    from embeddings.embedder import CodeEmbedder
    import os

    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    embedder = CodeEmbedder()

    chunk = MagicMock()
    chunk.relative_path = "nix/modules/lan-config.nix"
    chunk.chunk_type = "binding"
    chunk.name = "nft_pre_dport_map"
    chunk.parent_name = "options.services.lan-config"
    chunk.docstring = None
    chunk.content = (
        "nft_pre_dport_map = mkOption {\n  type = types.listOf types.str;\n};"
    )
    chunk.folder_structure = ["nix", "modules"]

    content = embedder.create_embedding_content(chunk)

    first_line = content.split("\n")[0]
    assert "lan-config.nix" in first_line
    assert "nft_pre_dport_map" in first_line


def test_embedding_content_header_disabled_by_env():
    """CONTEXTUAL_HEADERS=off should skip the header."""
    from embeddings.embedder import CodeEmbedder
    import os

    os.environ["EMBEDDING_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["CONTEXTUAL_HEADERS"] = "off"

    embedder = CodeEmbedder()

    chunk = MagicMock()
    chunk.relative_path = "test.py"
    chunk.chunk_type = "function"
    chunk.name = "foo"
    chunk.parent_name = None
    chunk.docstring = None
    chunk.content = "def foo(): pass"
    chunk.folder_structure = []

    content = embedder.create_embedding_content(chunk)

    # Should NOT start with a context header
    assert not content.startswith("# From")

    # Cleanup
    os.environ.pop("CONTEXTUAL_HEADERS", None)
