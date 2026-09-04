"""Code Search MCP - MCPServer (mcp 2.x) tool registration and management."""

import json
import logging
from importlib import metadata
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
import yaml
from mcp_server.code_search_server import CodeSearchServer
from mcp_server.evidence_tools import search_code_evidence as enrich_search_code

# Configure logging
logger = logging.getLogger(__name__)

SERVER_NAME = "code-search"
DISTRIBUTION_NAME = "code-search-mcp"
# Fallback when the package metadata is unavailable (source checkout without
# an install). Keep in sync with pyproject.toml.
FALLBACK_VERSION = "0.4.0"


def package_version() -> str:
    """Version reported in serverInfo: installed distribution, else the fallback."""
    try:
        return metadata.version(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION

# Tool annotations for read/write/destructive classification
TOOL_ANNOTATIONS = {
    "search_code": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "search_code_evidence": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "index_directory": ToolAnnotations(read_only_hint=False),
    "get_indexing_progress": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "find_similar_code": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "get_index_status": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "list_projects": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "search_all_projects": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "switch_project": ToolAnnotations(read_only_hint=False),
    "index_test_project": ToolAnnotations(read_only_hint=False),
    "clear_index": ToolAnnotations(read_only_hint=False, destructive_hint=True),
    "delete_project": ToolAnnotations(destructive_hint=True, idempotent_hint=False),
    "cancel_indexing": ToolAnnotations(destructive_hint=True, idempotent_hint=True),
    "verify_index_integrity": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "get_file_context": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    "code_localize": ToolAnnotations(read_only_hint=True, idempotent_hint=True),
}


class CodeSearchMCP(MCPServer):
    """MCP server that owns tool, resource, and prompt registration."""

    def __init__(
        self,
        server: "CodeSearchServer",
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Initialize the MCP server with a code search server instance.

        ``host`` and ``port`` apply to the network transports only; mcp 2.x
        takes them at ``run()`` time, so they are stored here and forwarded.
        """
        super().__init__(SERVER_NAME, version=package_version())
        self.server = server
        self.host = host
        self.port = port
        self._strings = self._load_strings()
        self._setup()

    def _load_strings(self) -> dict:
        """Load all strings (tool descriptions and help text) from strings.yaml file."""
        strings_file = Path(__file__).parent / "strings.yaml"
        with open(strings_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), "Expected a dict"
            return {
                "tools": data.get("tools", {}),
                "help": data.get("help", ""),
            }

    def _setup(self):
        """Setup all MCP tools, resources, and prompts."""

        # Register tools with descriptions and annotations
        for tool_name, description in self._strings["tools"].items():
            server_method = getattr(self.server, tool_name)
            annotations = TOOL_ANNOTATIONS.get(tool_name)
            self.tool(description=description, annotations=annotations)(server_method)

        # Additive evidence-preserving adapter. This intentionally wraps the
        # production search path rather than creating a second retrieval path.
        self.tool(
            description=(
                "Search code using the normal retrieval pipeline and attach "
                "canonical generation-bound evidence_candidates for exact "
                "nonblank source lines when the live index identity is ready. "
                "Result chunk ranges are retrieval context only; select an "
                "immutable evidence_ref.id from a candidate instead of "
                "inventing or widening a range. Use for evidence-backed FIND "
                "or PROVE workflows."
            ),
            annotations=TOOL_ANNOTATIONS["search_code_evidence"],
        )(self.search_code_evidence)

        # Register resources
        @self.resource("search://stats")
        def get_search_statistics() -> str:
            """Get detailed search index statistics."""
            try:
                index_manager = self.server.get_index_manager()
                stats = index_manager.get_stats()
                return json.dumps(stats, indent=2)
            except Exception as e:
                return json.dumps({"error": f"Failed to get statistics: {str(e)}"})

        # Register prompts
        @self.prompt()
        def search_help() -> str:
            """Get help on using code search tools."""
            return self._strings["help"]

    def search_code_evidence(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_pattern: Optional[str] = None,
        chunk_type: Optional[str] = None,
        include_context: bool = True,
        auto_reindex: bool = True,
        max_age_minutes: float = 5,
        provider: Optional[str] = None,
    ) -> str:
        """Search through the production path and bind results to evidence."""
        return enrich_search_code(
            self.server,
            query=query,
            k=k,
            search_mode=search_mode,
            file_pattern=file_pattern,
            chunk_type=chunk_type,
            include_context=include_context,
            auto_reindex=auto_reindex,
            max_age_minutes=max_age_minutes,
            provider=provider,
        )

    def run(self, transport: str = "stdio"):
        """Run the MCP server with the specified transport.

        ``http`` remains an alias for ``sse`` so existing client configs keep
        working; ``streamable-http`` is available for clients that prefer it.
        Network transports carry no authentication and bind to ``host`` only.
        """
        if transport == "http":
            transport = "sse"

        if transport in ("sse", "streamable-http"):
            logger.info("Starting HTTP server on %s:%s", self.host, self.port)
            return super().run(transport=transport, host=self.host, port=self.port)
        return super().run(transport="stdio")
