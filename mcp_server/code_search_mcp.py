"""Code Search MCP - FastMCP server for tool registration and management."""

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
import yaml
from mcp_server.code_search_server import CodeSearchServer

# Configure logging
logger = logging.getLogger(__name__)

# Tool annotations for read/write/destructive classification
TOOL_ANNOTATIONS = {
    "search_code": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "index_directory": ToolAnnotations(readOnlyHint=False),
    "get_indexing_progress": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "find_similar_code": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "get_index_status": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "list_projects": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "switch_project": ToolAnnotations(readOnlyHint=False),
    "index_test_project": ToolAnnotations(readOnlyHint=False),
    "clear_index": ToolAnnotations(readOnlyHint=False, destructiveHint=True),
    "delete_project": ToolAnnotations(destructiveHint=True, idempotentHint=False),
    "cancel_indexing": ToolAnnotations(destructiveHint=True, idempotentHint=True),
    "verify_index_integrity": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "get_file_context": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    "code_localize": ToolAnnotations(readOnlyHint=True, idempotentHint=True),
}


class CodeSearchMCP(FastMCP):
    """MCP server that manages FastMCP instance and tool registration."""

    def __init__(
        self,
        server: "CodeSearchServer",
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Initialize the MCP server with a code search server instance."""
        super().__init__("Code Search", host=host, port=port)
        self.server = server
        self._strings = self._load_strings()
        self._setup()

    def _load_strings(self) -> dict:
        """Load all strings (tool descriptions and help text) from strings.yaml file."""
        strings_file = Path(__file__).parent / "strings.yaml"
        with open(strings_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), "Expected a dict"
            return {
                "tools": data.get("tools", {}),
                "help": data.get("help", "")
            }

    def _setup(self):
        """Setup all MCP tools, resources, and prompts."""

        # Register tools with descriptions and annotations
        for tool_name, description in self._strings["tools"].items():
            server_method = getattr(self.server, tool_name)
            annotations = TOOL_ANNOTATIONS.get(tool_name)
            self.tool(description=description, annotations=annotations)(server_method)

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

    def run(self, transport: str = "stdio"):
        """Run the MCP server with specified transport."""
        if transport == "http":
            transport = "sse"

        if transport in ["sse", "streamable-http"]:
            logger.info(
                "Starting HTTP server on %s:%s",
                self.settings.host,
                self.settings.port,
            )
        return super().run(transport=transport)
