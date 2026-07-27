"""FastMCP server for Claude Code integration - main entry point."""
import json
import os
import threading
from typing import TYPE_CHECKING

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp_server.code_search_server import CodeSearchServer


def _run_startup_integrity_audit(server: "CodeSearchServer") -> None:
    """Run verify_index_integrity on startup; log the summary.

    Detects ghost projects, orphan rows, stats drift, and corrupt manifests
    so operators see issues without running /index-repo --audit by hand.
    Runs in a daemon thread so it can't block the MCP server from accepting
    connections, and so a hang in the verifier never wedges startup.

    Set CODE_SEARCH_STARTUP_AUDIT=0 to disable. Output lands in the
    code-search-mcp.log sidecar (auto-created in ~/.claude/logs).

    The verifier itself is non-destructive: this function only LOGS findings.
    Cleanup remains the operator's call via delete_project / /index-repo
    --audit and the documented remediation paths in the audit response.
    """
    if os.environ.get("CODE_SEARCH_STARTUP_AUDIT", "1") == "0":
        logger.info("[STARTUP_AUDIT] disabled via CODE_SEARCH_STARTUP_AUDIT=0")
        return
    audit_logger = logging.getLogger("search.startup_audit")
    try:
        result = server.verify_index_integrity()
        data = json.loads(result)
        summary = data.get("summary", {})
        audit_logger.warning(
            "[STARTUP_AUDIT] projects=%d clean=%d inconsistent=%d "
            "unscannable=%d manifest_fresh=%d manifest_corrupt=%d "
            "manifest_missing=%d total_fts5_orphans=%d "
            "total_metadata_orphans=%d total_stats_drift=%d",
            summary.get("total_projects", 0),
            summary.get("clean", 0),
            summary.get("inconsistent", 0),
            summary.get("unscannable", 0),
            summary.get("manifest_fresh", 0),
            summary.get("manifest_corrupt", 0),
            summary.get("manifest_missing", 0),
            summary.get("total_fts5_orphans", 0),
            summary.get("total_metadata_orphans", 0),
            summary.get("total_stats_drift", 0),
        )
        # Surface specific projects flagged as not-clean so operators
        # can find them without running the full audit themselves.
        for proj in data.get("projects", []):
            status = proj.get("status", "")
            manifest = proj.get("manifest_status", "")
            if status not in ("clean", "skipped") or manifest in (
                "corrupt", "missing",
            ):
                audit_logger.warning(
                    "[STARTUP_AUDIT] project=%s status=%s manifest=%s "
                    "fts5_orphans=%s metadata_orphans=%s stats_drift=%s "
                    "detail=%s",
                    proj.get("name", "?"),
                    status,
                    manifest,
                    proj.get("fts5_orphans"),
                    proj.get("metadata_orphans"),
                    proj.get("stats_drift"),
                    (proj.get("manifest_detail") or "")[:200],
                )
        remediation = data.get("remediation")
        if remediation:
            audit_logger.warning("[STARTUP_AUDIT] remediation: %s", remediation)
    except Exception as exc:
        audit_logger.warning(
            "[STARTUP_AUDIT] failed (non-blocking): %s", exc,
        )


def main():
    """Main entry point for the server."""
    import argparse
    from mcp_server.logging_config import configure_first_party_logging

    configure_first_party_logging()

    parser = argparse.ArgumentParser(description="Code Search MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)"
    )

    args = parser.parse_args()

    # Keep imports behind argument parsing so the installed console script can
    # answer --help before loading the heavyweight runtime dependency graph.
    # Normal startup still imports the same server and registration classes.
    from mcp_server.code_search_server import CodeSearchServer
    from mcp_server.code_search_mcp import CodeSearchMCP

    # Create and run server
    server = CodeSearchServer()
    # Run the integrity audit in a daemon thread so a slow/hung verifier
    # never wedges MCP startup. Findings go to the sidecar log; this is
    # log-only (no destructive cleanup).
    threading.Thread(
        target=_run_startup_integrity_audit,
        args=(server,),
        name="startup-audit",
        daemon=True,
    ).start()
    mcp_server = CodeSearchMCP(server)
    mcp_server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
