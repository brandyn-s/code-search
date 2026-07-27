"""End-to-end acceptance tests for MCP network transport configuration."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_sse_headers(
    process: subprocess.Popen[str],
    *,
    host: str,
    port: int,
    timeout: float = 15.0,
) -> bytes:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            pytest.fail(
                "MCP subprocess exited before accepting connections "
                f"(exit={process.returncode}).\nstdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.25) as client:
                client.settimeout(1.0)
                client.sendall(
                    (
                        "GET /sse HTTP/1.1\r\n"
                        f"Host: {host}:{port}\r\n"
                        "Accept: text/event-stream\r\n"
                        "Connection: close\r\n\r\n"
                    ).encode("ascii")
                )
                response = b""
                while b"\r\n\r\n" not in response:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                if (
                    response.startswith(b"HTTP/1.1 200")
                    and b"content-type: text/event-stream" in response.lower()
                ):
                    return response
                last_error = AssertionError(
                    f"unexpected HTTP response: {response[:500]!r}"
                )
        except (OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.05)
    pytest.fail(
        f"MCP server did not expose SSE on {host}:{port}: {last_error}"
    )


@pytest.mark.parametrize("transport", ["sse", "http"])
def test_cli_honors_nondefault_network_bind_address(
    tmp_path: Path,
    transport: str,
) -> None:
    host = "127.0.0.1"
    port = _unused_loopback_port()
    environment = os.environ.copy()
    environment.update(
        {
            "CODE_SEARCH_QUERY_HISTORY": "off",
            "CODE_SEARCH_STARTUP_AUDIT": "0",
            "CODE_SEARCH_STORAGE": str(tmp_path / "storage"),
            "HF_HUB_OFFLINE": "1",
            "RERANKER": "off",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for key in ("JINA_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY"):
        environment.pop(key, None)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_server.server",
            "--transport",
            transport,
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        response = _wait_for_sse_headers(
            process,
            host=host,
            port=port,
        )
        assert f"server: uvicorn".encode() in response.lower()
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
