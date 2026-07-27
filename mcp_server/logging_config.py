"""Package-scoped logging configuration for the MCP process."""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO


_FIRST_PARTY_LOGGERS = (
    "mcp_server",
    "search",
    "embeddings",
    "chunking",
    "merkle",
)
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_first_party_logging(*, stream: TextIO | None = None) -> int:
    """Configure first-party package loggers without changing the root logger."""
    configured_stream = stream if stream is not None else sys.stderr
    raw_level = os.environ.get("CODE_SEARCH_LOG_LEVEL", "INFO")
    level_name = raw_level.strip().upper()
    level = _LEVELS.get(level_name, logging.INFO)
    invalid_level = level_name not in _LEVELS

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for name in _FIRST_PARTY_LOGGERS:
        package_logger = logging.getLogger(name)
        package_logger.setLevel(level)
        package_logger.propagate = False
        handlers = [
            handler
            for handler in package_logger.handlers
            if getattr(handler, "_code_search_console", False)
        ]
        if handlers:
            handler = handlers[0]
            handler.setLevel(level)
            handler.setFormatter(formatter)
        else:
            handler = logging.StreamHandler(configured_stream)
            handler._code_search_console = True  # type: ignore[attr-defined]
            handler.setLevel(level)
            handler.setFormatter(formatter)
            package_logger.addHandler(handler)

    if invalid_level:
        logging.getLogger("mcp_server").warning(
            "Invalid CODE_SEARCH_LOG_LEVEL=%r; using INFO",
            raw_level,
        )
    return level
