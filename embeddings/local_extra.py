"""Guard for the optional local-embeddings dependency set.

`sentence-transformers` (and PyTorch behind it) ship in the ``local`` extra so
cloud-provider users do not download about a gigabyte of wheels they never
run. Every local provider calls :func:`require_local_extra` before touching
those packages so the failure is one clear sentence instead of a traceback.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

LOCAL_EXTRA_HINT = (
    "local embeddings need the optional dependency set: "
    "pip install 'code-search-mcp[local]' (or uvx --from 'code-search-mcp[local]' code-search-mcp), "
    "or set VOYAGE_API_KEY to use cloud embeddings"
)


class LocalEmbeddingsUnavailable(RuntimeError):
    """Raised when a local embedding provider is selected without the extra."""


def local_extra_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def require_local_extra() -> ModuleType:
    """Import and return ``sentence_transformers`` or raise a clear error."""
    try:
        return importlib.import_module("sentence_transformers")
    except ImportError as exc:  # pragma: no cover - exercised in scratch-venv check
        raise LocalEmbeddingsUnavailable(LOCAL_EXTRA_HINT) from exc
