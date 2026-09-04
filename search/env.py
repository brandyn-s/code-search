"""Single entry point for reading process environment variables.

Every runtime module reads configuration through ``env_get`` (or the typed
parsers in ``search.config`` that build on it) instead of touching
``os.environ`` directly. That gives one place to audit, mock, or log the
configuration surface, and ``tests/unit/test_env_inventory.py`` enforces
that every variable read here is documented in ``docs/ENV_REFERENCE.md``.

``env_get`` has exactly the semantics of ``os.environ.get`` so callers keep
their existing defaults and parsing. Writes (``os.environ.setdefault`` for
Hugging Face offline mode) and whole-environment copies for subprocesses stay
where they are; they are not configuration reads.
"""

from __future__ import annotations

import os
from typing import Optional, overload


@overload
def env_get(name: str) -> Optional[str]: ...


@overload
def env_get(name: str, default: str) -> str: ...


def env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return ``os.environ.get(name, default)``.

    Kept deliberately thin: typed parsing lives in ``search.config``.
    """
    return os.environ.get(name, default)


def env_flag(name: str, default: bool = False) -> bool:
    """Interpret ``on``/``1``/``true``/``yes`` (case-insensitive) as true.

    Convenience for new code; existing call sites keep their exact historical
    comparisons (for example ``== "on"``) so behaviour does not shift.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
