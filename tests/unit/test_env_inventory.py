"""Executable inventory of the configuration surface.

Two contracts:

1. Runtime modules never read ``os.environ`` directly; they go through
   ``search.env`` (or the typed parsers in ``search.config``). Whole-environment
   copies for subprocesses and ``setdefault`` writes are allowed.
2. Every variable name passed as a literal to an env accessor is documented in
   ``docs/ENV_REFERENCE.md``, and every documented variable is referenced by
   name somewhere in the runtime source.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = ("search", "embeddings", "chunking", "mcp_server", "merkle")
RUNTIME_FILES = [REPO_ROOT / "common_utils.py"]
for root in RUNTIME_ROOTS:
    RUNTIME_FILES.extend(sorted((REPO_ROOT / root).rglob("*.py")))

ENV_MODULE = REPO_ROOT / "search" / "env.py"
ACCESSORS = {"env_get", "env_flag", "parse_env_int", "parse_env_float",
             "parse_env_bool", "parse_env_enum", "_parse_optional_float"}
ALLOWED_DIRECT = {"copy", "setdefault"}

ENV_REFERENCE = (REPO_ROOT / "docs" / "ENV_REFERENCE.md").read_text(encoding="utf-8")
DOCUMENTED = set(re.findall(r"^\| `([A-Z][A-Z0-9_]+)`", ENV_REFERENCE, flags=re.MULTILINE))

# Variables that are legitimately read but belong to third parties or are
# passed through to subprocesses rather than configuring code-search itself.
PASSTHROUGH = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}


def _direct_environ_reads(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        # os.environ.get / os.environ[...] / os.getenv
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            inner = node.value
            if (
                isinstance(inner.value, ast.Name)
                and inner.value.id == "os"
                and inner.attr == "environ"
                and node.attr not in ALLOWED_DIRECT
            ):
                hits.append(f"os.environ.{node.attr}@{node.lineno}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            inner = node.value
            if isinstance(inner.value, ast.Name) and inner.value.id == "os" and inner.attr == "environ":
                hits.append(f"os.environ[]@{node.lineno}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr == "getenv":
                hits.append(f"os.getenv@{node.lineno}")
    return hits


def _literal_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in ACCESSORS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                keys.add(first.value)
    return keys


def test_runtime_modules_read_environment_only_through_search_env() -> None:
    offenders: list[str] = []
    for path in RUNTIME_FILES:
        if path == ENV_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for hit in _direct_environ_reads(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{hit}")
    assert offenders == [], "\n".join(offenders)


def test_every_variable_read_is_documented() -> None:
    read: set[str] = set()
    for path in RUNTIME_FILES:
        read |= _literal_keys(ast.parse(path.read_text(encoding="utf-8")))
    undocumented = sorted(read - DOCUMENTED - PASSTHROUGH)
    assert undocumented == [], f"read but not in docs/ENV_REFERENCE.md: {undocumented}"


def test_every_documented_variable_is_referenced_in_source() -> None:
    source = "\n".join(p.read_text(encoding="utf-8") for p in RUNTIME_FILES)
    unread = sorted(name for name in DOCUMENTED if f'"{name}"' not in source and f"'{name}'" not in source)
    assert unread == [], f"documented but never referenced: {unread}"
