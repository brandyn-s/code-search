"""`code-search-mcp doctor`: one report that answers most support questions.

Collects the resolved configuration (secrets redacted), storage location and
size, indexed projects with generation and freshness, the embedding provider
and whether the optional local extra is installed, the reranker mode, provider
reachability, the tree-sitter grammar list, and versions. Never imports torch
or sentence-transformers; availability is checked with ``find_spec``.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from search.env import env_get

SECRET_KEYS = ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GH_TOKEN")
CONFIG_KEYS = (
    "CODE_SEARCH_STORAGE",
    "CODE_SEARCH_ALLOWED_ROOTS",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "CONTENT_MODE",
    "RERANKER",
    "CODE_SYNONYM_PROFILE",
    "QUERY_EXPANSION",
    "QUANTIZATION",
    "CODE_SEARCH_LOG_LEVEL",
    "CODE_SEARCH_LOG_QUERY_TEXT",
    "CODE_SEARCH_QUERY_HISTORY",
    "CODE_SEARCH_QUERY_RETENTION_DAYS",
    "CODE_SEARCH_NONBLOCKING_SEARCH",
    "CODE_SEARCH_DISABLE_AUTO_REINDEX",
    "CODE_SEARCH_STARTUP_AUDIT",
)

PROVIDER_ENDPOINTS = {
    "voyage": ("VOYAGE_API_KEY", "https://api.voyageai.com/v1/models"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/models"),
}


def _version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _redact(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if name in SECRET_KEYS:
        return f"set ({len(value)} chars)"
    return value


def _dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            continue
    return total


def _projects(storage: Path) -> List[Dict[str, Any]]:
    from search import epoch_manifest, index_format

    projects_dir = storage / "projects"
    if not projects_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for project_dir in sorted(p for p in projects_dir.iterdir() if p.is_dir()):
        row: Dict[str, Any] = {"storage_dir": project_dir.name}
        info_path = project_dir / "project_info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            row["error"] = f"project_info.json unreadable: {exc}"
            rows.append(row)
            continue
        row.update(
            {
                "project_path": info.get("project_path"),
                "provider": info.get("embedding_provider"),
                "model": info.get("embedding_model"),
                "identity_status": info.get("index_identity_status"),
                "index_format_version": info.get(index_format.FIELD, 1),
                "pipeline_version": info.get("pipeline_version"),
            }
        )
        incompatible = index_format.format_incompatibility(info)
        if incompatible is not None:
            row["format_status"], row["format_message"] = incompatible
        try:
            read = epoch_manifest.read_with_fallback(project_dir / "index")
            row["manifest_freshness"] = read.freshness
            if read.manifest:
                row["generation"] = read.manifest.get("epoch_id")
                row["committed_at"] = read.manifest.get("created_at") or read.manifest.get("committed_at")
        except Exception as exc:  # noqa: BLE001 - report, never crash doctor
            row["manifest_freshness"] = f"error: {exc}"
        stats_path = project_dir / "index" / "stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                row["chunks"] = stats.get("total_chunks")
                row["files"] = stats.get("files_indexed")
            except (OSError, ValueError):
                pass
        row["size_bytes"] = _dir_size(project_dir)
        rows.append(row)
    return rows


def _reachability(timeout_s: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for provider, (key, url) in PROVIDER_ENDPOINTS.items():
        if not env_get(key):
            out[provider] = {"checked": False, "reason": f"{key} not set"}
            continue
        try:
            import httpx

            started = time.monotonic()
            response = httpx.head(url, timeout=timeout_s, follow_redirects=True)
            out[provider] = {
                "checked": True,
                "reachable": True,
                "status_code": response.status_code,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        except Exception as exc:  # noqa: BLE001 - reachability is best effort
            out[provider] = {"checked": True, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _grammars() -> List[Dict[str, str]]:
    try:
        from chunking.languages import LANGUAGE_MAP
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"chunkers unavailable: {exc}"}]
    seen: Dict[str, List[str]] = {}
    for suffix, (language, _chunker) in LANGUAGE_MAP.items():
        seen.setdefault(language, []).append(suffix)
    return [{"language": lang, "extensions": ", ".join(sorted(exts))} for lang, exts in sorted(seen.items())]


def collect(*, check_network: bool = True, timeout_s: float = 3.0) -> Dict[str, Any]:
    """Gather the doctor report as a JSON-serialisable dict."""
    from common_utils import get_storage_dir
    from embeddings.embedder import resolve_embedding_config
    from search.config import get_search_config
    from embeddings.local_extra import local_extra_available

    report: Dict[str, Any] = {
        "package": {
            "code_search_mcp": _version("code-search-mcp"),
            "mcp_sdk": _version("mcp"),
            "faiss": _version("faiss-cpu"),
            "tree_sitter": _version("tree-sitter"),
            "sentence_transformers": _version("sentence-transformers"),
        },
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
            "executable": sys.executable,
        },
        "config": {key: _redact(key, env_get(key)) for key in CONFIG_KEYS + SECRET_KEYS},
    }

    try:
        cfg = get_search_config()
        emb = resolve_embedding_config()
        report["resolved"] = {
            "embedding_provider": emb.provider,
            "embedding_model": getattr(emb, "model", None) or getattr(emb, "model_name", None),
            "reranker": cfg.reranker_mode,
            "synonym_profile": cfg.synonym_profile,
            "content_mode": cfg.content_mode,
            "local_extra_installed": local_extra_available(),
        }
    except Exception as exc:  # noqa: BLE001
        report["resolved"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        storage = get_storage_dir()
        report["storage"] = {"path": str(storage), "size_bytes": _dir_size(storage)}
        report["projects"] = _projects(storage)
    except Exception as exc:  # noqa: BLE001
        report["storage"] = {"error": f"{type(exc).__name__}: {exc}"}
        report["projects"] = []

    report["reachability"] = _reachability(timeout_s) if check_network else {"checked": False}
    report["grammars"] = _grammars()
    report["problems"] = _problems(report)
    return report


def _problems(report: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    resolved = report.get("resolved", {})
    if resolved.get("embedding_provider") in {"local", "jina", "jina-code", "gemma"} and not resolved.get("local_extra_installed"):
        problems.append("local embeddings selected but the [local] extra is not installed: pip install 'code-search-mcp[local]'")
    for project in report.get("projects", []):
        if project.get("format_status"):
            problems.append(f"{project.get('storage_dir')}: {project.get('format_message')}")
        if str(project.get("manifest_freshness", "")).startswith(("corrupt", "error")):
            problems.append(f"{project.get('storage_dir')}: manifest {project.get('manifest_freshness')}; run verify_index_integrity")
    for provider, result in report.get("reachability", {}).items():
        if isinstance(result, dict) and result.get("checked") and not result.get("reachable", True):
            problems.append(f"{provider} API unreachable: {result.get('error')}")
    return problems


def _human_size(size: Any) -> str:
    if not isinstance(size, (int, float)):
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    pkg = report["package"]
    plat = report["platform"]
    lines.append(f"code-search-mcp {pkg.get('code_search_mcp') or 'source checkout'}  mcp {pkg.get('mcp_sdk')}  python {plat['python']}  {plat['os']}")
    resolved = report.get("resolved", {})
    if "error" in resolved:
        lines.append(f"resolved config: ERROR {resolved['error']}")
    else:
        lines.append(
            f"embeddings={resolved.get('embedding_provider')}({resolved.get('embedding_model')})  "
            f"reranker={resolved.get('reranker')}  synonyms={resolved.get('synonym_profile')}  "
            f"local_extra={'installed' if resolved.get('local_extra_installed') else 'missing'}"
        )
    storage = report.get("storage", {})
    lines.append(f"storage: {storage.get('path', storage.get('error'))} ({_human_size(storage.get('size_bytes'))})")
    lines.append("")
    lines.append("config (secrets redacted):")
    for key, value in report["config"].items():
        if value is not None:
            lines.append(f"  {key}={value}")
    lines.append("")
    projects = report.get("projects", [])
    lines.append(f"projects: {len(projects)}")
    for project in projects:
        if "error" in project:
            lines.append(f"  {project['storage_dir']}: {project['error']}")
            continue
        lines.append(
            f"  {project.get('project_path')}  provider={project.get('provider')}  "
            f"chunks={project.get('chunks')}  identity={project.get('identity_status')}  "
            f"manifest={project.get('manifest_freshness')}  format={project.get('index_format_version')}  "
            f"size={_human_size(project.get('size_bytes'))}"
        )
    lines.append("")
    lines.append("provider reachability:")
    for provider, result in report.get("reachability", {}).items():
        if not isinstance(result, dict):
            continue
        if not result.get("checked"):
            lines.append(f"  {provider}: not checked ({result.get('reason', 'network checks disabled')})")
        elif result.get("reachable"):
            lines.append(f"  {provider}: reachable (HTTP {result.get('status_code')}, {result.get('latency_ms')} ms)")
        else:
            lines.append(f"  {provider}: UNREACHABLE {result.get('error')}")
    lines.append("")
    grammars = report.get("grammars", [])
    lines.append(f"grammars: {len(grammars)}")
    lines.append("  " + ", ".join(g.get("language", "?") for g in grammars))
    lines.append("")
    problems = report.get("problems", [])
    if problems:
        lines.append("problems:")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("problems: none")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None, *, out=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="code-search-mcp doctor", description="Diagnose a code-search installation.")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--no-network", action="store_true", help="skip provider reachability checks")
    parser.add_argument("--timeout", type=float, default=3.0, help="per-provider HTTP timeout in seconds")
    args = parser.parse_args(argv)
    report = collect(check_network=not args.no_network, timeout_s=args.timeout)
    stream = out or sys.stdout
    if args.json:
        json.dump(report, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    else:
        stream.write(render_text(report))
    return 1 if report.get("problems") else 0
