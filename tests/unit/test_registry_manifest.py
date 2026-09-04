"""server.json (MCP registry listing) stays consistent with the package."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _manifest() -> dict:
    return json.loads((REPO / "server.json").read_text(encoding="utf-8"))


def test_manifest_names_the_pypi_package_and_matches_pyproject_version() -> None:
    manifest = _manifest()
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # The registry manifest tracks released versions; rehearsal pre-releases
    # (X.Y.ZrcN) in pyproject compare on their base version.
    project = {**project, "version": re.sub(r"rc\d+$", "", project["version"])}
    assert manifest["name"] == "io.github.brandyn-s/code-search"
    assert manifest["$schema"].startswith("https://static.modelcontextprotocol.io/schemas/")
    assert manifest["version"] == project["version"]
    assert len(manifest["description"]) <= 100
    packages = manifest["packages"]
    assert len(packages) == 1
    package = packages[0]
    assert package["registryType"] == "pypi"
    assert package["identifier"] == project["name"] == "code-search-mcp"
    assert package["version"] == project["version"]
    assert package["runtimeHint"] == "uvx"
    assert package["transport"] == {"type": "stdio"}


def test_environment_variables_are_optional_and_secrets_are_marked() -> None:
    variables = {v["name"]: v for v in _manifest()["packages"][0]["environmentVariables"]}
    assert {"VOYAGE_API_KEY", "ANTHROPIC_API_KEY"} <= set(variables)
    for name, variable in variables.items():
        assert variable["isRequired"] is False, name
        assert variable["isSecret"] is name.endswith("_KEY"), name
        assert variable["description"]


def test_readme_carries_the_registry_ownership_marker() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "mcp-name: io.github.brandyn-s/code-search" in readme
