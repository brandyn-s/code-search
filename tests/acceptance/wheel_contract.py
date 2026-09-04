"""Installed-wheel acceptance contract.

This is intentionally an explicit acceptance runner instead of a pytest test:
the normal unit suite must not rebuild a distribution for every invocation.
CI invokes this file directly after installing the development dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        completed.check_returncode()
    return completed


def _assert_pytest_contract() -> None:
    probe = (
        "import json, pytest; "
        "c=pytest.Config.fromdictargs({}, []); "
        "print(json.dumps({"
        "'testpaths': c.getini('testpaths'), "
        "'norecursedirs': c.getini('norecursedirs'), "
        "'addopts': c.getini('addopts'), "
        "'filterwarnings': c.getini('filterwarnings')"
        "}))"
    )
    configured = json.loads(
        _run([sys.executable, "-c", probe], cwd=REPO_ROOT).stdout
    )
    assert configured["testpaths"] == ["tests"], configured
    assert "--strict-markers" in configured["addopts"], configured
    assert "--disable-warnings" not in configured["addopts"], configured
    assert not any(
        entry == "ignore::DeprecationWarning"
        for entry in configured["filterwarnings"]
    ), configured


def _assert_documented_source_cli() -> None:
    result = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cleanup_index_orphans.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
    )
    assert "--apply-all" in result.stdout


def _build_wheel(destination: Path) -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(destination),
            str(REPO_ROOT),
        ],
        cwd=destination,
    )
    wheels = list(destination.glob("code_search_mcp-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def _assert_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
        server_source = archive.read("mcp_server/server.py").decode("utf-8")
        implementation_source = archive.read(
            "mcp_server/code_search_server.py"
        ).decode("utf-8")

    assert "search/integrity_audit.py" in names
    assert "search/index_identity.py" in names
    assert "search/profiles/generic-v1.json" in names
    # Wheel METADATA normalizes specifier order (anthropic<1.0.0,>=0.40.0),
    # so match the line and assert both bounds instead of a literal prefix.
    anthropic_requirement = next(
        line
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: anthropic")
    )
    assert ">=" in anthropic_requirement
    assert "<1.0.0" in anthropic_requirement
    assert "Requires-Dist: PyYAML>=6.0" in metadata
    assert "sys.path.insert" not in server_source
    assert "sys.path.insert" not in implementation_source


def _wheel_install_command(
    python: Path,
    wheel: Path,
    *,
    install_dependencies: bool,
) -> list[str]:
    command = [str(python), "-m", "pip", "install"]
    if not install_dependencies:
        command.append("--no-deps")
    command.append(str(wheel))
    return command


def _pip_check_command(python: Path) -> list[str]:
    return [str(python), "-m", "pip", "check"]


def _venv_executable(
    environment_dir: Path,
    name: str,
    *,
    platform_name: str | None = None,
) -> Path:
    active_platform = os.name if platform_name is None else platform_name
    if active_platform == "nt":
        executable_name = name if name.lower().endswith(".exe") else f"{name}.exe"
        return environment_dir / "Scripts" / executable_name
    return environment_dir / "bin" / name


def _assert_fresh_install(
    wheel: Path,
    work_dir: Path,
    *,
    install_dependencies: bool,
) -> None:
    environment_dir = work_dir / "wheel-venv"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    python = _venv_executable(environment_dir, "python")
    entrypoint = _venv_executable(environment_dir, "code-search-mcp")

    _run(
        _wheel_install_command(
            python,
            wheel,
            install_dependencies=install_dependencies,
        ),
        cwd=work_dir,
    )
    _run(_pip_check_command(python), cwd=work_dir)

    isolated_env = os.environ.copy()
    isolated_env.pop("PYTHONPATH", None)
    isolated_env["PYTHONNOUSERSITE"] = "1"
    probe_dir = work_dir / "outside-source"
    probe_dir.mkdir()

    _run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; "
                "from search.integrity_audit import stats_drift; "
                "from search.index_identity import "
                "derive_index_generation, normalize_remote_url; "
                "from search.query_expansion import load_synonym_profile; "
                "assert stats_drift(Path('missing.json'), 0) == 0; "
                "assert normalize_remote_url("
                "'git@GitHub.COM:Org/Repo.git'"
                ") == 'https://github.com/Org/Repo'; "
                "assert load_synonym_profile('generic').id == 'generic-v1'; "
                "assert len(derive_index_generation("
                "repository_id='a' * 64, "
                "source_revision='unborn', "
                "dirty_fingerprint='clean'"
                ")) == 64"
            ),
        ],
        cwd=probe_dir,
        env=isolated_env,
    )
    help_result = _run(
        [str(entrypoint), "--help"],
        cwd=probe_dir,
        env=isolated_env,
    )
    assert "Code Search MCP Server" in help_result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Validate this exact pre-built wheel instead of building from source",
    )
    parser.add_argument(
        "--install-dependencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Install the wheel with its declared runtime dependencies "
            "(default: enabled)"
        ),
    )
    args = parser.parse_args(argv)

    _assert_pytest_contract()
    _assert_documented_source_cli()
    with tempfile.TemporaryDirectory(prefix="code-search-wheel-contract-") as raw:
        work_dir = Path(raw)
        wheel = args.wheel.resolve() if args.wheel else _build_wheel(work_dir)
        if not wheel.is_file() or wheel.suffix != ".whl":
            parser.error(f"--wheel must identify an existing .whl file: {wheel}")
        _assert_wheel_contents(wheel)
        _assert_fresh_install(
            wheel,
            work_dir,
            install_dependencies=args.install_dependencies,
        )
    print("wheel contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
