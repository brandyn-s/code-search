"""Contracts for the source installer and GitHub Release path."""

import subprocess
import sys
import re
import tomllib
from pathlib import Path

import yaml

from tests.acceptance import wheel_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
UNIT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"


def test_wheel_contract_can_validate_an_existing_artifact() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "acceptance" / "wheel_contract.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--wheel" in result.stdout
    assert "--install-dependencies" in result.stdout


def test_wheel_install_mode_controls_no_deps_flag() -> None:
    python = Path("/venv/python")
    wheel = Path("/dist/package.whl")

    dependency_aware = wheel_contract._wheel_install_command(
        python,
        wheel,
        install_dependencies=True,
    )
    metadata_only = wheel_contract._wheel_install_command(
        python,
        wheel,
        install_dependencies=False,
    )

    assert "--no-deps" not in dependency_aware
    assert "--no-deps" in metadata_only


def test_wheel_contract_resolves_platform_specific_venv_executables() -> None:
    environment = Path("/tmp/wheel-venv")

    assert wheel_contract._venv_executable(
        environment,
        "python",
        platform_name="posix",
    ) == environment / "bin" / "python"
    assert wheel_contract._venv_executable(
        environment,
        "python",
        platform_name="nt",
    ) == environment / "Scripts" / "python.exe"
    assert wheel_contract._venv_executable(
        environment,
        "code-search-mcp",
        platform_name="nt",
    ) == environment / "Scripts" / "code-search-mcp.exe"


def test_wheel_contract_checks_installed_dependency_consistency() -> None:
    python = Path("/venv/python")

    assert wheel_contract._pip_check_command(python) == [
        str(python),
        "-m",
        "pip",
        "check",
    ]


def test_merge_workflow_builds_once_and_gates_every_required_lane() -> None:
    workflow = yaml.safe_load(UNIT_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    required_jobs = {
        "build-wheel",
        "unit-tests",
        "frozen-retrieval",
        "wheel-smoke-linux",
        "wheel-smoke-platform",
        "merge-gate",
    }

    assert workflow["permissions"] == {"contents": "read"}
    assert required_jobs <= jobs.keys()
    assert jobs["build-wheel"]["outputs"]["platform_smoke_required"] == (
        "${{ steps.platform-scope.outputs.required }}"
    )

    build_source = "\n".join(
        step.get("run", "") for step in jobs["build-wheel"]["steps"]
    )
    non_build_source = "\n".join(
        step.get("run", "")
        for job_name, job in jobs.items()
        if job_name != "build-wheel"
        for step in job.get("steps", [])
    )
    assert build_source.count("pip wheel") == 1
    assert "py3-none-any.whl" in build_source
    assert "pip wheel" not in non_build_source

    wheel_smoke_linux = jobs["wheel-smoke-linux"]
    assert wheel_smoke_linux["needs"] == "build-wheel"
    assert wheel_smoke_linux["strategy"]["matrix"]["os"] == ["ubuntu-24.04"]
    assert set(wheel_smoke_linux["strategy"]["matrix"]["python-version"]) == {
        "3.12",
        "3.13",
    }

    wheel_smoke_platform = jobs["wheel-smoke-platform"]
    assert wheel_smoke_platform["needs"] == "build-wheel"
    assert "needs.build-wheel.outputs.platform_smoke_required" in wheel_smoke_platform["if"]
    assert set(wheel_smoke_platform["strategy"]["matrix"]["os"]) == {
        "macos-14",
        "windows-2022",
    }
    assert set(wheel_smoke_platform["strategy"]["matrix"]["python-version"]) == {
        "3.12",
        "3.13",
    }

    for smoke_job in (wheel_smoke_linux, wheel_smoke_platform):
        smoke_source = "\n".join(
            step.get("run", "") for step in smoke_job["steps"]
        )
        assert "tests/acceptance/wheel_contract.py" in smoke_source
        assert "--wheel" in smoke_source
        assert "--install-dependencies" in smoke_source
        assert all("cache" not in step.get("with", {}) for step in smoke_job["steps"])

    frozen_source = "\n".join(
        step.get("run", "") for step in jobs["frozen-retrieval"]["steps"]
    )
    assert "bench/eval/build_frozen_model.py" in frozen_source
    assert "bench/eval/check_retrieval_floor.py" in frozen_source
    assert "--mode index-and-eval" in frozen_source
    assert "--provider local" in frozen_source
    assert "--floor-semantic-mrr" in frozen_source
    assert "--floor-semantic-hr1" in frozen_source
    assert "--floor-keyword-mrr" in frozen_source
    assert "--floor-keyword-hr1" in frozen_source
    assert "requirements-dev.txt" in frozen_source
    assert (
        "tests/integration/test_frozen_retrieval_contract.py"
        in frozen_source
    )
    assert "test_frozen_retrieval_contract.py::" not in frozen_source

    gate = jobs["merge-gate"]
    assert gate["name"] == "merge-gate"
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == required_jobs - {"merge-gate"}



def test_release_workflow_attests_and_verifies_the_published_wheel() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "preflight-build:" in workflow
    assert "attest:" in workflow
    assert "publish:" in workflow
    assert workflow.count("id-token: write") == 2  # attest + publish-pypi
    assert workflow.count("attestations: read") == 1
    assert workflow.count("attestations: write") == 1
    assert workflow.count("contents: write") == 1
    assert "actions/download-artifact@" in workflow
    assert "actions/upload-artifact@" in workflow
    # These two assertions pin the SHA that release.yml ALSO pins, so they are a
    # change-DETECTOR, not an invariant-CHECKER: the test cannot independently know
    # the correct SHA, and any legitimate action bump (dependabot, a security
    # advisory) must edit BOTH sides in lockstep or CI goes red. That is by design
    # here — a supply-chain pin should not be silently movable — but it means the
    # red is EXPECTED on every bump and is not a defect in the bumping PR.
    #
    # What the pair genuinely protects: a pin cannot drift to a MUTABLE ref (a tag
    # or branch) without a human editing this literal. What it does NOT protect:
    # that the SHA corresponds to the version the trailing comment claims — nothing
    # here verifies 1e69f48... IS v4.2.2. Verifying that needs a registry lookup,
    # which a unit test deliberately does not do.
    #
    # attest bumped v4.2.0 -> v4.2.1 in the github_actions group (2026-08-04),
    # then v4.2.1 -> v4.2.2 in the same group (2026-08-12).
    assert (
        "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6"
        in workflow
    )
    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
        in workflow
    )
    assert "bundle-path" in workflow
    assert "--bundle \"$BUNDLE\"" in workflow
    assert workflow.count("--deny-self-hosted-runners") == 2
    assert "tests/acceptance/wheel_contract.py --wheel" in workflow
    assert "gh attestation verify" in workflow
    assert "RELEASE_ID" in workflow
    assert '"repos/$GITHUB_REPOSITORY/releases/$RELEASE_ID"' in workflow
    assert "gh release upload" not in workflow
    assert "upload_asset() {" in workflow
    assert '.upload_url | split("{")[0]' in workflow
    assert '--input "$ASSET"' in workflow
    assert '-f name="$ASSET_NAME"' in workflow
    assert '-f label="$ASSET_LABEL"' in workflow
    assert '"$UPLOAD_URL"' in workflow
    assert "-F draft=false" in workflow
    assert "trap cleanup EXIT" in workflow
    assert "|| true" not in workflow
    assert "Could not inspect unpublished tag" in workflow
    assert "Refusing to delete unexpected unpublished tag" in workflow
    assert "Unpublished tag cleanup failed" in workflow
    assert '"repos/$GITHUB_REPOSITORY/git/ref/tags/$TAG"' in workflow
    assert '"repos/$GITHUB_REPOSITORY/git/tags/$TAG_SHA"' in workflow
    assert "gh release verify " in workflow
    assert "gh release verify-asset" in workflow
    assert "unit-tests.yml/runs" in workflow
    assert "group: code-search-release" in workflow


def test_release_preflight_does_not_duplicate_runtime_dependencies() -> None:
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    steps = {
        step["name"]: step
        for step in workflow["jobs"]["preflight-build"]["steps"]
        if "name" in step
    }
    install = steps["Install build and acceptance dependencies"]["run"]

    assert "setuptools>=68.0" in install
    assert "wheel pytest PyYAML" in install
    assert "requirements.txt" not in install
    assert "requirements-dev.txt" not in install


def test_dispatch_inputs_are_never_interpolated_into_shell_source() -> None:
    for workflow_path in (RELEASE_WORKFLOW,):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        run_blocks = [
            step["run"]
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if "run" in step
        ]
        shell_source = "\n".join(run_blocks)

        assert "${{" not in shell_source


def test_sensitive_tokens_and_secrets_are_step_scoped() -> None:
    release = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    release_jobs = release["jobs"]

    assert release_jobs["preflight-build"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert release_jobs["attest"]["permissions"] == {
        "actions": "read",
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert release_jobs["publish"]["permissions"] == {
        "actions": "read",
        "attestations": "read",
        "contents": "write",
    }
    assert all(
        "GH_TOKEN" not in job.get("env", {})
        for job in release_jobs.values()
    )



def test_active_project_metadata_has_no_retired_organization_references() -> None:
    active_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".github" / "CODEOWNERS",
        REPO_ROOT / ".dependency-immutability-ignore",
        REPO_ROOT / "scripts" / "install.sh",
    ]
    active_text = "\n".join(
        path.read_text(encoding="utf-8") for path in active_paths
    )

    assert "-Internal-Apps" not in active_text
    assert "-Security/.github" not in active_text
    assert "github.com/brandyn-s/code-search" in active_text
    assert "github.com/brandyn-s/code-graph" in active_text
    assert (
        REPO_ROOT / ".github" / "CODEOWNERS"
    ).read_text(encoding="utf-8").strip().endswith("* @brandyn-s")


def test_readme_installs_the_verified_versioned_release() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = project["project"]["version"]
    # Rehearsal pre-releases (X.Y.ZrcN) keep the README on the base version;
    # the docs describe the release users will install, not the rehearsal.
    version = re.sub(r"rc\d+$", "", version)

    assert f'Install the verified v{version} release' in readme
    assert f'TAG="v{version}"' in readme
    assert f'WHEEL="code_search_mcp-{version}-py3-none-any.whl"' in readme
    assert (
        f'BUNDLE="code_search_mcp-{version}-provenance.jsonl"'
        in readme
    )
    assert 'gh release download "$TAG"' in readme
    assert 'gh attestation verify "$WHEEL"' in readme
    assert '--bundle "$BUNDLE"' in readme
    assert 'gh release verify "$TAG"' in readme
    assert "gh release verify-asset" in readme
    assert '.venv/bin/python -m pip install "$WHEEL"' in readme
    assert (
        "git clone https://github.com/brandyn-s/code-search.git"
        in readme
    )
    assert "./scripts/install.sh" in readme


def test_source_installer_only_installs_the_current_checkout() -> None:
    installer = REPO_ROOT / "scripts" / "install.sh"
    source = installer.read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", str(installer), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert "current source checkout" in result.stdout
    assert "-m venv" in source
    assert "-m pip install" in source
    assert "FarhanAliRaza" not in source
    assert "curl " not in source
    assert "git clone" not in source
    assert "rm -rf" not in source
    assert "${HOME}" not in source
    assert ".local/share" not in source
    assert ".claude_code_search" not in source
    assert "download_model" not in source
