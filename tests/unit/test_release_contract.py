"""Contracts for the source installer and GitHub Release path."""

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
EXTERNAL_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "external-benchmarks.yml"
)


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


def test_release_workflow_attests_and_verifies_the_published_wheel() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "preflight-build:" in workflow
    assert "attest:" in workflow
    assert "publish:" in workflow
    assert workflow.count("id-token: write") == 1
    assert workflow.count("attestations: read") == 1
    assert workflow.count("attestations: write") == 1
    assert workflow.count("contents: write") == 1
    assert "actions/download-artifact@" in workflow
    assert "actions/upload-artifact@" in workflow
    assert (
        "actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6"
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


def test_dispatch_inputs_are_never_interpolated_into_shell_source() -> None:
    for workflow_path in (RELEASE_WORKFLOW, EXTERNAL_WORKFLOW):
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

    external = yaml.safe_load(EXTERNAL_WORKFLOW.read_text(encoding="utf-8"))
    external_job = external["jobs"]["external-eval"]
    assert "VOYAGE_API_KEY" not in external_job.get("env", {})
    secret_steps = [
        step
        for step in external_job["steps"]
        if "VOYAGE_API_KEY" in step.get("env", {})
    ]
    assert [step["name"] for step in secret_steps] == [
        "Check VOYAGE_API_KEY",
        "Eval voyage-4-large vs voyage-code-3 (rerank off)",
    ]


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

    assert "redacted-org" not in active_text
    assert "redacted-org/.github" not in active_text
    assert "github.com/redacted-org/code-search" in active_text
    assert "github.com/redacted-org/code-graph" in active_text
    assert (
        REPO_ROOT / ".github" / "CODEOWNERS"
    ).read_text(encoding="utf-8").strip().endswith("* @redacted-brandyn")


def test_readme_installs_the_verified_versioned_release() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert 'TAG="v0.2.0"' in readme
    assert 'WHEEL="redacted_code_search-0.2.0-py3-none-any.whl"' in readme
    assert (
        'BUNDLE="redacted_code_search-0.2.0-provenance.jsonl"'
        in readme
    )
    assert 'gh release download "$TAG"' in readme
    assert 'gh attestation verify "$WHEEL"' in readme
    assert '--bundle "$BUNDLE"' in readme
    assert 'gh release verify "$TAG"' in readme
    assert "gh release verify-asset" in readme
    assert '.venv/bin/python -m pip install "$WHEEL"' in readme
    assert (
        "git clone https://github.com/redacted-org/code-search.git"
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
