"""Contract tests for cross-engine repository/index identity."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import search.index_identity as index_identity_module
from search.index_identity import (
    IdentityCaptureError,
    capture_index_identity,
    derive_index_generation,
    identity_mismatch_fields,
    normalize_remote_url,
    validate_index_identity_dict,
)


_SHARED_VECTORS = json.loads(
    (
        Path(__file__).parent.parent
        / "fixtures"
        / "index_identity_vectors.json"
    ).read_text(encoding="utf-8")
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def _committed_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Index Identity Test")
    _git(root, "config", "user.email", "index-identity@example.test")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "initial")
    return root


def test_clean_capture_matches_the_shared_identity_envelope(tmp_path: Path) -> None:
    root = _committed_repo(tmp_path)
    _git(
        root,
        "remote",
        "add",
        "origin",
        "git@GitHub.COM:Example-Org-Dev/code-search.git",
    )
    captured_at = datetime(2026, 7, 26, 18, 0, tzinfo=timezone.utc)

    identity = capture_index_identity(root, captured_at=captured_at).to_dict()

    normalized = "https://github.com/Example-Org-Dev/code-search"
    repository_id = hashlib.sha256(
        f"remote:{normalized}".encode("utf-8")
    ).hexdigest()
    source_revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_generation = hashlib.sha256(
        (
            repository_id
            + "\0"
            + source_revision
            + "\0"
            + "clean"
        ).encode("utf-8")
    ).hexdigest()

    assert identity == {
        "schema_version": 1,
        "repository_id": repository_id,
        "checkout_id": hashlib.sha256(
            f"path:{root.resolve().as_posix()}".encode("utf-8")
        ).hexdigest(),
        "source_revision": source_revision,
        "dirty_fingerprint": "clean",
        "index_generation": expected_generation,
        "captured_at": "2026-07-26T18:00:00Z",
    }


def test_dirty_capture_is_stable_until_source_changes(tmp_path: Path) -> None:
    root = _committed_repo(tmp_path)
    _git(root, "remote", "add", "origin", "https://example.test/org/repo.git")
    (root / "tracked.txt").write_text("worktree change\n", encoding="utf-8")
    (root / "untracked.txt").write_bytes(b"untracked\x00bytes\n")

    first = capture_index_identity(root)
    repeated = capture_index_identity(root)

    assert first.dirty_fingerprint != "clean"
    assert repeated.dirty_fingerprint == first.dirty_fingerprint
    assert repeated.index_generation == first.index_generation

    (root / "untracked.txt").write_bytes(b"changed untracked bytes\n")

    changed = capture_index_identity(root)
    assert changed.dirty_fingerprint != first.dirty_fingerprint
    assert changed.index_generation != first.index_generation


def test_subdirectory_capture_hashes_untracked_content_from_repository_root(
    tmp_path: Path,
) -> None:
    root = _committed_repo(tmp_path)
    indexed_root = root / "subdir"
    indexed_root.mkdir()
    untracked_source = indexed_root / "new.py"
    untracked_source.write_text("VALUE = 1\n", encoding="utf-8")

    first = capture_index_identity(indexed_root)
    untracked_source.write_text("VALUE = 200\n", encoding="utf-8")
    changed = capture_index_identity(indexed_root)

    assert changed.dirty_fingerprint != first.dirty_fingerprint
    assert changed.index_generation != first.index_generation


def test_dirty_submodule_tracked_content_changes_identity(
    tmp_path: Path,
) -> None:
    submodule_container = tmp_path / "submodule"
    submodule_container.mkdir()
    submodule_origin = _committed_repo(submodule_container)
    parent_container = tmp_path / "parent"
    parent_container.mkdir()
    parent = _committed_repo(parent_container)
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_origin),
        "vendor/dependency",
    )
    _git(parent, "commit", "-qam", "add dependency")
    dependency_source = parent / "vendor" / "dependency" / "tracked.txt"

    dependency_source.write_text("first dirty contents\n", encoding="utf-8")
    first_status = subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    first_diff = subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        ],
        check=True,
        capture_output=True,
    ).stdout
    first = capture_index_identity(parent)

    dependency_source.write_text("second dirty contents\n", encoding="utf-8")
    second_status = subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    second_diff = subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        ],
        check=True,
        capture_output=True,
    ).stdout
    changed = capture_index_identity(parent)

    assert first.source_revision == changed.source_revision
    assert first_status == second_status
    assert first_diff == second_diff
    assert changed.dirty_fingerprint != first.dirty_fingerprint
    assert changed.index_generation != first.index_generation


def test_nested_dirty_submodule_tracked_content_changes_identity(
    tmp_path: Path,
) -> None:
    leaf_container = tmp_path / "leaf"
    leaf_container.mkdir()
    leaf_origin = _committed_repo(leaf_container)
    middle_container = tmp_path / "middle"
    middle_container.mkdir()
    middle_origin = _committed_repo(middle_container)
    _git(
        middle_origin,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf_origin),
        "nested/leaf",
    )
    _git(middle_origin, "commit", "-qam", "add nested dependency")
    parent_container = tmp_path / "parent"
    parent_container.mkdir()
    parent = _committed_repo(parent_container)
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(middle_origin),
        "vendor/middle",
    )
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    _git(parent, "commit", "-qam", "add dependency")
    leaf_source = (
        parent
        / "vendor"
        / "middle"
        / "nested"
        / "leaf"
        / "tracked.txt"
    )

    leaf_source.write_text("first nested dirty contents\n", encoding="utf-8")
    first = capture_index_identity(parent)
    leaf_source.write_text("second nested dirty contents\n", encoding="utf-8")
    changed = capture_index_identity(parent)

    assert first.source_revision == changed.source_revision
    assert changed.dirty_fingerprint != first.dirty_fingerprint
    assert changed.index_generation != first.index_generation


def test_initialized_dirty_submodule_inspection_failure_is_fatal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    submodule_container = tmp_path / "submodule"
    submodule_container.mkdir()
    submodule_origin = _committed_repo(submodule_container)
    parent_container = tmp_path / "parent"
    parent_container.mkdir()
    parent = _committed_repo(parent_container)
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_origin),
        "vendor/dependency",
    )
    _git(parent, "commit", "-qam", "add dependency")
    submodule_root = (parent / "vendor" / "dependency").resolve()
    (submodule_root / "tracked.txt").write_text(
        "dirty contents\n",
        encoding="utf-8",
    )
    real_git = index_identity_module._git

    def fail_submodule_revision(
        captured_root: Path,
        *args: str,
        input_data: bytes | None = None,
    ) -> bytes:
        if (
            captured_root.resolve() == submodule_root
            and args == ("rev-parse", "--verify", "HEAD")
        ):
            raise IdentityCaptureError(
                "synthetic initialized submodule inspection failure"
            )
        return real_git(
            captured_root,
            *args,
            input_data=input_data,
        )

    monkeypatch.setattr(
        index_identity_module,
        "_git",
        fail_submodule_revision,
    )

    with pytest.raises(
        IdentityCaptureError,
        match="initialized submodule inspection failure",
    ):
        capture_index_identity(parent)


@pytest.mark.parametrize(
    "vector",
    _SHARED_VECTORS["submodule_framing"],
)
def test_submodule_framing_matches_shared_cross_engine_vectors(
    vector: dict[str, str],
) -> None:
    payload = index_identity_module._submodule_payload(
        bytes.fromhex(vector["relative_path_hex"]),
        vector["expected_object_id"].encode("ascii"),
        vector["current_object_id"].encode("ascii"),
        vector["nested_dirty_fingerprint"],
    )

    assert payload.hex() == vector["payload_hex"]
    digest = hashlib.sha256()
    digest.update(index_identity_module._frame("STATUS", b""))
    digest.update(index_identity_module._frame("WORKTREE_DIFF", b""))
    digest.update(index_identity_module._frame("CACHED_DIFF", b""))
    digest.update(index_identity_module._frame("SUBMODULE", payload))
    assert digest.hexdigest() == vector["dirty_fingerprint"]


@pytest.mark.parametrize(
    "vector",
    _SHARED_VECTORS["origin_normalization"],
    ids=lambda vector: vector["input"],
)
def test_remote_normalization_matches_shared_cross_engine_vectors(
    vector: dict[str, str],
) -> None:
    assert normalize_remote_url(vector["input"]) == vector["normalized"]


@pytest.mark.parametrize(
    "vector",
    _SHARED_VECTORS["index_generation"],
    ids=lambda vector: vector["source_revision"],
)
def test_index_generation_matches_shared_cross_engine_vectors(
    vector: dict[str, str],
) -> None:
    generation = derive_index_generation(
        repository_id=vector["repository_id"],
        source_revision=vector["source_revision"],
        dirty_fingerprint=vector["dirty_fingerprint"],
    )

    assert generation == vector["index_generation"]


def test_initialized_unborn_repo_uses_path_identity_and_empty_tree_diff(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unborn"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "first.txt").write_text("first contents\n", encoding="utf-8")

    identity = capture_index_identity(root)

    assert identity.source_revision == "unborn"
    assert identity.repository_id == hashlib.sha256(
        f"path:{root.resolve().as_posix()}".encode("utf-8")
    ).hexdigest()
    assert identity.dirty_fingerprint != "clean"
    assert capture_index_identity(root).index_generation == identity.index_generation


def test_initialized_unborn_sha256_repo_uses_native_empty_tree_diff(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unborn-sha256"
    root.mkdir()
    initialized = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "init",
            "-q",
            "--object-format=sha256",
        ],
        check=False,
        capture_output=True,
    )
    if initialized.returncode != 0:
        pytest.skip("Git SHA-256 repositories are unavailable")
    (root / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(root, "add", "staged.txt")
    (root / "staged.txt").write_text("worktree\n", encoding="utf-8")

    empty_tree = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "hash-object",
            "-t",
            "tree",
            "--stdin",
        ],
        check=True,
        input=b"",
        capture_output=True,
    ).stdout.decode("ascii").strip()
    assert len(empty_tree) == 64
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    expected_fingerprint = index_identity_module._dirty_fingerprint(
        root,
        status,
        empty_tree,
    )

    identity = capture_index_identity(root)

    assert identity.source_revision == "unborn"
    assert identity.dirty_fingerprint == expected_fingerprint
    assert (
        validate_index_identity_dict(identity.to_dict()).index_generation
        == identity.index_generation
    )


def test_blank_origin_falls_back_to_path_repository_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _committed_repo(tmp_path)
    real_git_optional = index_identity_module._git_optional

    def blank_origin(captured_root: Path, *args: str):
        if args == ("remote", "get-url", "origin"):
            return b"  \n"
        return real_git_optional(captured_root, *args)

    monkeypatch.setattr(
        index_identity_module,
        "_git_optional",
        blank_origin,
    )

    identity = capture_index_identity(root)

    assert identity.repository_id == hashlib.sha256(
        f"path:{root.resolve().as_posix()}".encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("repository_id", ""),
        ("checkout_id", "B" * 64),
        ("source_revision", "not-a-git-object-id"),
        ("dirty_fingerprint", "dirty"),
        ("index_generation", "0" * 64),
        ("captured_at", "2026-07-26T18:00:00-05:00"),
    ],
)
def test_persisted_identity_validation_rejects_corruption(
    field: str,
    value: object,
) -> None:
    repository_id = "a" * 64
    source_revision = "c" * 40
    dirty_fingerprint = "clean"
    valid = {
        "schema_version": 1,
        "repository_id": repository_id,
        "checkout_id": "b" * 64,
        "source_revision": source_revision,
        "dirty_fingerprint": dirty_fingerprint,
        "index_generation": derive_index_generation(
            repository_id=repository_id,
            source_revision=source_revision,
            dirty_fingerprint=dirty_fingerprint,
        ),
        "captured_at": "2026-07-26T18:00:00Z",
    }
    valid[field] = value

    with pytest.raises(ValueError, match=field):
        validate_index_identity_dict(valid)


def test_identity_mismatch_reports_checkout_even_when_generation_matches() -> None:
    first = index_identity_module.IndexIdentity(
        schema_version=1,
        repository_id="a" * 64,
        checkout_id="b" * 64,
        source_revision="c" * 40,
        dirty_fingerprint="clean",
        index_generation="d" * 64,
        captured_at="2026-07-26T18:00:00Z",
    )
    retargeted = index_identity_module.IndexIdentity(
        **{
            **first.to_dict(),
            "checkout_id": "e" * 64,
            "captured_at": "2026-07-26T18:01:00Z",
        }
    )

    assert identity_mismatch_fields(first, retargeted) == ["checkout_id"]


def test_git_capture_forces_the_cross_engine_c_locale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _committed_repo(tmp_path)
    real_run = subprocess.run
    observed_locales: list[str | None] = []

    def recording_run(*args, **kwargs):
        observed_locales.append(kwargs.get("env", {}).get("LC_ALL"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(index_identity_module.subprocess, "run", recording_run)

    capture_index_identity(root)

    assert observed_locales
    assert set(observed_locales) == {"C"}
