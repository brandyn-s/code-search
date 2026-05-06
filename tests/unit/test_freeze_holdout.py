"""Tests for the holdout freeze script (Plan-2 C3)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench" / "research"))
sys.path.insert(0, str(REPO_ROOT))

import freeze_holdout as fh  # type: ignore


def test_verify_lock_passes_on_committed_holdout():
    """The lockfile we ship matches the committed golden_multitarget.json.

    This is the regression guard: if anyone modifies the JSON without
    re-running --update, this test fails.
    """
    err = fh.verify_lock(version="v1")
    assert err is None, f"Holdout drift detected: {err}"


def test_verify_lock_returns_message_when_target_missing(tmp_path, monkeypatch):
    """Missing target file produces a structured error, not a crash."""
    fake_root = tmp_path / "fake_repo"
    (fake_root / "bench" / "eval" / "holdout").mkdir(parents=True)
    lock = fake_root / "bench" / "eval" / "holdout" / "multitarget_v1.lock"
    lock.write_text(
        "target: benchmarks/golden_multitarget.json\n"
        "sha256: deadbeef\n"
        "version: v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(fh, "_repo_root", lambda: fake_root)
    err = fh.verify_lock(version="v1")
    assert err is not None
    assert "Target file not found" in err


def test_verify_lock_returns_message_when_lock_missing(tmp_path, monkeypatch):
    """Missing lockfile produces a structured error."""
    monkeypatch.setattr(fh, "_repo_root", lambda: tmp_path)
    err = fh.verify_lock(version="v999")
    assert err is not None
    assert "Lockfile not found" in err


def test_verify_lock_detects_drift(tmp_path, monkeypatch):
    """SHA mismatch produces a structured error with the recovery hint."""
    fake_root = tmp_path / "fake_repo"
    (fake_root / "benchmarks").mkdir(parents=True)
    (fake_root / "bench" / "eval" / "holdout").mkdir(parents=True)
    target = fake_root / "benchmarks" / "golden_multitarget.json"
    target.write_text("[]", encoding="utf-8")
    # Lockfile pins a wrong hash on purpose
    lock = fake_root / "bench" / "eval" / "holdout" / "multitarget_v1.lock"
    lock.write_text(
        "target: benchmarks/golden_multitarget.json\n"
        "sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "version: v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(fh, "_repo_root", lambda: fake_root)
    err = fh.verify_lock(version="v1")
    assert err is not None
    assert "Holdout drift detected" in err
    assert "expected sha256" in err
    assert "actual sha256" in err
    assert "promote to a new version" in err


def test_update_lock_writes_correct_hash(tmp_path, monkeypatch):
    """update_lock should compute fresh hash + write a versioned file."""
    import hashlib
    fake_root = tmp_path / "fake_repo"
    (fake_root / "benchmarks").mkdir(parents=True)
    target = fake_root / "benchmarks" / "golden_multitarget.json"
    payload = b'[{"q":"test"}]'
    target.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(fh, "_repo_root", lambda: fake_root)

    out_path = fh.update_lock("v2")
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert f"sha256: {expected_sha}" in body
    assert "version: v2" in body
    assert f"bytes: {len(payload)}" in body
