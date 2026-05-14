"""Tests for the retrieval floor gate script (Phase α, 2026-05-14).

The script asserts MRR/HR@1 floors on PSM eval summaries (local workflow)
and indexes-and-evaluates a target project (CI mode). These tests cover
the summary-mode parsing + floor logic; index-and-eval mode is exercised
end-to-end via local invocations, not unit tests (Voyage API + indexing
required).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
GATE_SCRIPT = REPO_ROOT / "bench" / "eval" / "check_retrieval_floor.py"


def _run_gate(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the gate script and capture output."""
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else str(REPO_ROOT),
    )


def _write_summary(path: Path, *, golden_mrr: float, golden_hr1: float,
                   harvested_mrr: float, harvested_hr1: float) -> None:
    payload = {
        "provider": "voyage",
        "golden": {
            "label": "golden",
            "n": 102,
            "mrr": golden_mrr,
            "hr_1": golden_hr1,
            "hr_5": 0.85,
        },
        "harvested_labeled": {
            "label": "harvested-labeled",
            "n": 183,
            "mrr": harvested_mrr,
            "hr_1": harvested_hr1,
            "hr_5": 0.90,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_mode_passes_when_all_above_floor(tmp_path: Path) -> None:
    """Gate exits 0 when every measured metric clears its floor."""
    summary = tmp_path / "summary.json"
    _write_summary(summary, golden_mrr=0.70, golden_hr1=0.60,
                   harvested_mrr=0.80, harvested_hr1=0.70)

    result = _run_gate(
        "--mode", "summary",
        "--summary", str(summary),
        "--floor-golden-mrr", "0.62",
        "--floor-golden-hr1", "0.50",
        "--floor-harvested-mrr", "0.73",
        "--floor-harvested-hr1", "0.65",
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "PASS" in result.stdout
    assert "all floors cleared" in result.stdout


def test_summary_mode_fails_when_golden_mrr_below_floor(tmp_path: Path) -> None:
    """Gate exits 1 and reports the violation when golden MRR slips."""
    summary = tmp_path / "summary.json"
    _write_summary(summary, golden_mrr=0.55, golden_hr1=0.60,
                   harvested_mrr=0.80, harvested_hr1=0.70)

    result = _run_gate(
        "--mode", "summary",
        "--summary", str(summary),
        "--floor-golden-mrr", "0.62",
        "--floor-harvested-mrr", "0.73",
    )

    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "golden MRR 0.5500 < floor 0.6200" in result.stderr


def test_summary_mode_fails_when_harvested_hr1_below_floor(tmp_path: Path) -> None:
    """Gate detects harvested HR@1 regression specifically."""
    summary = tmp_path / "summary.json"
    _write_summary(summary, golden_mrr=0.70, golden_hr1=0.60,
                   harvested_mrr=0.80, harvested_hr1=0.50)

    result = _run_gate(
        "--mode", "summary",
        "--summary", str(summary),
        "--floor-harvested-hr1", "0.65",
    )

    assert result.returncode == 1
    assert "harvested HR@1 0.5000 < floor 0.6500" in result.stderr


def test_summary_mode_skips_unset_floors(tmp_path: Path) -> None:
    """Only floors explicitly passed are checked; absent floors are no-op."""
    summary = tmp_path / "summary.json"
    _write_summary(summary, golden_mrr=0.30, golden_hr1=0.20,
                   harvested_mrr=0.40, harvested_hr1=0.30)

    # Only golden MRR floor is set; the (low) HR@1 values are not enforced.
    result = _run_gate(
        "--mode", "summary",
        "--summary", str(summary),
        "--floor-golden-mrr", "0.25",
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_summary_mode_fails_when_summary_file_missing(tmp_path: Path) -> None:
    """Missing summary file is a clear FAIL, not a crash."""
    result = _run_gate(
        "--mode", "summary",
        "--summary", str(tmp_path / "does-not-exist.json"),
        "--floor-golden-mrr", "0.62",
    )

    assert result.returncode == 1
    assert "summary file not found" in result.stderr


def test_summary_mode_fails_when_required_field_missing(tmp_path: Path) -> None:
    """Summary missing golden.mrr produces a structured FAIL."""
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"golden": {}, "harvested_labeled": {}}),
                       encoding="utf-8")

    result = _run_gate(
        "--mode", "summary",
        "--summary", str(summary),
        "--floor-golden-mrr", "0.62",
    )

    assert result.returncode == 1
    assert "missing required fields" in result.stderr


def test_help_text_lists_both_modes() -> None:
    """The script's --help advertises both summary and index-and-eval modes."""
    result = _run_gate("--help")
    assert result.returncode == 0
    assert "summary" in result.stdout
    assert "index-and-eval" in result.stdout
