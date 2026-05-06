"""Tests for the quantization drift monitor (Plan-2 B2).

Pin the comparison logic — the rest of the script (FAISS sampling) is
covered by smoke testing against real indexes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the script as a module by adding scripts/ to path
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monitor_quantization_drift as mqd  # type: ignore


def test_compare_no_drift():
    """Identical current and baseline → drift_count=0, all stable."""
    current = {
        "proj_a": {"avg_top_1_cosine": 0.999, "ntotal": 1000},
        "proj_b": {"avg_top_1_cosine": 0.998, "ntotal": 500},
    }
    baseline = {
        "saved_at": "2026-01-01T00:00:00",
        "projects": {
            "proj_a": {"avg_top_1_cosine": 0.999, "ntotal": 1000},
            "proj_b": {"avg_top_1_cosine": 0.998, "ntotal": 500},
        },
    }
    report = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    assert report["drift_count"] == 0
    assert all(p["status"] == "stable" for p in report["projects"])


def test_compare_flags_drift():
    """avg_top_1 dropped from 0.99 to 0.90 → drift exceeds 0.05 threshold."""
    current = {"proj_x": {"avg_top_1_cosine": 0.90, "ntotal": 1000}}
    baseline = {
        "saved_at": "2026-01-01T00:00:00",
        "projects": {
            "proj_x": {"avg_top_1_cosine": 0.99, "ntotal": 1000},
        },
    }
    report = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    assert report["drift_count"] == 1
    row = report["projects"][0]
    assert row["status"] == "drift"
    # delta is negative when current < baseline
    assert row["delta"] < 0
    assert abs(row["delta"]) > 0.05


def test_compare_threshold_boundary():
    """Delta exactly at threshold → NOT flagged (uses strict `>`)."""
    current = {"proj_x": {"avg_top_1_cosine": 0.95, "ntotal": 1000}}
    baseline = {
        "saved_at": "2026-01-01T00:00:00",
        "projects": {
            "proj_x": {"avg_top_1_cosine": 1.00, "ntotal": 1000},
        },
    }
    report = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    # Delta is exactly -0.05; abs(delta) > 0.05 is False
    assert report["drift_count"] == 0


def test_compare_marks_new_project():
    """A project in current but not baseline shows status=new."""
    current = {"proj_new": {"avg_top_1_cosine": 0.999, "ntotal": 50}}
    baseline = {"saved_at": "2026-01-01T00:00:00", "projects": {}}
    report = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    assert report["drift_count"] == 0
    assert report["projects"][0]["status"] == "new"


def test_compare_tracks_missing_projects():
    """A project in baseline but not current → listed in missing_from_current."""
    current = {}
    baseline = {
        "saved_at": "2026-01-01T00:00:00",
        "projects": {
            "deleted_proj": {"avg_top_1_cosine": 0.99, "ntotal": 100},
        },
    }
    report = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    assert report["missing_from_current"] == ["deleted_proj"]
    assert len(report["projects"]) == 0


def test_compare_threshold_respected():
    """Custom threshold of 0.01 catches drift that 0.05 would miss."""
    current = {"proj_x": {"avg_top_1_cosine": 0.97, "ntotal": 1000}}
    baseline = {
        "saved_at": "2026-01-01T00:00:00",
        "projects": {
            "proj_x": {"avg_top_1_cosine": 1.00, "ntotal": 1000},
        },
    }
    # Threshold 0.05: not drift
    r05 = mqd.compare_to_baseline(current, baseline, threshold=0.05)
    assert r05["drift_count"] == 0
    # Threshold 0.01: drift
    r01 = mqd.compare_to_baseline(current, baseline, threshold=0.01)
    assert r01["drift_count"] == 1
