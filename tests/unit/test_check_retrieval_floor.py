"""Tests for the retrieval floor gate script (Phase α, 2026-05-14).

The script asserts MRR/HR@1 floors on PSM eval summaries (local workflow)
and indexes-and-evaluates a target project (CI mode). These tests cover
the summary-mode parsing + floor logic; index-and-eval mode is exercised
end-to-end via local invocations, not unit tests (Voyage API + indexing
required).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bench.eval import check_retrieval_floor as gate

REPO_ROOT = Path(__file__).parent.parent.parent
GATE_SCRIPT = REPO_ROOT / "bench" / "eval" / "check_retrieval_floor.py"
FROZEN_ROOT = REPO_ROOT / "bench" / "eval" / "fixtures" / "frozen-v1"


def _run_gate(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the gate script and capture output."""
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *args],
        check=False,
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


def test_frozen_fixture_has_five_queries_and_ten_corpus_files() -> None:
    manifest = json.loads(
        (FROZEN_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    gold = json.loads((FROZEN_ROOT / "gold.json").read_text(encoding="utf-8"))
    listed_corpus = {
        name
        for name in manifest["files"]
        if name.startswith("corpus/")
    }
    actual_corpus = {
        path.relative_to(FROZEN_ROOT).as_posix()
        for path in (FROZEN_ROOT / "corpus").rglob("*")
        if path.is_file()
    }

    assert len(gold) == 5
    assert len(actual_corpus) == 10
    assert listed_corpus == actual_corpus


class _PerArmSearchServer:
    def __init__(self, *, broken_arm: str) -> None:
        self.broken_arm = broken_arm
        self.calls: list[str] = []

    def search_code(self, **kwargs: object) -> str:
        search_mode = str(kwargs["search_mode"])
        self.calls.append(search_mode)
        results = (
            []
            if search_mode == self.broken_arm
            else [{"file": "target.py"}]
        )
        return json.dumps({"results": results})


def test_required_retrieval_arms_reject_malformed_gold_before_scoring(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            [
                {"query": "missing evidence", "expected_files": []},
                {
                    "query": "find target",
                    "expected_files": ["target.py"],
                },
            ]
        ),
        encoding="utf-8",
    )
    server = _PerArmSearchServer(broken_arm="never")

    with pytest.raises(
        ValueError,
        match=r"gold row 1.*expected_files",
    ):
        gate.eval_required_arms(server, gold)

    assert server.calls == []


@pytest.mark.parametrize("broken_arm", ["semantic", "keyword"])
def test_required_retrieval_arm_fails_when_independently_empty(
    tmp_path: Path,
    broken_arm: str,
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            [{"query": "find target", "expected_files": ["target.py"]}]
        ),
        encoding="utf-8",
    )
    server = _PerArmSearchServer(broken_arm=broken_arm)

    summaries = gate.eval_required_arms(server, gold)
    failures = gate.required_arm_floor_failures(
        summaries,
        floors={
            "semantic": {"mrr": 0.8, "hr_1": 0.8},
            "keyword": {"mrr": 0.8, "hr_1": 0.8},
        },
    )

    assert server.calls == ["semantic", "keyword"]
    assert summaries[broken_arm]["mrr"] == 0.0
    healthy_arm = "keyword" if broken_arm == "semantic" else "semantic"
    assert summaries[healthy_arm]["mrr"] == 1.0
    assert any(failure.startswith(f"{broken_arm} ") for failure in failures)


def test_required_retrieval_arms_reject_partial_scored_counts() -> None:
    summaries = {
        arm: {
            "n": 1,
            "loaded_count": 2,
            "scored_count": 1,
            "mrr": 1.0,
            "hr_1": 1.0,
        }
        for arm in gate.REQUIRED_RETRIEVAL_ARMS
    }

    failures = gate.required_arm_floor_failures(
        summaries,
        floors={
            arm: {"mrr": 0.8, "hr_1": 0.8}
            for arm in gate.REQUIRED_RETRIEVAL_ARMS
        },
    )

    assert "semantic scored_count 1 != loaded_count 2" in failures
    assert "keyword scored_count 1 != loaded_count 2" in failures


class _QueuedIndexServer:
    def __init__(
        self,
        progress: list[dict[str, object]],
        *,
        start_override: dict[str, object] | None = None,
        raw_start: object | None = None,
    ) -> None:
        self.progress = list(progress)
        self.index_calls: list[dict[str, object]] = []
        self.start_override = start_override or {}
        self.raw_start = raw_start

    def index_directory(self, **kwargs: object) -> str:
        self.index_calls.append(kwargs)
        if self.raw_start is not None:
            return json.dumps(self.raw_start)
        response = {
            "status": "indexing",
            "job_id": "job-123",
            "directory": kwargs["directory_path"],
            "project_name": Path(str(kwargs["directory_path"])).name,
            "provider": kwargs["provider"],
            "index_ready": False,
        }
        response.update(self.start_override)
        return json.dumps(response)

    def get_indexing_progress(self) -> str:
        return json.dumps(self.progress.pop(0))


class _SwitchServer:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def switch_project(self, **kwargs: str) -> str:
        self.calls.append(kwargs)
        return self.response


def test_switch_to_indexed_project_requires_bound_success(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    expected_info = {
        "project_name": project.name,
        "project_path": str(project.resolve()),
        "embedding_provider": "local",
    }
    invalid_responses = [
        "not-json",
        json.dumps([]),
        json.dumps({}),
        json.dumps({"success": False, "project_info": expected_info}),
        json.dumps(
            {
                "success": True,
                "project_info": {
                    **expected_info,
                    "project_name": "other",
                },
            }
        ),
        json.dumps(
            {
                "success": True,
                "project_info": {
                    **expected_info,
                    "project_path": str(tmp_path / "other"),
                },
            }
        ),
        json.dumps(
            {
                "success": True,
                "project_info": {
                    **expected_info,
                    "embedding_provider": "voyage",
                },
            }
        ),
        json.dumps(
            {
                "success": True,
                "error": "wrong index selected",
                "project_info": expected_info,
            }
        ),
    ]

    for response in invalid_responses:
        server = _SwitchServer(response)
        assert gate.switch_to_indexed_project(
            server,
            str(project),
            provider="local",
        ) is False

    server = _SwitchServer(
        json.dumps({"success": True, "project_info": expected_info})
    )
    assert gate.switch_to_indexed_project(
        server,
        str(project),
        provider="local",
    ) is True
    assert server.calls == [
        {
            "project_path": str(project.resolve()),
            "provider": "local",
        }
    ]


def test_index_project_rejects_outer_error_on_completed_progress(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    server = _QueuedIndexServer(
        [
            {
                "status": "completed",
                "job_id": "job-123",
                "directory": str(project),
                "project_name": project.name,
                "provider": "local",
                "index_ready": True,
                "error": "publication failed after result construction",
                "result": {
                    "success": True,
                    "index_ready": True,
                    "error": None,
                },
            }
        ]
    )

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_index_project_rejects_start_bound_to_another_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    server = _QueuedIndexServer(
        [
            {
                "status": "completed",
                "job_id": "job-123",
                "directory": str(other),
                "project_name": other.name,
                "provider": "local",
                "index_ready": True,
                "result": {
                    "success": True,
                    "index_ready": True,
                    "error": None,
                },
            }
        ],
        start_override={
            "directory": str(other),
            "project_name": other.name,
            "indexing_conflict": True,
            "requested_directory": str(project),
        },
    )

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_index_project_rejects_completed_progress_for_another_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    server = _QueuedIndexServer(
        [
            {
                "status": "completed",
                "job_id": "job-123",
                "directory": str(other),
                "project_name": other.name,
                "provider": "local",
                "index_ready": True,
                "result": {
                    "success": True,
                    "directory": str(other),
                    "project_name": other.name,
                    "provider": "local",
                    "index_ready": True,
                    "error": None,
                },
            }
        ]
    )

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_index_project_rejects_non_object_terminal_progress(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    server = _QueuedIndexServer([["completed"]])  # type: ignore[list-item]

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_index_project_rejects_non_object_start_response(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    server = _QueuedIndexServer([], raw_start=["indexing"])

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_index_project_rejects_malformed_terminal_error_field(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    server = _QueuedIndexServer(
        [
            {
                "status": "completed",
                "job_id": "job-123",
                "directory": str(project),
                "project_name": project.name,
                "provider": "local",
                "index_ready": True,
                "result": {
                    "success": True,
                    "directory": str(project),
                    "project_name": project.name,
                    "provider": "local",
                    "index_ready": True,
                    "error": 0,
                },
            }
        ]
    )

    assert gate.index_project(
        server,
        str(project),
        provider="local",
        timeout_seconds=1,
        poll_interval_seconds=0,
    ) is False


def test_fixture_manifest_rejects_an_unbound_gold_path(
    tmp_path: Path,
) -> None:
    alternate_gold = tmp_path / "gold.json"
    alternate_gold.write_text(
        (FROZEN_ROOT / "gold.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    verified, error = gate.verify_fixture_manifest(
        FROZEN_ROOT / "manifest.json",
        project_path=FROZEN_ROOT / "corpus",
        gold_path=alternate_gold,
    )

    assert verified is False
    assert "gold path" in error


def test_fixture_manifest_requires_exact_gold_checksum(tmp_path: Path) -> None:
    fixture = tmp_path / "frozen-v1"
    corpus = fixture / "corpus"
    corpus.mkdir(parents=True)
    source = b"def target():\n    return 'rank me'\n"
    gold = b'[{"query": "target", "expected_files": ["target.py"]}]\n'
    (corpus / "target.py").write_bytes(source)
    (fixture / "gold.json").write_bytes(gold)
    manifest = {
        "schema_version": 1,
        "files": {
            "corpus/target.py": {
                "sha256": hashlib.sha256(source).hexdigest(),
            },
        },
    }
    manifest_path = fixture / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verified, error = gate.verify_fixture_manifest(
        manifest_path,
        project_path=corpus,
        gold_path=fixture / "gold.json",
    )
    assert verified is False
    assert "gold.json" in error

    manifest["files"]["gold.json"] = {"sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified, error = gate.verify_fixture_manifest(
        manifest_path,
        project_path=corpus,
        gold_path=fixture / "gold.json",
    )
    assert verified is False
    assert "checksum mismatch for gold.json" in error
