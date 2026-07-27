"""Fail-closed contracts for external benchmark artifact validation."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from search.index_identity import derive_index_generation

MODELS = ("voyage-4-large", "voyage-code-3")


def _evidence_rows(
    ranking: list[str],
    *,
    score: float,
) -> list[dict[str, object]]:
    rows = []
    for index, path in enumerate(ranking, 1):
        lines = f"{index}-{index + 1}"
        name = f"result_{index}"
        rows.append(
            {
                "file": path,
                "lines": lines,
                "kind": "function",
                "score": score,
                "chunk_id": f"{path}:{lines}:function:{name}",
                "name": name,
            }
        )
    return rows


def _write_complete_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    golden_path = tmp_path / "golden.json"
    qrels_path = tmp_path / "qrels_graded.json"
    golden_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "find auth",
                    "expected_files": ["auth.py"],
                },
                {
                    "query_id": "q2",
                    "query": "find retry",
                    "expected_files": ["retry.py"],
                },
            ]
        ),
        encoding="utf-8",
    )
    qrels_path.write_text(
        json.dumps(
            {
                "q1": {"auth.py": 1.0},
                "q2": {"retry.py": 1.0},
            }
        ),
        encoding="utf-8",
    )
    q1_ranking = ["auth.py", *[f"q1-decoy-{i}.py" for i in range(9)]]
    q2_ranking = [f"q2-decoy-{i}.py" for i in range(10)]
    repository_id = "a" * 64
    source_revision = "b" * 40
    index_generation = derive_index_generation(
        repository_id=repository_id,
        source_revision=source_revision,
        dirty_fingerprint="clean",
    )
    for model_index, model in enumerate(MODELS, 1):
        (tmp_path / f"results_{model}.json").write_text(
            json.dumps(
                {
                    "label": model,
                    "retrieval_k": 10,
                    "rankings_underfilled": 0,
                    "hr_1": 0.5,
                    "hr_5": 0.5,
                    "hr_k": 0.5,
                    "mrr": 0.5,
                    "avg_latency_ms": 15.0,
                    "categories": {
                        "unknown": {
                            "hr_1": 0.5,
                            "hr_5": 0.5,
                            "mrr": 0.5,
                        }
                    },
                    "effective_identity": {
                        "embedding_provider": "voyage",
                        "embedding_model": model,
                        "embedding_dimension": 1024,
                        "content_mode": "code",
                        "input_type_enabled": True,
                        "pipeline_version": str(model_index) * 16,
                        "index_identity_status": "ready",
                        "source_identity": {
                            "repository_id": repository_id,
                            "source_revision": source_revision,
                            "dirty_fingerprint": "clean",
                            "index_generation": index_generation,
                        },
                        "index_epoch_id": (
                            f"2026-07-27T12-00-0{model_index}-deadbeef"
                        ),
                        "manifest_freshness": "fresh",
                        "production_reranker_mode": "off",
                        "manual_voyage_reranker_enabled": False,
                    },
                    "external_metrics": {
                        "queries_scored": 2,
                        "ndcg@10": 0.5,
                        "recall@10": 0.5,
                    },
                    "per_query": [
                        {
                            "query_id": "q1",
                            "query": "find auth",
                            "expected_files": ["auth.py"],
                            "category": "unknown",
                            "query_class": None,
                            "qrels_key": "q1",
                            "top_k_files": q1_ranking,
                            "top_k_rows": _evidence_rows(
                                q1_ranking,
                                score=1.0,
                            ),
                            "returned_document_count": 10,
                            "ranking_underfilled": False,
                            "hit_rank": 1,
                            "hit_1": True,
                            "hit_5": True,
                            "rr": 1.0,
                            "latency_ms": 10.0,
                            "ndcg@10": 1.0,
                            "recall@10": 1.0,
                        },
                        {
                            "query_id": "q2",
                            "query": "find retry",
                            "expected_files": ["retry.py"],
                            "category": "unknown",
                            "query_class": None,
                            "qrels_key": "q2",
                            "top_k_files": q2_ranking,
                            "top_k_rows": _evidence_rows(
                                q2_ranking,
                                score=0.0,
                            ),
                            "returned_document_count": 10,
                            "ranking_underfilled": False,
                            "hit_rank": None,
                            "hit_1": False,
                            "hit_5": False,
                            "rr": 0.0,
                            "latency_ms": 20.0,
                            "ndcg@10": 0.0,
                            "recall@10": 0.0,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    return golden_path, qrels_path, tmp_path


@pytest.mark.parametrize(
    "tampered_field",
    ["query", "expected_files", "top_k_rows"],
)
def test_validator_binds_query_evidence_to_golden_inputs(
    tmp_path: Path,
    tampered_field: str,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    row = invalid["per_query"][0]
    if tampered_field == "query":
        row["query"] = "tampered query"
    elif tampered_field == "expected_files":
        row["expected_files"] = ["q1-decoy-0.py"]
    else:
        row["top_k_rows"][0]["file"] = "q1-decoy-0.py"
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match=tampered_field):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_rejects_golden_positive_qrels_contradiction(
    tmp_path: Path,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    qrels["q1"] = {
        "auth.py": 0.0,
        "q1-decoy-0.py": 1.0,
    }
    qrels_path.write_text(json.dumps(qrels), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"expected_files.*positive qrels",
    ):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("score", None),
        ("score", float("nan")),
        ("score", -0.01),
        ("score", 1.01),
        ("lines", None),
        ("lines", "20-10"),
        ("lines", "0-2"),
        ("kind", None),
        ("kind", ""),
        ("chunk_id", None),
        ("chunk_id", "forged-chunk-identity"),
        ("name", None),
        ("chunk_type", "legacy-field"),
    ],
)
def test_validator_rejects_impossible_top_k_row_evidence(
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    invalid["per_query"][0]["top_k_rows"][0][field] = tampered_value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="top_k_rows"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("returned_document_count", 9),
        ("ranking_underfilled", True),
        ("hit_rank", 2),
        ("hit_1", False),
        ("hit_5", False),
        ("rr", 0.5),
        ("category", "tampered"),
        ("query_class", "tampered"),
    ],
)
def test_validator_recomputes_deterministic_query_fields(
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    invalid["per_query"][0][field] = tampered_value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("rankings_underfilled", 1),
        ("hr_1", 0.25),
        ("hr_5", 0.25),
        ("hr_k", 0.25),
        ("mrr", 0.25),
        (
            "categories",
            {"unknown": {"hr_1": 0.25, "hr_5": 0.5, "mrr": 0.5}},
        ),
    ],
)
def test_validator_recomputes_deterministic_run_fields(
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    invalid[field] = tampered_value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("scope", "tampered_value"),
    [
        ("row", -0.01),
        ("row", float("nan")),
        ("row", float("inf")),
        ("row", "10.0"),
        ("row", True),
        ("aggregate", -0.01),
        ("aggregate", float("nan")),
        ("aggregate", float("inf")),
        ("aggregate", "15.0"),
        ("aggregate", True),
        ("aggregate", 16.0),
    ],
)
def test_validator_requires_nonnegative_coherent_latency(
    tmp_path: Path,
    scope: str,
    tampered_value: object,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    if scope == "row":
        invalid["per_query"][0]["latency_ms"] = tampered_value
    else:
        invalid["avg_latency_ms"] = tampered_value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="latency"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("path", "value", "remove"),
    [
        ((), None, True),
        (("embedding_provider",), "openai", False),
        (("embedding_model",), "voyage-code-3", False),
        (("embedding_dimension",), 512, False),
        (("content_mode",), "docs", False),
        (("input_type_enabled",), False, False),
        (("index_identity_status",), "error", False),
        (("manifest_freshness",), "stale_using_prior_epoch", False),
        (("production_reranker_mode",), "sonnet", False),
        (("manual_voyage_reranker_enabled",), True, False),
    ],
)
def test_validator_requires_expected_effective_runtime_identity(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    remove: bool,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    if not path:
        invalid.pop("effective_identity")
    elif remove:
        invalid["effective_identity"].pop(path[0])
    else:
        invalid["effective_identity"][path[0]] = value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="effective"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_rejects_same_model_under_other_label(
    tmp_path: Path,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-code-3.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    invalid["effective_identity"]["embedding_model"] = "voyage-4-large"
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="effective.*model"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    "copied_or_unbound",
    ["pipeline", "epoch", "source"],
)
def test_validator_binds_distinct_arms_to_one_shared_source(
    tmp_path: Path,
    copied_or_unbound: str,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    large = json.loads(
        (results_dir / "results_voyage-4-large.json").read_text(
            encoding="utf-8"
        )
    )
    code_path = results_dir / "results_voyage-code-3.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))
    if copied_or_unbound == "pipeline":
        code["effective_identity"]["pipeline_version"] = (
            large["effective_identity"]["pipeline_version"]
        )
    elif copied_or_unbound == "epoch":
        code["effective_identity"]["index_epoch_id"] = (
            large["effective_identity"]["index_epoch_id"]
        )
    else:
        source = code["effective_identity"]["source_identity"]
        source["source_revision"] = "c" * 40
        source["index_generation"] = derive_index_generation(
            repository_id=source["repository_id"],
            source_revision=source["source_revision"],
            dirty_fingerprint=source["dirty_fingerprint"],
        )
    code_path.write_text(json.dumps(code), encoding="utf-8")

    with pytest.raises(ValueError, match=copied_or_unbound):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_requires_both_complete_query_id_sets(
    tmp_path: Path,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    validate_benchmark_outputs(golden_path, qrels_path, results_dir)

    result_path = results_dir / "results_voyage-4-large.json"
    complete = json.loads(result_path.read_text(encoding="utf-8"))
    invalid_runs = {
        "partial": {
            **complete,
            "external_metrics": {
                **complete["external_metrics"],
                "queries_scored": 1,
            },
            "per_query": complete["per_query"][:1],
        },
        "duplicate": {
            **complete,
            "per_query": [
                complete["per_query"][0],
                copy.deepcopy(complete["per_query"][0]),
            ],
        },
        "unknown": {
            **complete,
            "per_query": [
                complete["per_query"][0],
                {
                    **complete["per_query"][1],
                    "query_id": "q3",
                    "qrels_key": "q3",
                },
            ],
        },
        "empty ranking": {
            **complete,
            "per_query": [
                {
                    **complete["per_query"][0],
                    "top_k_files": [],
                },
                complete["per_query"][1],
            ],
        },
        "declared count": {
            **complete,
            "external_metrics": {
                **complete["external_metrics"],
                "queries_scored": 1,
            },
        },
    }
    for label, invalid in invalid_runs.items():
        result_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match=label):
            validate_benchmark_outputs(golden_path, qrels_path, results_dir)

    result_path.write_text(json.dumps(complete), encoding="utf-8")
    qrels_path.write_text(
        json.dumps({"q1": {"auth.py": 1.0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="golden/qrels query IDs"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_requires_both_expected_model_files(tmp_path: Path) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    (results_dir / "results_voyage-code-3.json").unlink()

    with pytest.raises(ValueError, match="voyage-code-3"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_requires_finite_aggregate_metrics(tmp_path: Path) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    complete = json.loads(result_path.read_text(encoding="utf-8"))
    invalid_metrics = [
        ("ndcg@10", None, True),
        ("ndcg@10", "0.75", False),
        ("ndcg@10", -0.01, False),
        ("ndcg@10", 1.01, False),
        ("ndcg@10", float("nan"), False),
        ("recall@10", float("inf"), False),
        ("recall@10", True, False),
    ]

    for metric, value, remove in invalid_metrics:
        invalid = copy.deepcopy(complete)
        if remove:
            invalid["external_metrics"].pop(metric)
        else:
            invalid["external_metrics"][metric] = value
        result_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match=metric):
            validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("retrieval_k", "remove"),
    [
        (5, False),
        (11, False),
        ("10", False),
        (True, False),
        (None, True),
    ],
)
def test_validator_binds_external_cutoff_to_ten(
    tmp_path: Path,
    retrieval_k: object,
    remove: bool,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    if remove:
        invalid.pop("retrieval_k")
    else:
        invalid["retrieval_k"] = retrieval_k
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="retrieval_k"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize("ranking_failure", ["underfilled", "duplicate"])
def test_validator_requires_ten_unique_ranked_documents_per_query(
    tmp_path: Path,
    ranking_failure: str,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    ranking = invalid["per_query"][0]["top_k_files"]
    if ranking_failure == "underfilled":
        invalid["per_query"][0]["top_k_files"] = ranking[:9]
    else:
        invalid["per_query"][0]["top_k_files"][-1] = ranking[0]
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="10 unique"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("scope", "metric", "value"),
    [
        ("per_query", "ndcg@10", -0.01),
        ("per_query", "ndcg@10", 1.01),
        ("per_query", "recall@10", float("nan")),
        ("per_query", "recall@10", float("inf")),
        ("per_query", "recall@10", True),
        ("per_query", "recall@10", "1.0"),
        ("aggregate", "recall@10", -0.01),
        ("aggregate", "recall@10", 1.01),
    ],
)
def test_validator_rejects_metrics_outside_finite_unit_interval(
    tmp_path: Path,
    scope: str,
    metric: str,
    value: object,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    if scope == "per_query":
        invalid["per_query"][0][metric] = value
    else:
        invalid["external_metrics"][metric] = value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match=metric):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


@pytest.mark.parametrize(
    ("scope", "metric", "value"),
    [
        ("per_query", "ndcg@10", 0.5),
        ("per_query", "recall@10", 0.5),
        ("aggregate", "ndcg@10", 0.75),
        ("aggregate", "recall@10", 0.75),
    ],
)
def test_validator_recomputes_metrics_from_rankings_and_qrels(
    tmp_path: Path,
    scope: str,
    metric: str,
    value: float,
) -> None:
    from bench.research.validate_external_benchmark import (
        validate_benchmark_outputs,
    )

    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    result_path = results_dir / "results_voyage-4-large.json"
    invalid = json.loads(result_path.read_text(encoding="utf-8"))
    if scope == "per_query":
        invalid["per_query"][0][metric] = value
    else:
        invalid["external_metrics"][metric] = value
    result_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="recomputed"):
        validate_benchmark_outputs(golden_path, qrels_path, results_dir)


def test_validator_cli_reports_success_and_failure(tmp_path: Path) -> None:
    golden_path, qrels_path, results_dir = _write_complete_artifacts(tmp_path)
    script = (
        Path(__file__).resolve().parents[2]
        / "bench"
        / "research"
        / "validate_external_benchmark.py"
    )
    command = [
        sys.executable,
        str(script),
        "--golden",
        str(golden_path),
        "--qrels",
        str(qrels_path),
        "--results-dir",
        str(results_dir),
    ]

    valid = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    assert "external benchmark artifacts: PASS" in valid.stdout

    broken_path = results_dir / "results_voyage-code-3.json"
    broken = json.loads(broken_path.read_text(encoding="utf-8"))
    broken["external_metrics"].pop("recall@10")
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    invalid = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 1
    assert "recall@10" in invalid.stderr
