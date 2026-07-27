"""Contracts for public-benchmark ranked-list scoring."""

import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.research import coir_adapter
from bench.research.coir_adapter import convert
from bench.research.ndcg import score_ranked_run
from benchmarks._eval_worker import run_eval


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_score_ranked_run_rejects_zero_measurement() -> None:
    with pytest.raises(ValueError, match="at least one ranked query"):
        score_ranked_run([], {}, k=10)


def test_score_ranked_run_rejects_query_without_positive_qrels() -> None:
    run = [{"query_id": "q1", "top_k_files": []}]

    with pytest.raises(ValueError, match="positive graded qrel"):
        score_ranked_run(run, {"q1": {}}, k=10)


def test_score_ranked_run_rejects_duplicate_or_underfilled_rankings() -> None:
    qrels = {"q1": {"auth.py": 2.0, "helper.py": 1.0}}

    with pytest.raises(ValueError, match="duplicate document"):
        score_ranked_run(
            [{"query_id": "q1", "top_k_files": ["auth.py", "auth.py"]}],
            qrels,
            k=2,
        )

    with pytest.raises(ValueError, match="expected 2 unique documents"):
        score_ranked_run(
            [{"query_id": "q1", "top_k_files": ["auth.py"]}],
            qrels,
            k=2,
        )


def test_coir_download_uses_full_corpus_and_complete_qrels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qrel_rows = [
        {"query-id": "q1", "corpus-id": f"d{i}", "score": 1}
        for i in range(21)
    ]
    query_rows = [{"_id": "q1", "text": "find all relevant docs"}]
    corpus_rows = [
        {"_id": f"d{i}", "text": f"relevant {i}", "language": "python"}
        for i in range(21)
    ] + [{"_id": "decoy", "text": "irrelevant", "language": "python"}]
    calls = []

    def fake_fetch(
        _dataset: str,
        config: str,
        split: str,
        limit: int | None = None,
    ) -> list[dict]:
        calls.append((config, split, limit))
        rows = {
            "default": qrel_rows,
            "queries": query_rows,
            "corpus": corpus_rows,
        }[config]
        return list(rows if limit is None else rows[:limit])

    monkeypatch.setattr(coir_adapter, "_fetch_rows", fake_fetch)

    corpus_dir, _, qrels_path = coir_adapter.download(
        "CoIR-Retrieval/cosqa",
        "test",
        1,
        tmp_path,
    )

    graded = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert len(graded["q1"]) == 21
    assert (corpus_dir / "decoy.py").is_file()
    assert calls == [
        ("default", "test", None),
        ("queries", "queries", None),
        ("corpus", "corpus", None),
    ]


def test_coir_download_rejects_nonpositive_query_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coir_adapter,
        "_fetch_rows",
        lambda *_args, **_kwargs: [
            {"query-id": "q1", "corpus-id": "d1", "score": 1}
        ],
    )

    with pytest.raises(ValueError, match="max_queries must be positive"):
        coir_adapter.download("example/dataset", "test", 0, tmp_path)


def test_coir_fetch_fails_when_reported_split_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            io.BytesIO(
                json.dumps(
                    {
                        "num_rows_total": 2,
                        "rows": [{"row": {"_id": "first"}}],
                    }
                ).encode()
            ),
            io.BytesIO(
                json.dumps({"num_rows_total": 2, "rows": []}).encode()
            ),
        ]
    )
    monkeypatch.setattr(
        coir_adapter.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="incomplete corpus/corpus download"):
        coir_adapter._fetch_rows("example/dataset", "corpus", "corpus")


def test_coir_adapter_preserves_query_ids_when_text_is_duplicated(tmp_path: Path) -> None:
    corpus = [
        {"_id": "d1", "text": "first", "language": "python"},
        {"_id": "d2", "text": "second", "language": "python"},
    ]
    queries = [
        {"_id": "q1", "text": "find duplicate"},
        {"_id": "q2", "text": "find duplicate"},
    ]
    qrels = [
        {"query-id": "q1", "corpus-id": "d1", "score": 2},
        {"query-id": "q2", "corpus-id": "d2", "score": 1},
    ]

    _, golden_path, qrels_path = convert(corpus, queries, qrels, tmp_path)

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    graded = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert [row["query_id"] for row in golden] == ["q1", "q2"]
    assert graded == {
        "q1": {"d1.py": 2.0},
        "q2": {"d2.py": 1.0},
    }


def test_score_ranked_run_emits_graded_ndcg_and_recall() -> None:
    per_query = [
        {
            "query_id": "q1",
            "query": "find auth",
            "top_k_files": ["irrelevant.py", "auth.py", "helper.py"],
        }
    ]
    qrels = {"q1": {"auth.py": 2.0, "helper.py": 1.0}}

    scored = score_ranked_run(per_query, qrels, k=3)

    assert scored["aggregate"] == {
        "queries_scored": 1,
        "ndcg@3": scored["per_query"][0]["ndcg@3"],
        "recall@3": 1.0,
    }
    assert 0.0 < scored["aggregate"]["ndcg@3"] < 1.0
    assert scored["per_query"][0]["qrels_key"] == "q1"


def test_eval_worker_emits_unique_ranked_documents_and_query_ids() -> None:
    golden = [
        {
            "query_id": "q1",
            "query": "find auth",
            "expected_files": ["auth.py"],
        }
    ]

    requested_cutoffs = []

    def search(_query: str, search_k: int) -> list[dict]:
        requested_cutoffs.append(search_k)
        return [
            {"file": "helper.py", "score": 0.9, "chunk_id": "helper:1"},
            {"file": "helper.py", "score": 0.8, "chunk_id": "helper:2"},
            {"file": "auth.py", "score": 0.7, "chunk_id": "auth:1"},
            {"file": "other.py", "score": 0.6, "chunk_id": "other:1"},
        ]

    metrics = run_eval(
        golden,
        search,
        k=3,
        label="public",
        unique_documents=True,
    )

    assert metrics["retrieval_k"] == 3
    assert requested_cutoffs == [12]
    assert metrics["hr_k"] == 1.0
    assert metrics["per_query"][0]["query_id"] == "q1"
    assert metrics["per_query"][0]["top_k_files"] == [
        "helper.py",
        "auth.py",
        "other.py",
    ]
    assert metrics["rankings_underfilled"] == 0
    assert metrics["per_query"][0]["returned_document_count"] == 3
    assert metrics["per_query"][0]["ranking_underfilled"] is False


def test_eval_worker_surfaces_underfilled_document_rankings() -> None:
    golden = [
        {
            "query_id": "q1",
            "query": "find auth",
            "expected_files": ["auth.py"],
        }
    ]

    metrics = run_eval(
        golden,
        lambda _query, _k: [{"file": "auth.py"}, {"file": "auth.py"}],
        k=3,
        label="public",
        unique_documents=True,
    )

    assert metrics["rankings_underfilled"] == 1
    assert metrics["per_query"][0]["returned_document_count"] == 1
    assert metrics["per_query"][0]["ranking_underfilled"] is True


def test_eval_worker_documents_public_benchmark_output_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "_eval_worker.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--qrels" in result.stdout
    assert "--output" in result.stdout
    assert "--k" in result.stdout
    assert "--unique-documents" in result.stdout


def test_external_benchmark_workflow_persists_scored_rankings() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "external-benchmarks.yml"
    ).read_text(encoding="utf-8")

    assert '--qrels "$QRELS"' in workflow
    assert '--output "$RUNNER_TEMP/results_$M.json"' in workflow
    assert "--unique-documents" in workflow
    assert "--k 10" in workflow
    assert "${{ runner.temp }}/results_*.json" in workflow
