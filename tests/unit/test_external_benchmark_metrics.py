"""Contracts for public-benchmark ranked-list scoring."""

import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.parse

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
    requested_json = []
    requested_bytes = []
    selected_calls = []

    def fake_request_json(url: str, label: str) -> dict:
        requested_json.append((url, label))
        if urllib.parse.urlparse(url).path == "/size":
            return {
                "partial": False,
                "pending": [],
                "failed": [],
                "size": {
                    "splits": [
                        {
                            "dataset": "CoIR-Retrieval/cosqa",
                            "config": "default",
                            "split": "test",
                            "num_rows": len(qrel_rows),
                        },
                        {
                            "dataset": "CoIR-Retrieval/cosqa",
                            "config": "corpus",
                            "split": "corpus",
                            "num_rows": len(corpus_rows),
                        },
                    ]
                },
            }
        return {
            "partial": False,
            "pending": [],
            "failed": [],
            "parquet_files": [
                {
                    "dataset": "CoIR-Retrieval/cosqa",
                    "config": "default",
                    "split": "test",
                    "url": "https://huggingface.co/qrels.parquet",
                    "filename": "qrels.parquet",
                    "size": 5,
                },
                {
                    "dataset": "CoIR-Retrieval/cosqa",
                    "config": "corpus",
                    "split": "corpus",
                    "url": "https://huggingface.co/corpus.parquet",
                    "filename": "corpus.parquet",
                    "size": 6,
                },
            ],
        }

    def fake_request_bytes(url: str, label: str) -> bytes:
        requested_bytes.append((url, label))
        return {
            "https://huggingface.co/qrels.parquet": b"qrels",
            "https://huggingface.co/corpus.parquet": b"corpus",
        }[url]

    def fake_parse_parquet(payload: bytes, _label: str) -> list[dict]:
        return {
            b"qrels": qrel_rows,
            b"corpus": corpus_rows,
        }[payload]

    def forbid_row_scan(*_args, **_kwargs):
        pytest.fail("complete CoIR splits must use the Parquet export")

    def fake_fetch_selected(
        _dataset: str,
        config: str,
        split: str,
        column: str,
        values: list[str],
    ) -> list[dict]:
        selected_calls.append((config, split, column, values))
        return list(query_rows)

    monkeypatch.setattr(coir_adapter, "_fetch_rows", forbid_row_scan)
    monkeypatch.setattr(
        coir_adapter,
        "_request_json",
        fake_request_json,
    )
    monkeypatch.setattr(
        coir_adapter,
        "_request_bytes",
        fake_request_bytes,
        raising=False,
    )
    monkeypatch.setattr(
        coir_adapter,
        "_parse_parquet_rows",
        fake_parse_parquet,
        raising=False,
    )
    monkeypatch.setattr(
        coir_adapter,
        "_fetch_selected_rows",
        fake_fetch_selected,
    )

    corpus_dir, _, qrels_path = coir_adapter.download(
        "CoIR-Retrieval/cosqa",
        "test",
        1,
        tmp_path,
    )

    graded = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert len(graded["q1"]) == 21
    assert (corpus_dir / "decoy.py").is_file()
    assert [label for _, label in requested_json] == [
        "CoIR Parquet manifest",
        "CoIR size manifest",
        "CoIR Parquet manifest",
        "CoIR size manifest",
    ]
    assert requested_bytes == [
        ("https://huggingface.co/qrels.parquet", "default/test qrels.parquet"),
        ("https://huggingface.co/corpus.parquet", "corpus/corpus corpus.parquet"),
    ]
    assert selected_calls == [
        ("queries", "queries", "_id", ["q1"]),
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


def test_coir_fetch_retries_rate_limit_using_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps = []

    def fake_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                "https://datasets-server.example/rows",
                429,
                "Too Many Requests",
                {"Retry-After": "2"},
                io.BytesIO(b"rate limited"),
            )
        return io.BytesIO(
            json.dumps(
                {
                    "num_rows_total": 1,
                    "rows": [{"row": {"_id": "q1"}}],
                }
            ).encode()
        )

    monkeypatch.setattr(
        coir_adapter.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(coir_adapter.time, "sleep", sleeps.append)

    rows = coir_adapter._fetch_rows(
        "example/dataset",
        "queries",
        "queries",
        limit=1,
    )

    assert rows == [{"_id": "q1"}]
    assert attempts == 2
    assert sleeps == [2.0]


def test_coir_fetch_fails_closed_after_bounded_rate_limit_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps = []

    def always_rate_limited(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://datasets-server.example/rows",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b"rate limited"),
        )

    monkeypatch.setattr(
        coir_adapter.urllib.request,
        "urlopen",
        always_rate_limited,
    )
    monkeypatch.setattr(coir_adapter.time, "sleep", sleeps.append)
    monkeypatch.setattr(coir_adapter, "_MAX_REQUEST_ATTEMPTS", 3)

    with pytest.raises(
        RuntimeError,
        match=r"queries/queries request failed with HTTP 429 after 3 attempts",
    ):
        coir_adapter._fetch_rows(
            "example/dataset",
            "queries",
            "queries",
            limit=1,
        )

    assert attempts == 3
    assert sleeps == [0.0, 0.0]


def test_coir_selected_query_fetch_is_encoded_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls = []

    def fake_request(url: str, label: str) -> dict:
        requested_urls.append((url, label))
        return {
            "partial": False,
            "num_rows_total": 2,
            "rows": [
                {"row": {"_id": "q one", "text": "first"}},
                {"row": {"_id": "q'two", "text": "second"}},
            ],
        }

    monkeypatch.setattr(coir_adapter, "_request_json", fake_request)

    rows = coir_adapter._fetch_selected_rows(
        "example/dataset",
        "queries",
        "queries",
        "_id",
        ["q one", "q'two"],
    )

    assert [row["_id"] for row in rows] == ["q one", "q'two"]
    assert len(requested_urls) == 1
    url, label = requested_urls[0]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert label == "queries/queries filtered rows"
    assert query["dataset"] == ["example/dataset"]
    assert query["where"] == [
        '"_id"=\'q one\' OR "_id"=\'q\'\'two\'',
    ]


def test_coir_selected_query_fetch_rejects_partial_filter_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coir_adapter,
        "_request_json",
        lambda *_args: {
            "partial": True,
            "num_rows_total": 1,
            "rows": [{"row": {"_id": "q1"}}],
        },
    )

    with pytest.raises(RuntimeError, match="filter returned partial results"):
        coir_adapter._fetch_selected_rows(
            "example/dataset",
            "queries",
            "queries",
            "_id",
            ["q1"],
        )


def test_coir_parquet_fetch_rejects_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(url: str, _label: str) -> dict:
        path = urllib.parse.urlparse(url).path
        if path == "/parquet":
            return {
                "partial": False,
                "pending": [],
                "failed": [],
                "parquet_files": [
                    {
                        "dataset": "example/dataset",
                        "config": "corpus",
                        "split": "corpus",
                        "url": "https://huggingface.co/corpus.parquet",
                        "filename": "corpus.parquet",
                        "size": 4,
                    }
                ],
            }
        assert path == "/size"
        return {
            "partial": False,
            "pending": [],
            "failed": [],
            "size": {
                "splits": [
                    {
                        "dataset": "example/dataset",
                        "config": "corpus",
                        "split": "corpus",
                        "num_rows": 2,
                    }
                ]
            },
        }

    monkeypatch.setattr(coir_adapter, "_request_json", fake_request_json)
    monkeypatch.setattr(
        coir_adapter,
        "_request_bytes",
        lambda *_args: b"data",
    )
    monkeypatch.setattr(
        coir_adapter,
        "_parse_parquet_rows",
        lambda *_args: [{"_id": "only-row"}],
    )

    with pytest.raises(
        RuntimeError,
        match=r"incomplete corpus/corpus Parquet export: expected 2 rows, received 1",
    ):
        coir_adapter._fetch_parquet_rows(
            "example/dataset",
            "corpus",
            "corpus",
        )


def test_coir_parquet_fetch_rejects_untrusted_shard_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(url: str, _label: str) -> dict:
        if urllib.parse.urlparse(url).path == "/parquet":
            return {
                "partial": False,
                "pending": [],
                "failed": [],
                "parquet_files": [
                    {
                        "dataset": "example/dataset",
                        "config": "corpus",
                        "split": "corpus",
                        "url": "https://attacker.example/corpus.parquet",
                        "filename": "corpus.parquet",
                        "size": 4,
                    }
                ],
            }
        return {
            "partial": False,
            "pending": [],
            "failed": [],
            "size": {
                "splits": [
                    {
                        "dataset": "example/dataset",
                        "config": "corpus",
                        "split": "corpus",
                        "num_rows": 1,
                    }
                ]
            },
        }

    monkeypatch.setattr(coir_adapter, "_request_json", fake_request_json)
    monkeypatch.setattr(
        coir_adapter,
        "_request_bytes",
        lambda *_args: pytest.fail("untrusted URL must not be requested"),
    )

    with pytest.raises(RuntimeError, match="untrusted Parquet URL"):
        coir_adapter._fetch_parquet_rows(
            "example/dataset",
            "corpus",
            "corpus",
        )


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
    assert "python -m pip install pyarrow==25.0.0" in workflow
    assert "${{ runner.temp }}/coir_task/golden.json" in workflow
    assert "${{ runner.temp }}/coir_task/qrels_graded.json" in workflow
    assert "if-no-files-found: error" in workflow
