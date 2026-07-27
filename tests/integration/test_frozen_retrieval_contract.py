"""End-to-end contract for the keyless frozen retrieval merge gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.eval import check_retrieval_floor as gate
from bench.eval.build_frozen_model import build as build_frozen_model
from chunking.code_chunk import CodeChunk
from embeddings.embedder import CodeEmbedder, EffectiveEmbeddingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "bench" / "eval" / "fixtures" / "frozen-v1"


def test_frozen_fixture_indexes_and_clears_catastrophic_floors(
    tmp_path: Path,
) -> None:
    model = tmp_path / "frozen-model"
    storage = tmp_path / "storage"
    environment = os.environ.copy()
    for secret in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(secret, None)
    environment.update(
        {
            "CODE_SEARCH_STORAGE": str(storage),
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    build = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bench" / "eval" / "build_frozen_model.py"),
            "--output",
            str(model),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build.returncode == 0, (
        f"stdout={build.stdout}\nstderr={build.stderr}"
    )

    gate = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bench" / "eval" / "check_retrieval_floor.py"),
            "--mode",
            "index-and-eval",
            "--project",
            str(FIXTURE_ROOT / "corpus"),
            "--gold",
            str(FIXTURE_ROOT / "gold.json"),
            "--manifest",
            str(FIXTURE_ROOT / "manifest.json"),
            "--provider",
            "local",
            "--model",
            str(model),
            "--floor-semantic-mrr",
            "0.80",
            "--floor-semantic-hr1",
            "0.80",
            "--floor-keyword-mrr",
            "0.80",
            "--floor-keyword-hr1",
            "0.80",
            "--rerank",
            "off",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert gate.returncode == 0, (
        f"stdout={gate.stdout}\nstderr={gate.stderr}"
    )
    assert "n=5" in gate.stdout
    assert "MRR=1.0000" in gate.stdout
    assert "HR@1=1.0000" in gate.stdout


def test_real_local_provider_canonicalizes_voyage_input_type_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "frozen-model"
    build_frozen_model(model)
    monkeypatch.setenv(
        "CODE_SEARCH_STORAGE",
        str(tmp_path / "storage"),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VOYAGE_INPUT_TYPE", "on")
    configuration = EffectiveEmbeddingConfig(
        provider="local",
        model_name=str(model),
        content_mode="code",
        input_type_enabled=True,
    )
    embedder = CodeEmbedder(
        cache_dir=str(tmp_path / "model-cache"),
        configuration=configuration,
    )
    chunk = CodeChunk(
        content="def validate_token():\n    return True",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path="/source/auth.py",
        relative_path="auth.py",
        folder_structure=[],
        name="validate_token",
    )

    document = embedder.embed_chunks([chunk])[0].embedding
    query = embedder.embed_query("validate token")

    assert embedder.configuration.input_type_enabled is False
    assert document.shape == query.shape


def _index_frozen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = tmp_path / "frozen-model"
    storage = tmp_path / "storage"
    build_frozen_model(model)
    environment = {
        "CODE_SEARCH_STORAGE": str(storage),
        "EMBEDDING_PROVIDER": "local",
        "HF_HUB_OFFLINE": "1",
        "LOCAL_EMBEDDING_MODEL": str(model),
        "PYTHONHASHSEED": "0",
        "QUERY_EXPANSION": "off",
        "QUANTIZATION": "float32",
        "RERANKER": "off",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VOYAGE_INPUT_TYPE": "off",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for secret in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(secret, raising=False)

    server = gate.setup_server_for_project(
        str(FIXTURE_ROOT / "corpus"),
        provider="local",
        rerank="off",
        model=str(model),
    )
    assert gate.index_project(
        server,
        str(FIXTURE_ROOT / "corpus"),
        provider="local",
        timeout_seconds=30,
        poll_interval_seconds=0,
    )
    assert gate.switch_to_indexed_project(
        server,
        str(FIXTURE_ROOT / "corpus"),
        provider="local",
    )
    return server


@pytest.mark.parametrize("broken_arm", ["semantic", "keyword"])
def test_real_frozen_gate_rejects_each_empty_production_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken_arm: str,
) -> None:
    server = _index_frozen_fixture(tmp_path, monkeypatch)
    manager = server.get_searcher().index_manager
    if broken_arm == "semantic":
        manager.index.reset()
    else:
        manager._fts_conn.execute("DELETE FROM chunk_fts")
        manager._fts_conn.commit()

    summaries = gate.eval_required_arms(
        server,
        FIXTURE_ROOT / "gold.json",
    )
    failures = gate.required_arm_floor_failures(
        summaries,
        floors={
            "semantic": {"mrr": 0.8, "hr_1": 0.8},
            "keyword": {"mrr": 0.8, "hr_1": 0.8},
        },
    )

    healthy_arm = "keyword" if broken_arm == "semantic" else "semantic"
    assert summaries[broken_arm]["mrr"] == 0.0
    assert summaries[healthy_arm]["mrr"] == 1.0
    assert any(failure.startswith(f"{broken_arm} ") for failure in failures)
