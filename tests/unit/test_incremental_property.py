"""Property test: incremental indexing converges to a fresh full index.

For seeded random sequences of add / modify / delete operations over a copy of
the frozen fixture corpus, the incrementally maintained index must equal a
full index built from scratch over the same tree after every step. Equality is
checked on the chunk-id set, the per-chunk content, and the set of indexed
files. The deterministic in-process embedder keeps this offline and fast.

The short form (5 seeds x 6 operations) runs in the unit suite. Set
``CODE_SEARCH_PROPERTY_LONG=1`` for 20 seeds x 15 operations.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import pytest

from chunking.multi_language_chunker import MultiLanguageChunker
from merkle.snapshot_manager import SnapshotManager
from search.incremental_indexer import IncrementalIndexer
from search.indexer import CodeIndexManager
from tests.unit.test_incremental_indexer import _FakeEmbedder

CORPUS = Path(__file__).resolve().parents[2] / "bench" / "eval" / "fixtures" / "frozen-v1" / "corpus"
LONG = os.environ.get("CODE_SEARCH_PROPERTY_LONG") == "1"

pytestmark = pytest.mark.slow


def _function_source(name: str, body_token: str) -> str:
    return (
        f"def {name}(value):\n"
        f'    """Generated for the incremental property test ({body_token})."""\n'
        f"    return value + len({body_token!r})\n\n"
    )


class _Workspace:
    def __init__(self, root: Path, rng: random.Random) -> None:
        self.root = root
        self.rng = rng
        self.counter = 0

    def python_files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.py") if ".git" not in p.parts)

    def add(self) -> str:
        self.counter += 1
        target_dir = self.rng.choice([p for p in self.root.iterdir() if p.is_dir()] or [self.root])
        path = target_dir / f"generated_{self.counter}.py"
        path.write_text(_function_source(f"generated_{self.counter}", f"tok{self.counter}"))
        return f"add {path.relative_to(self.root)}"

    def modify(self) -> str:
        files = self.python_files()
        if not files:
            return self.add()
        path = self.rng.choice(files)
        self.counter += 1
        if self.rng.random() < 0.5:
            path.write_text(path.read_text() + _function_source(f"appended_{self.counter}", f"mod{self.counter}"))
            return f"append {path.relative_to(self.root)}"
        path.write_text(_function_source(f"rewritten_{self.counter}", f"rw{self.counter}"))
        return f"rewrite {path.relative_to(self.root)}"

    def delete(self) -> str:
        files = self.python_files()
        if len(files) <= 1:
            return self.add()
        path = self.rng.choice(files)
        path.unlink()
        return f"delete {path.relative_to(self.root)}"

    def step(self) -> str:
        return self.rng.choice([self.add, self.modify, self.modify, self.delete])()


def _build_indexer(base: Path, project: Path) -> tuple[IncrementalIndexer, CodeIndexManager]:
    index_dir = base / "index"
    snapshot_dir = base / "snapshots"
    index_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    manager = CodeIndexManager(str(index_dir))
    indexer = IncrementalIndexer(
        indexer=manager,
        embedder=_FakeEmbedder(dim=8),
        chunker=MultiLanguageChunker(str(project)),
        snapshot_manager=SnapshotManager(snapshot_dir),
    )
    return indexer, manager


def _state(manager: CodeIndexManager) -> dict[str, tuple[str, str]]:
    entries = manager.get_chunk_entries()
    return {
        chunk_id: (meta.get("relative_path", ""), meta.get("full_content") or meta.get("content_preview", ""))
        for chunk_id, meta in entries
    }


def _close(manager: CodeIndexManager) -> None:
    for attr in ("_metadata_db", "_fts_conn"):
        conn = getattr(manager, attr, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort
                pass
            setattr(manager, attr, None)


def _run_sequence(tmp_path: Path, seed: int, steps: int) -> None:
    rng = random.Random(seed)
    project = tmp_path / "proj"
    shutil.copytree(CORPUS, project)
    workspace = _Workspace(project, rng)

    incremental, inc_manager = _build_indexer(tmp_path / "incremental", project)
    first = incremental.incremental_index(str(project), "proj", force_full=True)
    assert first.success, first.error

    history: list[str] = []
    for step_no in range(steps):
        history.append(workspace.step())
        result = incremental.incremental_index(str(project), "proj")
        assert result.success, f"seed={seed} step={step_no} {history[-1]}: {result.error}"

        fresh_base = tmp_path / f"fresh-{step_no}"
        fresh, fresh_manager = _build_indexer(fresh_base, project)
        full = fresh.incremental_index(str(project), "proj", force_full=True)
        assert full.success, full.error

        incremental_state = _state(inc_manager)
        full_state = _state(fresh_manager)
        _close(fresh_manager)
        shutil.rmtree(fresh_base, ignore_errors=True)

        assert set(incremental_state) == set(full_state), (
            f"seed={seed} step={step_no} history={history}\n"
            f"only_incremental={sorted(set(incremental_state) - set(full_state))[:5]}\n"
            f"only_full={sorted(set(full_state) - set(incremental_state))[:5]}"
        )
        assert incremental_state == full_state, f"seed={seed} step={step_no} content drift after {history}"
        # Chunk ids are "<relative path>:<start>-<end>:<type>[:<name>]", so the
        # indexed file set is recoverable from the ids alone.
        indexed_py = {cid.split(":", 1)[0] for cid in incremental_state if cid.split(":", 1)[0].endswith(".py")}
        assert indexed_py == {str(p.relative_to(project)) for p in workspace.python_files()}, (
            f"seed={seed} step={step_no} file set drift after {history}"
        )
    _close(inc_manager)


@pytest.mark.parametrize("seed", range(5))
def test_incremental_matches_full_index_short(tmp_path: Path, seed: int) -> None:
    _run_sequence(tmp_path, seed, steps=6)


@pytest.mark.skipif(not LONG, reason="set CODE_SEARCH_PROPERTY_LONG=1 for the long run")
@pytest.mark.parametrize("seed", range(5, 25))
def test_incremental_matches_full_index_long(tmp_path: Path, seed: int) -> None:
    _run_sequence(tmp_path, seed, steps=15)
