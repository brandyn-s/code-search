"""Phase A3 (2026-05-08) cancel-during-file-walk tests.

The pre-fix wedge: code-search MCP, spawned with cwd=$HOME, called
ensure_project_indexed → index_directory → MerkleDAG.build, which
walked ~150K files. cancel_indexing flipped the cancel flag but
nothing in the walk loop polled it. Two cancel calls returned
success without effect.

A2 (PR #142) prevents the precondition (refuses $HOME). A3 closes
the deeper gap: any long legitimate walk now polls cancel_check
at every directory boundary inside MerkleDAG.build_node and raises
IndexingCancelled within seconds.
"""

from __future__ import annotations

import pytest

from merkle.merkle_dag import MerkleDAG, IndexingCancelled


def test_merkle_walks_to_completion_when_cancel_check_returns_false(tmp_path):
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "a.py").write_text("def a(): pass\n")
    (proj / "sub").mkdir()
    (proj / "sub" / "b.py").write_text("def b(): pass\n")

    dag = MerkleDAG(str(proj), cancel_check=lambda: False)
    dag.build()
    files = dag.get_all_files()
    assert any(f.endswith("a.py") for f in files)
    assert any(f.endswith("b.py") for f in files)


def test_merkle_walks_to_completion_when_cancel_check_is_none(tmp_path):
    """Backward compat: no cancel_check → no cancel support, walk completes."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "a.py").write_text("def a(): pass\n")

    dag = MerkleDAG(str(proj))  # no cancel_check
    dag.build()
    files = dag.get_all_files()
    assert any(f.endswith("a.py") for f in files)


def test_merkle_raises_indexing_cancelled_at_first_directory_boundary(tmp_path):
    """Cancel signal observed → IndexingCancelled raised."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "a.py").write_text("def a(): pass\n")
    (proj / "sub").mkdir()
    (proj / "sub" / "b.py").write_text("def b(): pass\n")

    dag = MerkleDAG(str(proj), cancel_check=lambda: True)
    with pytest.raises(IndexingCancelled):
        dag.build()


def test_indexing_cancelled_is_subclass_of_interruptederror():
    """Existing `except InterruptedError` handlers in
    code_search_server.py / incremental_indexer.py must still catch
    the new exception class without modification."""
    assert issubclass(IndexingCancelled, InterruptedError)


def test_merkle_cancel_check_polled_repeatedly(tmp_path):
    """cancel_check is called multiple times during a deep walk —
    not just once. Verifies the polling happens at every directory
    boundary, not only at the root."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    # Three nested levels.
    deep = proj / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "x.py").write_text("def x(): pass\n")

    call_count = 0

    def counting_cancel():
        nonlocal call_count
        call_count += 1
        return False

    dag = MerkleDAG(str(proj), cancel_check=counting_cancel)
    dag.build()
    # Root + a + b + c = 4 directory boundaries minimum.
    assert call_count >= 4, (
        f"cancel_check should be polled at every directory boundary; "
        f"got {call_count} calls (expected >= 4)"
    )


def test_merkle_cancel_partial_state_dict_initialized(tmp_path):
    """Even when cancel raises immediately, dag.nodes is initialized
    to an empty dict in __init__ — accessing it after a cancelled
    build doesn't raise AttributeError."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "a.py").write_text("def a(): pass\n")

    dag = MerkleDAG(str(proj), cancel_check=lambda: True)
    with pytest.raises(IndexingCancelled):
        dag.build()
    # nodes was initialized in __init__; access must not raise.
    assert isinstance(dag.nodes, dict)
