"""Regression + property tests for overlapping-emission handling in
merge_file_chunks (2026-06-10 V&V session).

Production chunkers emit nested/overlapping ranges (a class chunk plus its
method chunks — multi-granularity by design). Two merge-pass bugs duplicated
content under that shape:

1. REWIND: a nested chunk reset `last_end_line` below its parent's end, so
   the trailing-gap logic re-emitted already-covered parent lines as phantom
   `module_level` chunks.
2. HOLE-SPANNING: merged-group content is rebuilt as one contiguous source
   span; a group of method segments with a hole at the parent's exclusive
   region (e.g., a trailing class attr after the last method) absorbed that
   hole's lines, duplicating them into a second chunk.

Fixes: monotonic coverage (`last_end_line = max(...)`), group break on
contentful holes, and group end = max segment end.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chunking.chunk_merging import merge_file_chunks
from chunking.code_chunk import CodeChunk
from chunking.multi_language_chunker import MultiLanguageChunker


def _mk(source: str, name, typ, start, end):
    lines = source.split("\n")[start - 1:end]
    return CodeChunk(
        content="\n".join(lines), chunk_type=typ, start_line=start,
        end_line=end, file_path="/t.py", relative_path="t.py",
        folder_structure=[], name=name,
    )


def _line_multiplicity(chunks, source: str, needle: str) -> int:
    return sum(c.content.count(needle) for c in chunks)


def test_rewind_does_not_fabricate_phantom_gap():
    """Nested chunk after its parent must not re-emit covered parent lines."""
    source = (
        "class Outer:\n    x = 1\n    def method(self):\n"
        "        return 2\n    y = 3\n    z = 4"
    )
    chunks = [_mk(source, "Outer", "class", 1, 6),
              _mk(source, "method", "method", 3, 4)]
    # Tiny budget: nothing merges, so the trailing-gap path is exercised.
    merged = merge_file_chunks(chunks, source, "/t.py", "t.py", [], max_nws=10)
    assert _line_multiplicity(merged, source, "y = 3") == 1
    assert _line_multiplicity(merged, source, "z = 4") == 1
    # No phantom module_level chunk covering class-body lines.
    for c in merged:
        if c.chunk_type == "module_level":
            assert "y = 3" not in c.content


def test_group_does_not_span_contentful_hole():
    """Method segments must not absorb the parent's exclusive trailing line."""
    source = "\n".join([
        "class C:",            # 1   (class chunk 1-5)
        "    def m1(self):",   # 2   (method chunk 2-3)
        "        return 1",    # 3
        "    def m2(self):",   # 4   (method chunk 4-4, hole at 5)
        "    ATTR = 'alpha'",  # 5   class-exclusive line
        "",                    # 6
        "TRAILER = 'beta'",    # 7
    ])
    chunks = [
        _mk(source, "C", "class", 1, 5),
        _mk(source, "m1", "method", 2, 3),
        _mk(source, "m2", "method", 4, 4),
    ]
    merged = merge_file_chunks(chunks, source, "/t.py", "t.py", [], max_nws=60)
    assert _line_multiplicity(merged, source, "ATTR = 'alpha'") <= \
        _line_multiplicity(chunks, source, "ATTR = 'alpha'"), (
        "merge must not INCREASE a line's multiplicity beyond the input's"
    )
    assert _line_multiplicity(merged, source, "TRAILER = 'beta'") == 1


def test_group_end_uses_max_not_last():
    """A nested chunk sorting after its parent must not shrink the parent.

    Written against the pre-#229 flat merge, where class + nested method
    packed into ONE group and the regression was the group end regressing
    to the nested segment's end (dropping the trailing class attribute).
    The containment-aware merge (#229) deliberately preserves dual
    granularity — class chunk and method chunk both survive — so the same
    invariant is now asserted on the class chunk directly: full span, the
    trailing attribute intact, the method un-flattened, and no phantom
    gap chunk fabricated from container-interior lines.
    """
    source = "class C:\n    a = 1\n    def m(self):\n        return 1\n    b = 2"
    chunks = [_mk(source, "C", "class", 1, 5), _mk(source, "m", "method", 3, 4)]
    merged = merge_file_chunks(chunks, source, "/t.py", "t.py", [], max_nws=10_000)

    by_type = {c.chunk_type: c for c in merged}
    assert set(by_type) == {"class", "method"}, (
        f"expected dual granularity, got "
        f"{[(c.chunk_type, c.start_line, c.end_line) for c in merged]}"
    )
    klass = by_type["class"]
    assert (klass.start_line, klass.end_line) == (1, 5)
    assert "b = 2" in klass.content  # trailing attr stays in the container
    assert by_type["method"].content == "    def m(self):\n        return 1"


def test_real_chunker_no_duplication_on_trailing_class_attr():
    """End-to-end with the production Python chunker: the original repro."""
    body = "\n".join(
        f"    def method_{i}(self):\n"
        f"        \"\"\"Docstring for method {i} with some padding text.\"\"\"\n"
        f"        return {i} * 31337 + len('padding padding padding pad')"
        for i in range(12)
    )
    src = (
        "class BigService:\n    CONSTANT = 1\n" + body +
        "\n    TRAILING_CLASS_ATTR = 'sentinel_alpha'\n\n"
        "MODULE_TRAILER = 'sentinel_beta'\n"
    )
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "svc.py"
        f.write_text(src)
        chunks = MultiLanguageChunker(root_path=td).chunk_file(str(f))
    for sentinel in ("sentinel_alpha", "sentinel_beta", "CONSTANT = 1"):
        assert sum(c.content.count(sentinel) for c in chunks) == 1, sentinel


# ---------------------------------------------------------------------------
# Property test: merge never increases per-line multiplicity
# ---------------------------------------------------------------------------

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402


@st.composite
def _layouts(draw):
    """Random source + a chunk layout that may include nested ranges."""
    n_lines = draw(st.integers(min_value=4, max_value=40))
    # Unique, greppable content per line; some lines blank.
    lines = [
        "" if draw(st.booleans()) and i % 5 == 0 else f"line_{i}_token"
        for i in range(n_lines)
    ]
    source = "\n".join(lines)
    n_chunks = draw(st.integers(min_value=1, max_value=6))
    chunks = []
    for k in range(n_chunks):
        s = draw(st.integers(min_value=1, max_value=n_lines))
        e = draw(st.integers(min_value=s, max_value=n_lines))
        chunks.append(_mk(source, f"c{k}", "function", s, e))
    budget = draw(st.sampled_from([10, 60, 200, 1500]))
    return source, chunks, budget


@settings(max_examples=200, deadline=None)
@given(_layouts())
def test_merge_never_increases_line_multiplicity(layout):
    source, chunks, budget = layout
    merged = merge_file_chunks(
        chunks, source, "/t.py", "t.py", [], max_nws=budget
    )
    for i, line in enumerate(source.split("\n")):
        if not line.strip():
            continue
        token = f"line_{i}_token"
        in_mult = sum(c.content.count(token) for c in chunks)
        out_mult = sum(c.content.count(token) for c in merged)
        # Gap capture may introduce a line once when no chunk covered it;
        # it must never EXCEED the input multiplicity when covered.
        assert out_mult <= max(1, in_mult), (
            f"line {i + 1} multiplied {in_mult}->{out_mult} "
            f"(budget={budget}, chunks={[(c.start_line, c.end_line) for c in chunks]})"
        )
        # And no content loss: every non-blank line appears at least once.
        assert out_mult >= 1, f"line {i + 1} lost (budget={budget})"
