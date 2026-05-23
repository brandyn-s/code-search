"""R10: cross-boundary chunk-merge signals via `multi_chunk_merge` tag.

The chunk-merge pass (chunking/chunk_merging.py) greedily packs adjacent
segments up to MAX_CHUNK_NWS=1500. When two unrelated functions are
separated by only a small gap (whitespace, a comment), they get fused
into one chunk. Pre-R10, the merged chunk's `chunk_type` was either
`function` (if both inputs were functions — indistinguishable from a
single function chunk) or `merged` (if mixed types). The two-functions
case is the dominant bad shape and was invisible.

R10 adds a stable `multi_chunk_merge` tag to any merged chunk produced
by combining 2+ NAMED non-gap segments. This is additive: existing
downstream code ignores unknown tags; new code can filter/deboost.

Threshold rationale: a single function merged with its preceding
imports/gap is NOT a cross-boundary merge — only one semantic unit.
The tag fires only when 2+ named units are fused.
"""
from __future__ import annotations

from chunking.chunk_merging import merge_file_chunks
from chunking.code_chunk import CodeChunk


def _mk(name: str, chunk_type: str, start: int, end: int, content: str) -> CodeChunk:
    return CodeChunk(
        content=content, chunk_type=chunk_type,
        start_line=start, end_line=end,
        file_path="/t.py", relative_path="t.py", folder_structure=[],
        name=name,
    )


# ---------------------------------------------------------------------------
# The bad-shape case the tag is meant to catch
# ---------------------------------------------------------------------------

class TestMultiChunkMergeTagFires:
    """When 2+ named non-gap chunks fuse, the tag must be present so
    downstream consumers can detect cross-boundary smush."""

    def test_two_unrelated_functions_get_tagged(self):
        source = (
            "def authenticate(user):\n"
            "    return validate(user)\n"
            "\n"
            "\n"
            "def render_template(name):\n"
            "    return open(name).read()"
        )
        chunks = [
            _mk("authenticate", "function", 1, 2,
                "def authenticate(user):\n    return validate(user)"),
            _mk("render_template", "function", 5, 6,
                "def render_template(name):\n    return open(name).read()"),
        ]
        result = merge_file_chunks(chunks, source, "/t.py", "t.py", [])
        # Both inputs fit under MAX_CHUNK_NWS=1500, so they merge.
        merged = [c for c in result if c.name and " + " in c.name]
        assert len(merged) == 1, (
            f"expected 2 functions to merge into 1 chunk, got {len(result)}"
        )
        assert "multi_chunk_merge" in merged[0].tags, (
            f"merged chunk must carry multi_chunk_merge tag. "
            f"tags={merged[0].tags}"
        )

    def test_function_plus_class_get_tagged(self):
        """Mixed-type merge: same signal, in addition to the existing
        `chunk_type='merged'` for type-mismatched fuses."""
        source = (
            "class Config:\n"
            "    debug = False\n"
            "\n"
            "def main():\n"
            "    print('hi')"
        )
        chunks = [
            _mk("Config", "class", 1, 2, "class Config:\n    debug = False"),
            _mk("main", "function", 4, 5, "def main():\n    print('hi')"),
        ]
        result = merge_file_chunks(chunks, source, "/t.py", "t.py", [])
        merged = [c for c in result if c.name and " + " in c.name]
        assert len(merged) == 1
        assert "multi_chunk_merge" in merged[0].tags
        # And the existing 'merged' chunk_type still fires for mixed types
        # — additive, doesn't replace.
        assert merged[0].chunk_type == "merged"


# ---------------------------------------------------------------------------
# Cases where the tag must NOT fire
# ---------------------------------------------------------------------------

class TestMultiChunkMergeTagDoesNotOverFire:
    """The tag must NOT appear on legitimate single-chunk outcomes — those
    are not cross-boundary merges and tagging them would mislead consumers."""

    def test_single_large_function_no_merge_no_tag(self):
        content = "def big():\n" + "    x = 1\n" * 100
        chunks = [_mk("big", "function", 1, 101, content)]
        result = merge_file_chunks(chunks, content, "/t.py", "t.py", [])
        assert len(result) == 1
        assert "multi_chunk_merge" not in result[0].tags

    def test_function_merged_with_only_imports_gap_no_tag(self):
        """A single named function merged with a preceding imports gap is
        ONE semantic unit + module-level code — not a cross-boundary merge.
        named_non_gap_count is 1, so no tag."""
        source = (
            "import os\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    print(os.path.exists(sys.argv[1]))"
        )
        chunks = [
            _mk("main", "function", 4, 5,
                "def main():\n    print(os.path.exists(sys.argv[1]))"),
        ]
        result = merge_file_chunks(chunks, source, "/t.py", "t.py", [])
        # The function merges with the imports gap into one chunk.
        assert len(result) == 1
        # But only ONE named non-gap chunk, so no tag.
        assert "multi_chunk_merge" not in result[0].tags, (
            f"function+imports merge should NOT be tagged "
            f"multi_chunk_merge. tags={result[0].tags}"
        )


# ---------------------------------------------------------------------------
# Same-type merges keep their chunk_type (regression pin)
# ---------------------------------------------------------------------------

class TestExistingTaxonomyPreserved:
    """The existing test_same_type_merge_keeps_type expectation in
    chunk_merging.py must not regress: R10 is additive, not a chunk_type
    rewrite. Eval data has been baked against the current taxonomy."""

    def test_two_functions_merge_chunk_type_stays_function(self):
        source = "def a():\n    pass\ndef b():\n    pass"
        chunks = [
            _mk("a", "function", 1, 2, "def a():\n    pass"),
            _mk("b", "function", 3, 4, "def b():\n    pass"),
        ]
        result = merge_file_chunks(chunks, source, "/t.py", "t.py", [])
        merged = [c for c in result if c.name and " + " in c.name]
        assert merged[0].chunk_type == "function", (
            "R10 is additive — same-type merges still get the type of "
            "the constituents. The multi_chunk_merge tag is the new signal."
        )
        # But the tag is also present.
        assert "multi_chunk_merge" in merged[0].tags
