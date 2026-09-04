"""Post-processing merge step for AST chunks.

Greedily merges adjacent small chunks from the same file until they reach
an optimal NWS (non-whitespace) character budget. Also captures gap code
(imports, constants, comments) between semantic units.

Based on the cAST paper (CMU, 2025): split at AST boundaries, then merge
adjacent siblings. Our variant: keep existing language-specific AST chunking,
add a file-level merge pass as post-processing.

WHY NWS characters: Lines are meaningless for sizing — a 50-line file of
blank lines and a 50-line file of dense code are not equivalent. NWS chars
measure actual information content. (cAST paper, Section 3.2)
"""

import re
from typing import List

from chunking.code_chunk import CodeChunk

# Budget: merge up to this size. Chroma 2025 context rot research:
# degradation cliff at ~2500 tokens. 1500 NWS ≈ 500-600 tokens,
# well under the ceiling with room for contextual headers.
# (Sizing context: sub-100-token chunks degrade retrieval 6-16% per
# Ekimetrics 2026 — the greedy merge below absorbs most of them by
# packing adjacent segments toward this budget. A hard minimum-size
# floor is deliberately NOT enforced: that would change chunk output
# and is a measured chunking change, not a refactor.)
# 2500 hard-set by the P1 three-arm sweep (2026-06-12): golden MRR +0.0431,
# 95% CI [+0.0022, +0.0839] vs 1500; billing +0.20*/netlib +0.07*;
# harvested −0.0395 (caveat) and ~+58% median result size — tradeoffs and
# gate application in internal eval finding (2026-06-12).
# Production indexes realize this only after a one-time full reindex.
MAX_CHUNK_NWS = 2500


def nws_count(text: str) -> int:
    """Count non-whitespace characters."""
    return len(re.sub(r'\s', '', text))


def merge_file_chunks(
    chunks: List[CodeChunk],
    source_code: str,
    file_path: str,
    relative_path: str,
    folder_structure: List[str],
    max_nws: int = MAX_CHUNK_NWS,
) -> List[CodeChunk]:
    """Merge adjacent small chunks to reach optimal embedding size.

    Algorithm:
    1. Sort existing chunks by start_line
    2. Build a segment list: [gap, chunk, gap, chunk, ..., gap]
    3. Greedily pack adjacent segments into merged chunks up to max_nws
    4. Segments already over max_nws are kept as-is (never split further)

    The gap segments capture code between semantic units (imports, constants,
    module-level assignments, comments) that the AST chunker skips.

    Args:
        chunks: Per-file chunks from the language-specific chunker
        source_code: Full source code of the file
        file_path: Absolute path to the file
        relative_path: Path relative to project root
        folder_structure: Folder parts for CodeChunk
        max_nws: Maximum NWS characters per merged chunk

    Returns:
        Merged list of CodeChunks
    """
    if not chunks and not source_code.strip():
        return []

    source_lines = source_code.split('\n')
    total_lines = len(source_lines)

    # Tree-sitter chunkers intentionally emit BOTH a class chunk and its
    # nested method chunks (dual retrieval granularity), so the input is a
    # containment forest, not a flat disjoint list. The previous flat merge
    # mishandled that: the gap tracker regressed to a nested chunk's end and
    # fabricated overlapping "module_level" gap chunks from class-interior
    # lines, and greedy packing stitched a trailing method to code AFTER its
    # class (cross-boundary chunks). Measured on a 12-method class: 58 of 71
    # lines double-indexed plus a method+module-code chunk.
    #
    # The merge is now containment-aware: siblings merge among themselves
    # within their level's region (top level = whole file; children = the
    # container's range), so groups never cross a container boundary and gap
    # segments only cover lines no chunk at that level covers.
    chunks = sorted(chunks, key=lambda c: (c.start_line, -c.end_line))

    roots: List[_Node] = []
    stack: List[_Node] = []
    for chunk in chunks:
        node = _Node(chunk)
        while stack and not _contains(stack[-1].chunk, chunk):
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    result: List[CodeChunk] = []

    def emit_level(nodes: 'List[_Node]', lo: int, hi: int, capture_gaps: bool) -> None:
        result.extend(_merge_siblings(
            [n.chunk for n in nodes], source_lines, lo, hi,
            file_path, relative_path, folder_structure, max_nws,
            capture_gaps=capture_gaps,
        ))
        for n in nodes:
            if n.children:
                # The children's level merges within their own envelope, not
                # the container's full range. capture_gaps=False: EVERY line
                # inside the container — header, footer, and lines between
                # siblings (class attributes between methods) — belongs to
                # the container chunk alone. Fabricating child-level gap
                # segments for between-sibling lines duplicated them into a
                # second chunk (container + phantom module_level), the same
                # double-indexing class PR #229 fixed for the rewind case.
                # Between-sibling contentful lines are instead HOLES, which
                # the hole rule below already refuses to pack across.
                lo_c = min(c.chunk.start_line for c in n.children)
                hi_c = max(c.chunk.end_line for c in n.children)
                emit_level(n.children, lo_c, hi_c, capture_gaps=False)

    # Only the root level captures gaps: a line outside every root chunk is
    # covered by nothing, so a module_level gap chunk is the only way it
    # gets indexed. At child levels the container already covers everything.
    emit_level(roots, 1, total_lines, capture_gaps=True)
    result.sort(key=lambda c: (c.start_line, c.end_line))
    return result


class _Node:
    """Containment-forest node for the merge algorithm."""
    __slots__ = ('chunk', 'children')

    def __init__(self, chunk: CodeChunk):
        self.chunk = chunk
        self.children: List[_Node] = []


def _contains(parent: CodeChunk, child: CodeChunk) -> bool:
    """True when child's line range is strictly inside parent's."""
    return (
        parent.start_line <= child.start_line
        and child.end_line <= parent.end_line
        and (parent.start_line < child.start_line or child.end_line < parent.end_line)
    )


def _merge_siblings(
    chunks: List[CodeChunk],
    source_lines: List[str],
    region_start: int,
    region_end: int,
    file_path: str,
    relative_path: str,
    folder_structure: List[str],
    max_nws: int,
    capture_gaps: bool = True,
) -> List[CodeChunk]:
    """Merge DISJOINT sibling chunks within [region_start, region_end].

    This is the original greedy gap+pack algorithm, bounded to a region so
    it can run per containment level. capture_gaps=False (child levels)
    skips gap-segment fabrication entirely — the enclosing container chunk
    already carries every uncovered line in the region.
    """
    total_lines = region_end

    # Build segment list: alternating gaps and chunks
    segments = []
    last_end_line = region_start - 1  # 0-based exclusive (line after last covered line)

    for chunk in chunks:
        chunk_start_0 = chunk.start_line - 1  # convert to 0-based
        chunk_end_0 = chunk.end_line  # exclusive (0-based)

        # Gap before this chunk
        if capture_gaps and last_end_line < chunk_start_0:
            gap_text = '\n'.join(source_lines[last_end_line:chunk_start_0])
            if gap_text.strip():
                segments.append(_Segment(
                    content=gap_text,
                    start_line=last_end_line + 1,
                    end_line=chunk_start_0,
                    chunk_type='module_level',
                    name=None,
                    tags=chunk.tags[:] if chunk.tags else [],
                    is_gap=True,
                ))

        # The chunk itself
        segments.append(_Segment(
            content=chunk.content,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            tags=chunk.tags[:] if chunk.tags else [],
            is_gap=False,
            original_chunk=chunk,
        ))
        # Coverage must be MONOTONIC. Sibling chunks at one level are
        # disjoint for tree-sitter input (nesting is handled by the
        # containment forest in merge_file_chunks), but non-tree chunkers
        # could still emit partial overlaps; a bare assignment would let a
        # chunk REWIND coverage and re-emit already-covered lines as
        # phantom gap chunks.
        last_end_line = max(last_end_line, chunk_end_0)

    # Trailing gap
    if capture_gaps and last_end_line < total_lines:
        gap_text = '\n'.join(source_lines[last_end_line:])
        if gap_text.strip():
            segments.append(_Segment(
                content=gap_text,
                start_line=last_end_line + 1,
                end_line=total_lines,
                chunk_type='module_level',
                name=None,
                tags=chunks[0].tags[:] if chunks else [],
                is_gap=True,
            ))

    if not segments:
        # File has no parseable content — return original chunks
        return chunks

    # Greedy packing: merge adjacent segments up to max_nws.
    #
    # Hole rule: merged-group content is rebuilt below as ONE contiguous
    # source span, so every line between the group's first and last segment
    # gets included — even lines belonging to no segment in the group. With
    # non-overlapping emissions that's always correct (contentful gaps became
    # gap segments; remaining holes are whitespace-only). With OVERLAPPING
    # emissions (a class chunk plus its nested method chunks — produced by
    # the production chunkers), a group of method segments can have a hole
    # that is the parent's exclusive region (e.g., a trailing class attr
    # after the last method): spanning it duplicates that line into a second
    # chunk. So: never pack a segment into the current group across a hole
    # containing non-whitespace content.
    groups = []
    current_group = [segments[0]]
    current_nws = segments[0].nws
    group_max_end = segments[0].end_line

    for seg in segments[1:]:
        hole_has_content = False
        if seg.start_line > group_max_end + 1:
            hole_text = '\n'.join(
                source_lines[group_max_end:seg.start_line - 1]
            )
            hole_has_content = bool(hole_text.strip())

        combined = current_nws + seg.nws
        if combined <= max_nws and not hole_has_content:
            current_group.append(seg)
            current_nws = combined
        else:
            groups.append(current_group)
            current_group = [seg]
            current_nws = seg.nws
            group_max_end = 0
        group_max_end = max(group_max_end, seg.end_line)
    groups.append(current_group)

    # Convert groups back to CodeChunks
    result = []
    for group in groups:
        if len(group) == 1 and not group[0].is_gap and group[0].original_chunk:
            # Single non-gap segment — return the original chunk unchanged
            result.append(group[0].original_chunk)
            continue

        # Merged group: use source lines for clean, gap-inclusive content.
        # End is the MAX across the group, not the last segment's — with
        # overlapping emissions a nested segment can sort after its parent
        # while ending before it.
        start_line = group[0].start_line
        end_line = max(seg.end_line for seg in group)
        content = '\n'.join(source_lines[start_line - 1:end_line])

        # Collect metadata from constituent chunks
        names = []
        types = set()
        all_tags = set()
        docstring = None
        decorators = []
        parent_name = None
        # Count distinctly-named non-gap segments. This is the load-bearing
        # signal for "this chunk merged semantic units across a boundary"
        # — not just "function body + its imports" (one named chunk + gaps)
        # but actual e.g. authenticate() + render_template() smushed together
        # because their NWS sum fit under MAX_CHUNK_NWS.
        named_non_gap_count = 0

        for seg in group:
            if seg.name:
                names.append(seg.name)
            types.add(seg.chunk_type)
            if seg.tags:
                all_tags.update(seg.tags)
            if not seg.is_gap and seg.original_chunk:
                oc = seg.original_chunk
                if oc.docstring and not docstring:
                    docstring = oc.docstring
                if oc.decorators:
                    decorators.extend(oc.decorators)
                if oc.parent_name and not parent_name:
                    parent_name = oc.parent_name
                if oc.name:
                    named_non_gap_count += 1

        # Determine chunk type for the merged result
        non_gap_types = types - {'module_level'}
        if len(non_gap_types) == 1:
            chunk_type = next(iter(non_gap_types))
        elif non_gap_types:
            chunk_type = 'merged'
        else:
            chunk_type = 'module_level'

        # R10: signal cross-boundary merges via a stable tag so downstream
        # search consumers can filter or deboost them. The chunk_type alone
        # is lossy: two `function` chunks fused with a comment gap currently
        # come out as `chunk_type='function'`, indistinguishable from a
        # single function. Adding `multi_chunk_merge` is additive (existing
        # downstream code ignores unknown tags) and preserves the existing
        # taxonomy that the eval data has been baked against.
        #
        # Threshold of 2+ NAMED non-gap chunks: a single function chunk
        # merged with its preceding imports/comments is NOT a cross-
        # boundary merge — only one semantic unit. Two functions merged
        # via a small gap IS the bad pattern.
        if named_non_gap_count >= 2:
            all_tags.add('multi_chunk_merge')

        merged_name = ' + '.join(names) if names else None

        merged_chunk = CodeChunk(
            content=content,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            relative_path=relative_path,
            folder_structure=folder_structure,
            name=merged_name,
            parent_name=parent_name,
            docstring=docstring,
            decorators=decorators,
            tags=list(all_tags),
        )
        result.append(merged_chunk)

    return result


class _Segment:
    """Internal segment representation for the merge algorithm."""
    __slots__ = (
        'content', 'start_line', 'end_line', 'chunk_type',
        'name', 'tags', 'is_gap', 'original_chunk', 'nws',
    )

    def __init__(
        self,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        name: str,
        tags: list,
        is_gap: bool,
        original_chunk: CodeChunk = None,
    ):
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type
        self.name = name
        self.tags = tags
        self.is_gap = is_gap
        self.original_chunk = original_chunk
        self.nws = nws_count(content)


# ---------------------------------------------------------------------------
# Co-located tests — run with: pytest chunking/chunk_merging.py -v
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])


def _make_chunk(name, chunk_type, start, end, content='x = 1'):
    """Test helper: create a minimal CodeChunk."""
    return CodeChunk(
        content=content,
        chunk_type=chunk_type,
        start_line=start,
        end_line=end,
        file_path='/test.py',
        relative_path='test.py',
        folder_structure=[],
        name=name,
    )


class TestMergeFileChunks:
    """Tests for merge_file_chunks output contract.

    These verify the naming and typing conventions that downstream code
    (integration tests, search filters) depends on.
    """

    def test_merged_name_uses_plus_separator(self):
        """When two named chunks merge, name = 'A + B'."""
        source = 'def foo():\n    pass\ndef bar():\n    pass'
        chunks = [
            _make_chunk('foo', 'function', 1, 2, 'def foo():\n    pass'),
            _make_chunk('bar', 'function', 3, 4, 'def bar():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, '/t.py', 't.py', [])
        merged = [c for c in result if c.name and ' + ' in c.name]
        assert len(merged) == 1
        assert merged[0].name == 'foo + bar'

    def test_merged_mixed_types_become_merged(self):
        """When a function and class merge, chunk_type = 'merged'."""
        source = 'class A:\n    pass\ndef b():\n    pass'
        chunks = [
            _make_chunk('A', 'class', 1, 2, 'class A:\n    pass'),
            _make_chunk('b', 'function', 3, 4, 'def b():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, '/t.py', 't.py', [])
        merged = [c for c in result if c.name and ' + ' in c.name]
        assert len(merged) == 1
        assert merged[0].chunk_type == 'merged'

    def test_single_chunk_preserves_original(self):
        """A single large chunk is returned unchanged (no merge)."""
        content = 'class Big:\n' + '    x = 1\n' * 100
        source = content
        chunks = [_make_chunk('Big', 'class', 1, 101, content)]
        result = merge_file_chunks(chunks, source, '/t.py', 't.py', [])
        assert len(result) == 1
        assert result[0].name == 'Big'
        assert result[0].chunk_type == 'class'

    def test_same_type_merge_keeps_type(self):
        """When two functions merge, chunk_type stays 'function'."""
        source = 'def a():\n    pass\ndef b():\n    pass'
        chunks = [
            _make_chunk('a', 'function', 1, 2, 'def a():\n    pass'),
            _make_chunk('b', 'function', 3, 4, 'def b():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, '/t.py', 't.py', [])
        merged = [c for c in result if c.name and ' + ' in c.name]
        assert len(merged) == 1
        assert merged[0].chunk_type == 'function'

    def test_nws_count(self):
        assert nws_count('  hello  world  ') == 10
        assert nws_count('') == 0
        assert nws_count('abc') == 3
