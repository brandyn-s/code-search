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

# Floor: chunks below this are noise for embedding models.
# Ekimetrics 2026: sub-100-token chunks degrade retrieval by 6-16%.
# 400 NWS chars ≈ 100 tokens.
MIN_CHUNK_NWS = 400

# Budget: merge up to this size. Chroma 2025 context rot research:
# degradation cliff at ~2500 tokens. 1500 NWS ≈ 500-600 tokens,
# well under the ceiling with room for contextual headers.
MAX_CHUNK_NWS = 1500


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

    # Sort by start line
    chunks = sorted(chunks, key=lambda c: c.start_line)

    # Build segment list: alternating gaps and chunks
    segments = []
    last_end_line = 0  # 0-based exclusive (line after last covered line)

    for chunk in chunks:
        chunk_start_0 = chunk.start_line - 1  # convert to 0-based
        chunk_end_0 = chunk.end_line  # exclusive (0-based)

        # Gap before this chunk
        if last_end_line < chunk_start_0:
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
        last_end_line = chunk_end_0

    # Trailing gap
    if last_end_line < total_lines:
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

    # Greedy packing: merge adjacent segments up to max_nws
    groups = []
    current_group = [segments[0]]
    current_nws = segments[0].nws

    for seg in segments[1:]:
        combined = current_nws + seg.nws
        if combined <= max_nws:
            current_group.append(seg)
            current_nws = combined
        else:
            groups.append(current_group)
            current_group = [seg]
            current_nws = seg.nws
    groups.append(current_group)

    # Convert groups back to CodeChunks
    result = []
    for group in groups:
        if len(group) == 1 and not group[0].is_gap and group[0].original_chunk:
            # Single non-gap segment — return the original chunk unchanged
            result.append(group[0].original_chunk)
            continue

        # Merged group: use source lines for clean, gap-inclusive content
        start_line = group[0].start_line
        end_line = group[-1].end_line
        content = '\n'.join(source_lines[start_line - 1:end_line])

        # Collect metadata from constituent chunks
        names = []
        types = set()
        all_tags = set()
        docstring = None
        decorators = []
        parent_name = None

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

        # Determine chunk type for the merged result
        non_gap_types = types - {'module_level'}
        if len(non_gap_types) == 1:
            chunk_type = next(iter(non_gap_types))
        elif non_gap_types:
            chunk_type = 'merged'
        else:
            chunk_type = 'module_level'

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
