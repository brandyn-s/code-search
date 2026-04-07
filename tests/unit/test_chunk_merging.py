"""Tests for chunk merging logic (cAST-style split-then-merge)."""

from chunking.chunk_merging import merge_file_chunks, nws_count
from chunking.code_chunk import CodeChunk


def _make_chunk(name, start, end, content, chunk_type='function', tags=None):
    """Helper to create a CodeChunk for testing."""
    return CodeChunk(
        content=content,
        chunk_type=chunk_type,
        start_line=start,
        end_line=end,
        file_path='test.py',
        relative_path='test.py',
        folder_structure=[],
        name=name,
        tags=tags or ['python'],
    )


class TestNwsCount:
    def test_counts_non_whitespace(self):
        assert nws_count('hello world') == 10
        assert nws_count('  def foo():  ') == 9  # d,e,f,f,o,o,(,),:
        assert nws_count('\n\n\n') == 0
        assert nws_count('') == 0

    def test_ignores_all_whitespace_types(self):
        assert nws_count('a\tb\nc\r d') == 4


class TestMergeFileChunks:
    def test_empty_chunks_empty_source(self):
        result = merge_file_chunks([], '', 'test.py', 'test.py', [])
        assert result == []

    def test_single_large_chunk_unchanged(self):
        """A chunk already over max_nws stays as-is."""
        content = 'x' * 2000
        source = content
        chunk = _make_chunk('big', 1, 1, content)
        result = merge_file_chunks([chunk], source, 'test.py', 'test.py', [])
        assert len(result) == 1
        assert result[0].name == 'big'

    def test_small_chunks_merged(self):
        """Adjacent small chunks are merged into one."""
        source = 'def a():\n    return 1\n\ndef b():\n    return 2\n'
        chunks = [
            _make_chunk('a', 1, 2, 'def a():\n    return 1'),
            _make_chunk('b', 4, 5, 'def b():\n    return 2'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert len(result) == 1
        assert 'a' in result[0].name
        assert 'b' in result[0].name

    def test_gap_code_captured(self):
        """Code between chunks (imports, constants) is included in merged output."""
        source = 'import os\nimport sys\n\nCONST = 42\n\ndef func():\n    return CONST\n'
        chunks = [
            _make_chunk('func', 6, 7, 'def func():\n    return CONST'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert len(result) == 1
        # Merged chunk should include imports from lines 1-4
        assert 'import os' in result[0].content
        assert 'CONST = 42' in result[0].content

    def test_budget_prevents_over_merge(self):
        """Chunks exceeding max_nws budget are not merged together."""
        # Each line has 21 NWS chars (result_var=compute(x))
        # 30 lines * 21 = 630 NWS per function body, plus 'defa():' = 7 = ~637
        line = '    result_var = compute(x)\n'
        assert nws_count(line) == 21  # sanity check
        content_a = 'def a():\n' + line * 30  # ~637 NWS
        content_b = 'def b():\n' + line * 30
        source = content_a + '\n' + content_b

        lines_a = content_a.split('\n')
        lines_b = content_b.split('\n')

        chunks = [
            _make_chunk('a', 1, len(lines_a), content_a),
            _make_chunk('b', len(lines_a) + 2, len(lines_a) + 1 + len(lines_b), content_b),
        ]
        result = merge_file_chunks(
            chunks, source, 'test.py', 'test.py', [], max_nws=1000
        )
        # Each chunk is ~637 NWS, combined ~1274 > 1000 budget
        assert len(result) >= 2, f"Expected >=2 chunks, got {len(result)} (budget exceeded)"

    def test_preserves_chunk_order(self):
        """Chunks maintain their source order after merging."""
        source = 'def first():\n    pass\n\ndef second():\n    pass\n\ndef third():\n    pass\n'
        chunks = [
            _make_chunk('first', 1, 2, 'def first():\n    pass'),
            _make_chunk('second', 4, 5, 'def second():\n    pass'),
            _make_chunk('third', 7, 8, 'def third():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert len(result) == 1
        # Names should be in source order
        assert result[0].name == 'first + second + third'

    def test_merged_type_single(self):
        """Merging chunks of the same type keeps that type."""
        source = 'def a():\n    pass\n\ndef b():\n    pass\n'
        chunks = [
            _make_chunk('a', 1, 2, 'def a():\n    pass', chunk_type='function'),
            _make_chunk('b', 4, 5, 'def b():\n    pass', chunk_type='function'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert result[0].chunk_type == 'function'

    def test_merged_type_mixed(self):
        """Merging chunks of different types produces 'merged' type."""
        source = 'def func():\n    pass\n\nclass Cls:\n    pass\n'
        chunks = [
            _make_chunk('func', 1, 2, 'def func():\n    pass', chunk_type='function'),
            _make_chunk('Cls', 4, 5, 'class Cls:\n    pass', chunk_type='class'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert result[0].chunk_type == 'merged'

    def test_coverage_complete(self):
        """All non-empty source lines should be covered by merged chunks."""
        source = 'import os\n\nVAR = 1\n\ndef a():\n    return 1\n\n# comment\n\ndef b():\n    return 2\n'
        chunks = [
            _make_chunk('a', 5, 6, 'def a():\n    return 1'),
            _make_chunk('b', 10, 11, 'def b():\n    return 2'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])

        covered = set()
        for c in result:
            for i in range(c.start_line, c.end_line + 1):
                covered.add(i)

        source_lines = source.split('\n')
        non_empty = {i + 1 for i, line in enumerate(source_lines) if line.strip()}
        assert non_empty.issubset(covered), f"Uncovered: {non_empty - covered}"

    def test_trailing_gap_captured(self):
        """Code after the last chunk is included."""
        source = 'def func():\n    pass\n\n# trailing comment\nEND = True\n'
        chunks = [
            _make_chunk('func', 1, 2, 'def func():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert 'END = True' in result[0].content

    def test_leading_gap_captured(self):
        """Code before the first chunk is included."""
        source = '#!/usr/bin/env python\n"""Module doc."""\nimport os\n\ndef func():\n    pass\n'
        chunks = [
            _make_chunk('func', 5, 6, 'def func():\n    pass'),
        ]
        result = merge_file_chunks(chunks, source, 'test.py', 'test.py', [])
        assert '#!/usr/bin/env python' in result[0].content
        assert 'import os' in result[0].content
