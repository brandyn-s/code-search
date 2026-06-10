"""Containment-aware merge tests (2026-06-10).

Tree-sitter chunkers intentionally emit BOTH a class chunk and its nested
method chunks (dual retrieval granularity). The flat merge algorithm
mishandled that input shape in two ways:

1. The gap tracker REGRESSED to a nested chunk's end line after the class
   had already advanced it, fabricating overlapping "module_level" gap
   chunks out of class-interior lines (lines indexed 3x).
2. Greedy packing stitched a class's trailing method to module-level code
   AFTER the class (cross-boundary chunks: `method_11 + trailing_helper`).

The merge is now containment-aware: siblings merge among themselves within
their level's envelope, so groups never cross a container boundary. These
tests pin the fix with concrete cases plus randomized invariants over
generated containment forests.
"""

import random

from chunking.chunk_merging import merge_file_chunks, nws_count
from chunking.code_chunk import CodeChunk


def _mk(source_lines, start, end, name, chunk_type='function'):
    return CodeChunk(
        content='\n'.join(source_lines[start - 1:end]),
        chunk_type=chunk_type,
        start_line=start,
        end_line=end,
        file_path='/t.py',
        relative_path='t.py',
        folder_structure=[],
        name=name,
        parent_name=None,
        docstring=None,
        decorators=[],
        tags=['t'],
    )


def _partial_overlap(a, b):
    """Overlap that is neither disjoint nor containment."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo > hi:
        return False
    a_contains_b = a[0] <= b[0] and b[1] <= a[1]
    b_contains_a = b[0] <= a[0] and a[1] <= b[1]
    return not a_contains_b and not b_contains_a


class TestContainmentMerge:
    def _class_with_methods_source(self):
        """A class + nested methods + trailing module code (1-based lines):

        1  import os
        2  (blank)
        3  class Svc:          <- class chunk [3, 12]
        4      def a(self):    <- method chunk [4, 6]
        5          x = 1
        6          return x
        7      (blank)
        8      def b(self):    <- method chunk [8, 10]
        9          y = 2
        10         return y
        11     (blank)
        12     LEVEL = 3       <- class attribute (gap between/after methods)
        13 (blank)
        14 def trailing():     <- function chunk [14, 15]
        15     return os.sep
        """
        lines = [
            'import os',
            '',
            'class Svc:',
            '    def a(self):',
            '        x = 1',
            '        return x',
            '',
            '    def b(self):',
            '        y = 2',
            '        return y',
            '',
            '    LEVEL = 3',
            '',
            'def trailing():',
            '    return os.sep',
        ]
        src = '\n'.join(lines)
        chunks = [
            _mk(lines, 3, 12, 'Svc', 'class'),
            _mk(lines, 4, 6, 'a', 'method'),
            _mk(lines, 8, 10, 'b', 'method'),
            _mk(lines, 14, 15, 'trailing', 'function'),
        ]
        return src, lines, chunks

    def test_no_partial_overlaps(self):
        """Output ranges are disjoint or nested — never partially overlapping.

        The pre-fix merge fabricated gap chunks that partially overlapped
        the class chunk (class-interior lines indexed three times).
        """
        src, _lines, chunks = self._class_with_methods_source()
        out = merge_file_chunks(chunks, src, '/t.py', 't.py', [], max_nws=60)
        ranges = [(c.start_line, c.end_line) for c in out]
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                assert not _partial_overlap(ranges[i], ranges[j]), (
                    f'partial overlap {ranges[i]} vs {ranges[j]}: {ranges}')

    def test_no_cross_class_boundary_stitch(self):
        """A method must never merge with module-level code after its class."""
        src, _lines, chunks = self._class_with_methods_source()
        out = merge_file_chunks(chunks, src, '/t.py', 't.py', [], max_nws=60)
        class_range = (3, 12)
        for c in out:
            inside = class_range[0] <= c.start_line <= class_range[1]
            if inside and (c.start_line, c.end_line) != class_range:
                assert c.end_line <= class_range[1], (
                    f'chunk [{c.start_line},{c.end_line}] {c.name!r} crosses '
                    f'the class boundary at line {class_range[1]}')

    def test_class_chunk_survives_whole(self):
        src, _lines, chunks = self._class_with_methods_source()
        out = merge_file_chunks(chunks, src, '/t.py', 't.py', [], max_nws=60)
        assert any(
            (c.start_line, c.end_line) == (3, 12) and c.chunk_type == 'class'
            for c in out), [(c.start_line, c.end_line, c.chunk_type) for c in out]

    def test_sibling_methods_still_merge(self):
        """Dual granularity is preserved: methods merge among themselves."""
        src, _lines, chunks = self._class_with_methods_source()
        out = merge_file_chunks(chunks, src, '/t.py', 't.py', [], max_nws=1500)
        merged_methods = [c for c in out if c.name == 'a + b']
        assert len(merged_methods) == 1, [(c.name, c.start_line, c.end_line) for c in out]
        # The merge stays inside the children's envelope (4..10), not the
        # class header.
        assert merged_methods[0].start_line == 4
        assert merged_methods[0].end_line == 10

    def test_flat_disjoint_behavior_unchanged(self):
        """Files without nesting take the original single-level path."""
        lines = ['def a():', '    pass', '', 'def b():', '    pass']
        src = '\n'.join(lines)
        chunks = [_mk(lines, 1, 2, 'a'), _mk(lines, 4, 5, 'b')]
        out = merge_file_chunks(chunks, src, '/t.py', 't.py', [])
        assert len(out) == 1
        assert out[0].name == 'a + b'

    def test_randomized_forest_invariants(self):
        """Generated containment forests: no partial overlaps, no lost
        lines, budget respected on merged chunks."""
        for seed in range(1, 101):
            rng = random.Random(seed)
            n = rng.randint(8, 70)
            lines = [
                f'line_{i} = {i}' if rng.random() > 0.2 else ''
                for i in range(n)
            ]
            src = '\n'.join(lines)
            chunks, line = [], 1
            while line <= n - 2:
                if rng.random() < 0.45:
                    length = rng.randint(2, 14)
                    e = min(line + length, n)
                    chunks.append(_mk(lines, line, e, f'top{line}',
                                      'class' if length > 6 else 'function'))
                    cl = line + 1
                    while cl < e - 1 and rng.random() < 0.6:
                        ce = min(cl + rng.randint(1, 3), e - 1)
                        chunks.append(_mk(lines, cl, ce, f'kid{cl}', 'method'))
                        cl = ce + rng.randint(1, 2)
                    line = e + rng.randint(1, 3)
                else:
                    line += rng.randint(1, 3)
            if not chunks:
                continue
            max_nws = rng.choice([40, 120, 1500])
            out = merge_file_chunks(chunks, src, '/t.py', 't.py', [], max_nws=max_nws)

            ranges = [(c.start_line, c.end_line) for c in out]
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    assert not _partial_overlap(ranges[i], ranges[j]), (
                        f'seed {seed}: partial overlap {ranges[i]} vs {ranges[j]}')

            cov = set()
            for s, e in ranges:
                cov.update(range(s, e + 1))
            for c in chunks:
                for ln in range(c.start_line, c.end_line + 1):
                    assert ln in cov, f'seed {seed}: lost chunk line {ln}'
            for ln in range(1, n + 1):
                if lines[ln - 1].strip():
                    assert ln in cov, f'seed {seed}: lost non-blank line {ln}'

            for c in out:
                if c.name and ' + ' in c.name:
                    assert nws_count(c.content) <= max_nws, (
                        f'seed {seed}: merged chunk over budget')
