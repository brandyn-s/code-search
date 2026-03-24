"""Tests for parent-document dedup (best chunk per file)."""
from dataclasses import dataclass


@dataclass
class FakeResult:
    relative_path: str
    file_path: str
    similarity_score: float
    chunk_type: str = "function"
    name: str = ""
    content_preview: str = ""
    start_line: int = 1
    end_line: int = 10


def dedup_by_file(results):
    """Deduplicate results by file, keeping best score per file."""
    seen = {}
    for r in results:
        path = r.relative_path or r.file_path
        if path not in seen or r.similarity_score > seen[path].similarity_score:
            seen[path] = r
    # Preserve original order, only keep winners
    return [r for r in results if seen.get(r.relative_path or r.file_path) is r]


def test_dedup_keeps_best_per_file():
    results = [
        FakeResult("search/searcher.py", "search/searcher.py", 0.9),
        FakeResult("search/searcher.py", "search/searcher.py", 0.7),
        FakeResult("search/indexer.py", "search/indexer.py", 0.8),
    ]
    deduped = dedup_by_file(results)
    assert len(deduped) == 2
    assert deduped[0].similarity_score == 0.9
    assert deduped[1].relative_path == "search/indexer.py"


def test_dedup_preserves_order():
    results = [
        FakeResult("a.py", "a.py", 0.5),
        FakeResult("b.py", "b.py", 0.9),
        FakeResult("c.py", "c.py", 0.7),
    ]
    deduped = dedup_by_file(results)
    assert len(deduped) == 3  # All unique files
    assert [r.relative_path for r in deduped] == ["a.py", "b.py", "c.py"]


def test_dedup_empty():
    assert dedup_by_file([]) == []


def test_dedup_single_result():
    results = [FakeResult("a.py", "a.py", 0.9)]
    assert dedup_by_file(results) == results
