"""Result models shared by the search pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Enhanced search result with rich metadata."""

    chunk_id: str
    similarity_score: float
    content_preview: str
    file_path: str
    relative_path: str
    folder_structure: List[str]
    chunk_type: str
    name: Optional[str]
    parent_name: Optional[str]
    start_line: int
    end_line: int
    docstring: Optional[str]
    tags: List[str]
    context_info: Dict[str, Any]
