"""Tree-sitter based code chunking with support for multiple languages."""

import logging
from pathlib import Path
from typing import List, Optional

from chunking.base_chunker import TreeSitterChunk
from chunking.languages import LANGUAGE_MAP

logger = logging.getLogger(__name__)


class TreeSitterChunker:
    """Main tree-sitter chunker that delegates to language-specific implementations."""

    def __init__(self):
        """Initialize the tree-sitter chunker."""
        self.chunkers = {}

    def get_chunker(self, file_path: str):
        """Get the appropriate chunker for a file.

        Args:
            file_path: Path to the file

        Returns:
            LanguageChunker instance or None if unsupported
        """
        suffix = Path(file_path).suffix.lower()

        if suffix not in LANGUAGE_MAP:
            return None

        language_name, chunker_class = LANGUAGE_MAP[suffix]

        # Lazy initialization of chunkers
        if suffix not in self.chunkers:
            assert callable(chunker_class), (
                f"Chunker should be callable, got {type(chunker_class)}"
            )
            try:
                self.chunkers[suffix] = chunker_class()
            except (ValueError, ImportError) as e:
                logger.debug(f"Chunker for {language_name} not available: {e}")
                return None

        return self.chunkers[suffix]

    def chunk_file(
        self, file_path: str, content: Optional[str] = None
    ) -> List[TreeSitterChunk]:
        """Chunk a file into semantic units.

        Args:
            file_path: Path to the file
            content: Optional file content (will read from file if not provided)

        Returns:
            List of TreeSitterChunk objects
        """
        chunker = self.get_chunker(file_path)

        if not chunker:
            logger.debug(f"No tree-sitter chunker available for {file_path}")
            return []

        if content is None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                # Structured per-file diagnostic. The outer chunker
                # (multi_language_chunker.chunk_file) never sees this — we
                # return [] gracefully — so emit the [CHUNKING_DIAG_FILE]
                # log line here so encoding-error and other file-read
                # failures are visible alongside parse failures in
                # `grep CHUNKING_DIAG_FILE`. Without this, encoding errors
                # show up only as `files_zero_chunks` with no per-file
                # signal to disambiguate from genuinely empty files.
                logger.error(
                    "[CHUNKING_DIAG_FILE] file=%s error_class=%s error=%s",
                    file_path, type(e).__name__, e,
                )
                return []

        try:
            return chunker.chunk_code(content)
        except Exception as e:
            # Same rationale as the file-read catch above: surface parse
            # failures as a structured per-file diagnostic so operators can
            # categorize without grepping mixed log shapes.
            logger.warning(
                "[CHUNKING_DIAG_FILE] file=%s error_class=%s error=%s",
                file_path, type(e).__name__, e,
            )
            return []

    def is_supported(self, file_path: str) -> bool:
        """Check if a file type is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if file type is supported
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in LANGUAGE_MAP
