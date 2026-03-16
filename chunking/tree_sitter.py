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
                logger.error(f"Failed to read file {file_path}: {e}")
                return []

        try:
            return chunker.chunk_code(content)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
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
