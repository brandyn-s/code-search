"""TOML section-based chunker. No tree-sitter grammar needed."""

import re
from typing import List

from chunking.base_chunker import TreeSitterChunk


class TomlChunker:
    """Chunks TOML files by [section] headers.

    Does not use tree-sitter. Splits on lines matching `[section]` or
    `[parent.child]` patterns. Each section becomes a chunk.
    """

    language_name = "toml"

    def __init__(self, overlap_chars: int = 0):
        self.overlap_chars = overlap_chars

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        lines = source_code.split("\n")
        chunks = []
        section_pattern = re.compile(r"^\s*\[([^\]]+)\]\s*$")

        current_name = None
        current_start = 1
        current_lines = []
        prev_overlap = ""

        for i, line in enumerate(lines, 1):
            match = section_pattern.match(line)
            if match:
                # Emit previous section
                if current_lines and current_name is not None:
                    content = "\n".join(current_lines)
                    if content.strip():
                        full_content = (
                            prev_overlap + content if prev_overlap else content
                        )
                        chunks.append(
                            TreeSitterChunk(
                                content=full_content,
                                start_line=current_start,
                                end_line=i - 1,
                                node_type="section",
                                language=self.language_name,
                                metadata={"name": current_name, "node_type": "section"},
                            )
                        )
                        prev_overlap = (
                            content[-self.overlap_chars :] + "\n"
                            if self.overlap_chars > 0
                            else ""
                        )
                current_name = match.group(1).strip()
                current_start = i
                current_lines = [line]
            else:
                if current_name is None and line.strip():
                    # Lines before first section (preamble)
                    current_name = "_preamble"
                    current_start = i
                current_lines.append(line)

        # Emit last section
        if current_lines and current_name is not None:
            content = "\n".join(current_lines)
            if content.strip():
                full_content = prev_overlap + content if prev_overlap else content
                chunks.append(
                    TreeSitterChunk(
                        content=full_content,
                        start_line=current_start,
                        end_line=len(lines),
                        node_type="section",
                        language=self.language_name,
                        metadata={"name": current_name, "node_type": "section"},
                    )
                )

        # If no sections found, emit whole file
        if not chunks and source_code.strip():
            chunks.append(
                TreeSitterChunk(
                    content=source_code,
                    start_line=1,
                    end_line=len(lines),
                    node_type="module",
                    language=self.language_name,
                    metadata={"type": "module"},
                )
            )

        return chunks

    def chunk_file(self, file_path: str) -> List[TreeSitterChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            return self.chunk_code(f.read())

    def is_supported(self, file_path: str) -> bool:
        return file_path.endswith(".toml")
