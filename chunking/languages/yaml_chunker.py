"""YAML top-level key chunker. No tree-sitter grammar needed."""

import re
from typing import List

from chunking.base_chunker import TreeSitterChunk


class YamlChunker:
    """Chunks YAML files by top-level keys (0-indent keys).

    Does not use tree-sitter. Splits on lines that start at column 0
    and end with `:`. Each top-level key becomes a chunk.
    """

    language_name = "yaml"

    def __init__(self, overlap_chars: int = 0):
        self.overlap_chars = overlap_chars

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        lines = source_code.split("\n")
        chunks = []
        top_key_pattern = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_.-]*)\s*:")

        current_name = None
        current_start = 1
        current_lines = []
        prev_overlap = ""

        for i, line in enumerate(lines, 1):
            # Skip YAML front matter delimiter
            if line.strip() == "---":
                if current_lines:
                    current_lines.append(line)
                continue

            match = top_key_pattern.match(line)
            if match:
                # Emit previous section
                if current_lines and current_name is not None:
                    content = "\n".join(current_lines)
                    if content.strip() and len(current_lines) >= 2:
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
                current_name = match.group(1)
                current_start = i
                current_lines = [line]
            else:
                current_lines.append(line)

        # Emit last section
        if current_lines and current_name is not None:
            content = "\n".join(current_lines)
            if content.strip() and len(current_lines) >= 2:
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
        return file_path.endswith(".yml") or file_path.endswith(".yaml")
