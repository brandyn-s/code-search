"""HCL/Terraform block-based chunker. No tree-sitter grammar needed."""

import re
from typing import List

from chunking.base_chunker import TreeSitterChunk


class HclChunker:
    """Chunks HCL/Terraform files by top-level block declarations.

    Does not use tree-sitter. Uses brace counting to find block boundaries
    for resource, data, variable, output, locals, module, and provider blocks.
    """

    language_name = "hcl"

    # Patterns for top-level HCL blocks
    BLOCK_START = re.compile(
        r"^(resource|data|variable|output|locals|module|provider|terraform)\s+"
        r'(?:"([^"]+)"\s+"([^"]+)"\s*\{|"([^"]+)"\s*\{|\{)',
        re.MULTILINE,
    )

    def __init__(self, overlap_chars: int = 0):
        self.overlap_chars = overlap_chars

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        lines = source_code.split("\n")
        chunks = []
        prev_overlap = ""

        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.BLOCK_START.match(line.strip())
            if match:
                block_type = match.group(1)
                # Build block name from captured groups
                parts = [
                    p for p in [match.group(2), match.group(3), match.group(4)] if p
                ]
                block_name = f"{block_type} {' '.join(parts)}" if parts else block_type

                # Find matching closing brace
                start_line = i + 1
                brace_count = line.count("{") - line.count("}")
                block_lines = [line]
                j = i + 1

                while j < len(lines) and brace_count > 0:
                    block_lines.append(lines[j])
                    brace_count += lines[j].count("{") - lines[j].count("}")
                    j += 1

                content = "\n".join(block_lines)
                full_content = prev_overlap + content if prev_overlap else content
                chunks.append(
                    TreeSitterChunk(
                        content=full_content,
                        start_line=start_line,
                        end_line=j,
                        node_type=block_type,
                        language=self.language_name,
                        metadata={"name": block_name, "node_type": block_type},
                    )
                )
                prev_overlap = (
                    content[-self.overlap_chars :] + "\n"
                    if self.overlap_chars > 0
                    else ""
                )
                i = j
            else:
                i += 1

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
        return file_path.endswith(".tf")
