"""Nix-specific tree-sitter based chunker."""

from typing import Any, Dict, List, Set

from chunking.base_chunker import LanguageChunker, TreeSitterChunk


class NixChunker(LanguageChunker):
    """Nix-specific chunker using tree-sitter.

    Chunks Nix files on binding nodes inside let/attrset expressions.
    Recurses into let_expression to find inner bindings, unlike the
    base chunker which stops at the first splittable node.
    """

    MIN_CHUNK_LINES = 5

    def __init__(self):
        super().__init__("nix")

    def _get_splittable_node_types(self) -> Set[str]:
        return {"binding", "let_expression"}

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        if node.type == "binding":
            for child in node.children:
                if child.type == "attrpath":
                    metadata["name"] = self.get_node_text(child, source)
                    break
            for child in node.children:
                if child.type in (
                    "function_expression",
                    "attrset_expression",
                    "rec_attrset_expression",
                    "list_expression",
                    "with_expression",
                    "if_expression",
                    "let_expression",
                    "string_expression",
                    "integer_expression",
                    "path_expression",
                    "apply_expression",
                ):
                    metadata["value_type"] = child.type
                    break
            if metadata.get("value_type") == "function_expression":
                metadata["is_function"] = True

        elif node.type == "let_expression":
            metadata["name"] = "let"

        return metadata

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        """Chunk Nix source with recursive binding extraction."""
        source_bytes = bytes(source_code, "utf-8")
        tree = self.parser.parse(source_bytes)
        chunks = []

        self._extract_chunks(tree.root_node, source_bytes, source_code, chunks)

        if not chunks and source_code.strip():
            chunks.append(
                TreeSitterChunk(
                    content=source_code,
                    start_line=1,
                    end_line=len(source_code.split("\n")),
                    node_type="module",
                    language=self.language_name,
                    metadata={"type": "module"},
                )
            )

        return chunks

    def _extract_chunks(
        self, node, source_bytes, source_code, chunks, parent_info=None
    ):
        """Recursively extract chunks, splitting large bindings.

        Skips emitting a parent binding when its child bindings cover >50%
        of its line count (the children are already separate chunks).
        """
        if node.type == "binding":
            start, end = self.get_line_numbers(node)
            lines = end - start + 1

            if lines >= self.MIN_CHUNK_LINES:
                metadata = self.extract_metadata(node, source_bytes)
                if parent_info:
                    metadata.update(parent_info)

                # First, recursively extract child bindings
                child_chunks_before = len(chunks)
                for child in node.children:
                    self._extract_chunks(
                        child,
                        source_bytes,
                        source_code,
                        chunks,
                        {"parent_name": metadata.get("name"), "parent_type": "binding"},
                    )
                child_chunks_after = len(chunks)

                # Calculate how many lines children cover
                child_lines = sum(
                    c.end_line - c.start_line + 1
                    for c in chunks[child_chunks_before:child_chunks_after]
                )

                # Only emit parent if children cover less than 50% of its lines
                if child_lines < lines * 0.5:
                    content = self.get_node_text(node, source_bytes)
                    chunks.append(
                        TreeSitterChunk(
                            content=content,
                            start_line=start,
                            end_line=end,
                            node_type="binding",
                            language=self.language_name,
                            metadata=metadata,
                        )
                    )
                return

        for child in node.children:
            self._extract_chunks(child, source_bytes, source_code, chunks, parent_info)
