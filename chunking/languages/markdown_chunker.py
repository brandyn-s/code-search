"""Markdown-specific tree-sitter based chunker."""

from typing import Any, Dict, List, Set

from chunking.base_chunker import LanguageChunker, TreeSitterChunk


class MarkdownChunker(LanguageChunker):
    """Markdown-specific chunker using tree-sitter.

    Chunks markdown by sections based on headers (ATX headings).
    Each section includes a header and all content until the next header.
    """

    def __init__(self):
        super().__init__('markdown')

    def _get_splittable_node_types(self) -> Set[str]:
        """Markdown-specific splittable node types."""
        return {
            'section',  # Markdown sections
            'atx_heading',  # Headers like # Title, ## Subtitle
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Markdown-specific metadata."""
        metadata = {'node_type': node.type}

        # Extract heading text and level
        if node.type == 'atx_heading':
            # Get the heading level (count of # characters)
            heading_text = self.get_node_text(node, source).strip()
            level = 0
            for char in heading_text:
                if char == '#':
                    level += 1
                else:
                    break

            metadata['heading_level'] = level
            metadata['name'] = heading_text.lstrip('#').strip()
            metadata['type'] = 'heading'

        elif node.type == 'section':
            # Extract section information
            # Find the heading within the section
            for child in node.children:
                if child.type == 'atx_heading':
                    heading_text = self.get_node_text(child, source).strip()
                    level = heading_text.count('#', 0, heading_text.index(' ') if ' ' in heading_text else len(heading_text))
                    metadata['heading_level'] = level
                    metadata['name'] = heading_text.lstrip('#').strip()
                    metadata['type'] = 'section'
                    break

        return metadata

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        """Chunk markdown into sections based on headers.

        Args:
            source_code: Markdown source code string

        Returns:
            List of TreeSitterChunk objects
        """
        source_bytes = bytes(source_code, 'utf-8')
        tree = self.parser.parse(source_bytes)
        chunks = []

        # Find all heading nodes
        def find_headings(node, headings=None):
            """Recursively find all heading nodes."""
            if headings is None:
                headings = []

            if node.type == 'atx_heading':
                headings.append(node)

            for child in node.children:
                find_headings(child, headings)

            return headings

        headings = find_headings(tree.root_node)

        if not headings:
            # No headings found, treat entire document as one chunk
            if source_code.strip():
                chunks.append(TreeSitterChunk(
                    content=source_code,
                    start_line=1,
                    end_line=len(source_code.splitlines()),
                    node_type='document',
                    language=self.language_name,
                    metadata={'type': 'document', 'name': 'Document'}
                ))
            return chunks

        # Create sections from headings
        lines = source_code.splitlines()

        for i, heading_node in enumerate(headings):
            start_line, _ = self.get_line_numbers(heading_node)

            # Determine end line - either the start of the next heading or end of document
            if i + 1 < len(headings):
                next_heading_line, _ = self.get_line_numbers(headings[i + 1])
                end_line = next_heading_line - 1
            else:
                end_line = len(lines)

            # Extract content for this section
            section_lines = lines[start_line - 1:end_line]
            content = '\n'.join(section_lines)

            # Extract metadata
            heading_text = self.get_node_text(heading_node, source_bytes).strip()
            level = 0
            for char in heading_text:
                if char == '#':
                    level += 1
                else:
                    break

            heading_name = heading_text.lstrip('#').strip()

            # Create chunk
            chunk = TreeSitterChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                node_type='section',
                language=self.language_name,
                metadata={
                    'name': heading_name,
                    'heading_level': level,
                    'type': 'section'
                }
            )
            chunks.append(chunk)

        # Handle content before first heading (preamble)
        if headings and headings[0].start_point[0] > 0:
            first_heading_line = headings[0].start_point[0] + 1
            preamble_lines = lines[:first_heading_line - 1]
            preamble_content = '\n'.join(preamble_lines).strip()

            if preamble_content:
                preamble_chunk = TreeSitterChunk(
                    content=preamble_content,
                    start_line=1,
                    end_line=first_heading_line - 1,
                    node_type='preamble',
                    language=self.language_name,
                    metadata={
                        'name': 'Preamble',
                        'type': 'preamble'
                    }
                )
                # Insert at beginning
                chunks.insert(0, preamble_chunk)

        return chunks
