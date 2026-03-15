"""Nix-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class NixChunker(LanguageChunker):
    """Nix-specific chunker using tree-sitter.

    Chunks Nix files on `binding` nodes (the primary unit of Nix code).
    Each `name = value;` declaration becomes a searchable chunk, covering
    NixOS module options, package definitions, and function bindings.
    """

    def __init__(self):
        super().__init__('nix')

    def _get_splittable_node_types(self) -> Set[str]:
        """Nix-specific splittable node types."""
        return {
            'binding',           # name = value; (primary unit)
            'let_expression',    # let ... in ... (local definitions)
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Nix-specific metadata."""
        metadata = {'node_type': node.type}

        if node.type == 'binding':
            # Extract the attrpath (dotted name like services.nginx.enable)
            for child in node.children:
                if child.type == 'attrpath':
                    metadata['name'] = self.get_node_text(child, source)
                    break

            # Detect value type (function, attrset, list, etc.)
            for child in node.children:
                if child.type in (
                    'function_expression',
                    'attrset_expression',
                    'rec_attrset_expression',
                    'list_expression',
                    'with_expression',
                    'if_expression',
                    'let_expression',
                    'string_expression',
                    'integer_expression',
                    'path_expression',
                    'apply_expression',
                ):
                    metadata['value_type'] = child.type
                    break

            # Mark if the value is a function (lambda)
            if metadata.get('value_type') == 'function_expression':
                metadata['is_function'] = True

        elif node.type == 'let_expression':
            metadata['name'] = 'let'

        return metadata
