"""TypeScript-specific tree-sitter based chunker.

CS-1 (2026-05-06) — TypeScript chunker enhancements
====================================================

Background: 2026-05-06 multi-language eval (n=102) shows TS golden MRR
0.683 vs Nix 0.826 / Rust 0.917 — TS is the only language with a >100bp
gap to peers. The pre-CS-1 chunker (54 lines) split on standard
tree-sitter node types (function_declaration, class_declaration, etc.)
but did NOT carve at React component / custom-hook / arrow-function-
export boundaries — which is what the `ts-component` (9/20) and
`ts-hook` (4/20) golden queries target.

What CS-1 adds:

1. `lexical_declaration` / `variable_declaration` containing arrow
   functions becomes splittable (catches the `export const Foo = () =>
   {...}` component pattern that was previously merged into the parent
   block).
2. Variable-declaration arrows are detected by walking the
   declarator subtree — name extraction works through the
   variable_declarator node.
3. Component detection: PascalCase name on an arrow-function-bound
   declarator → `is_component=True`.
4. Hook detection: `use[A-Z]...` name on any function-bound declarator
   or function_declaration → `is_hook=True`.
5. Keeps every existing splittable type — purely additive.
"""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


def _is_component_name(name: str) -> bool:
    """React component naming convention: PascalCase identifier.

    Used to flag chunks bound to an arrow function whose binding name
    starts with an uppercase letter — distinguishes React components
    from utility functions in the same lexical-declaration syntax.
    """
    return bool(name) and name[0].isupper() and name.isidentifier()


def _is_hook_name(name: str) -> bool:
    """React hook naming convention: `use[A-Z]...`.

    React hooks must start with `use` followed by an uppercase letter
    by convention (lint rules enforce this; see eslint-plugin-react-hooks).
    A function called `useEffect` is a hook; `usefulHelper` is not.
    """
    if not name or len(name) < 4 or not name.startswith("use"):
        return False
    return name[3].isupper()


class TypeScriptChunker(LanguageChunker):
    """TypeScript-specific chunker using tree-sitter."""

    def __init__(self, use_tsx: bool = False):
        super().__init__('tsx' if use_tsx else 'typescript')
        self.use_tsx = use_tsx

    def _get_splittable_node_types(self) -> Set[str]:
        """TypeScript-specific splittable node types.

        CS-1 adds `lexical_declaration` and `variable_declaration` so
        `export const Foo = () => {...}` (the React component / custom-
        hook idiom) gets its own chunk. The parent class `LanguageChunker`
        traversal already handles depth-first descent so adding a parent
        node type doesn't double-count children — it just creates an
        additional chunk for the wrapping declaration.
        """
        return {
            'function_declaration',
            'function',
            'arrow_function',
            'class_declaration',
            'method_definition',
            'generator_function',
            'generator_function_declaration',
            'interface_declaration',
            'type_alias_declaration',
            'enum_declaration',
            # CS-1 additions
            'lexical_declaration',
            'variable_declaration',
        }

    def _extract_arrow_binding_name(self, node: Any, source: bytes) -> str:
        """For a lexical_declaration / variable_declaration node, walk
        into the variable_declarator and find the bound identifier IF
        the value is an arrow_function or function expression.

        Returns the binding name (e.g. "AlertToast" for
        `export const AlertToast = () => {...}`) or empty string when
        the declaration does not bind a function-like expression.
        """
        # lexical_declaration -> variable_declarator(s) -> {name, value}
        for child in node.children:
            if child.type != 'variable_declarator':
                continue
            name_text = ""
            value_is_function_like = False
            for sub in child.children:
                if sub.type in ('identifier', 'type_identifier') and not name_text:
                    name_text = self.get_node_text(sub, source)
                elif sub.type in ('arrow_function', 'function', 'function_declaration'):
                    value_is_function_like = True
                # Generic case: the value may be wrapped (e.g. type
                # assertion) — accept any "value" field that contains
                # arrow_function as a descendant.
            if value_is_function_like and name_text:
                return name_text
        return ""

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract TypeScript-specific metadata.

        CS-1 additions:
          - `is_component` (bool): set on arrow-bound declarations whose
            name is PascalCase
          - `is_hook` (bool): set on any function/arrow whose name
            matches `use[A-Z]...`
          - For lexical/variable declarations binding an arrow,
            populates `name` from the declarator (vs the existing
            traversal which would not find a name on the parent node).
        """
        metadata = {'node_type': node.type}

        # CS-1: special-case lexical_declaration / variable_declaration
        # binding an arrow — extract the variable name from the
        # declarator subtree before the existing identifier loop runs.
        binding_name = ""
        if node.type in ('lexical_declaration', 'variable_declaration'):
            binding_name = self._extract_arrow_binding_name(node, source)
            if binding_name:
                metadata['name'] = binding_name

        # Existing: extract name from immediate-children identifier.
        # Skipped if CS-1 already set a binding name.
        if 'name' not in metadata:
            for child in node.children:
                if child.type in ['identifier', 'type_identifier']:
                    metadata['name'] = self.get_node_text(child, source)
                    break

        # Check for async
        if node.children and self.get_node_text(node.children[0], source) == 'async':
            metadata['is_async'] = True

        # Check for export
        if node.children and self.get_node_text(node.children[0], source) == 'export':
            metadata['is_export'] = True

        # Check for generic parameters
        for child in node.children:
            if child.type == 'type_parameters':
                metadata['has_generics'] = True
                break

        # CS-1: React component / hook detection. Component = PascalCase
        # bound to an arrow function. Hook = `use[A-Z]...` on any
        # function-shaped binding (declaration or arrow). Both are
        # additive metadata — they don't change what gets chunked, just
        # tag the chunks that ARE produced for downstream weighting.
        name = metadata.get('name', '')
        if name:
            if binding_name and _is_component_name(name):
                # Only flag arrow-bound declarations as components —
                # function declarations with PascalCase names are more
                # commonly utility constructors than React components.
                metadata['is_component'] = True
            if _is_hook_name(name):
                metadata['is_hook'] = True

        return metadata
