"""Nix per-option chunker — Arc B of 2026-05-11 graph-augmented + per-option-nix plan.

Extends NixChunker to emit one chunk per mkOption with:
- Fully-qualified option name (walks parent attrpath chain)
- Structured prefix: `(option <FQN>) <description>` prepended to chunk content
- One chunk per option binding (no parent grouping for option-declaration blocks)

Targets PSM nix/modules/ — 137 modules using mkOption, 717 mkOption occurrences total.
Hypothesis: smaller focused chunks with FQN-first signal give Voyage embedding +
sonnet rerank a sharper target on the 18 strict rerank-error nix queries (per Phase E
classification, 2026-05-10 baseline).

Gated by env var `CODE_SEARCH_NIX_OPTION_CHUNKING=1` (default off).
Scope (B2.1): activates for path-prefix `nix/modules/` only — other .nix files use
existing NixChunker so we don't disturb the index for non-module nix files.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from chunking.base_chunker import TreeSitterChunk
from chunking.languages.nix_chunker import NixChunker, _NIXOS_OPTION_FUNCS


class NixOptionChunker(NixChunker):
    """Per-option chunker for nix/modules/*.nix files.

    Walks AST. For every `binding` whose value is an `apply_expression` with
    function in {mkOption, mkEnableOption, mkPackageOption, mkSinkUndeclaredOptions},
    emits ONE chunk per option with a structured-header prefix.

    Other binding nodes (non-option) fall back to the parent's chunking
    behavior (recursive binding extraction).
    """

    MIN_CHUNK_LINES = 2  # mkOption blocks can be small; don't drop them

    def chunk_code(self, source_code: str) -> List[TreeSitterChunk]:
        source_bytes = bytes(source_code, "utf-8")
        tree = self.parser.parse(source_bytes)
        chunks: List[TreeSitterChunk] = []

        # Walk the tree collecting option bindings with their parent attrpath chain.
        self._collect_options(tree.root_node, source_bytes, [], chunks)

        if not chunks and source_code.strip():
            # Fall back to module chunk if no options found (file may not use mkOption)
            chunks.append(
                TreeSitterChunk(
                    content=source_code,
                    start_line=1,
                    end_line=len(source_code.splitlines()),
                    node_type="module",
                    language=self.language_name,
                    metadata={"type": "module", "nix_chunker": "option-fallback"},
                )
            )

        return chunks

    def _collect_options(
        self,
        node: Any,
        source: bytes,
        parent_path: List[str],
        chunks: List[TreeSitterChunk],
    ) -> None:
        """DFS for option bindings, accumulating parent attrpath as we descend."""
        if node.type == "binding":
            # Extract this binding's attrpath
            attrpath = ""
            for child in node.children:
                if child.type == "attrpath":
                    attrpath = self.get_node_text(child, source)
                    break

            # Check if this binding's value is an mkOption call
            value_node = None
            for child in node.children:
                if child.type in (
                    "apply_expression",
                    "attrset_expression",
                    "rec_attrset_expression",
                    "with_expression",
                    "let_expression",
                    "if_expression",
                ):
                    value_node = child
                    break

            if value_node and value_node.type == "apply_expression":
                func_name = self._extract_apply_func_name(value_node, source)
                if func_name in _NIXOS_OPTION_FUNCS:
                    # This IS an option declaration.
                    full_path = ".".join(parent_path + [attrpath]) if attrpath else ".".join(parent_path)
                    self._emit_option_chunk(node, source, full_path, func_name, value_node, chunks)
                    return  # don't recurse into mkOption's body

            # Not an mkOption — recurse into children with parent_path updated
            new_path = parent_path + [attrpath] if attrpath else parent_path
            for child in node.children:
                self._collect_options(child, source, new_path, chunks)
            return

        # Non-binding: recurse children
        for child in node.children:
            self._collect_options(child, source, parent_path, chunks)

    def _emit_option_chunk(
        self,
        binding_node: Any,
        source: bytes,
        full_path: str,
        func_name: str,
        value_node: Any,
        chunks: List[TreeSitterChunk],
    ) -> None:
        """Emit one chunk for an mkOption binding with structured header prefix."""
        start, end = self.get_line_numbers(binding_node)
        body = self.get_node_text(binding_node, source)

        # Extract description if present (mkOption { ... description = "..."; ... })
        description = self._extract_description(value_node, source)
        type_text = self._extract_type_text(value_node, source)
        default_text = self._extract_default_text(value_node, source)

        # Strip the "options." prefix if present (it's redundant in FQN context).
        canonical_name = full_path
        if canonical_name.startswith("options."):
            canonical_name = canonical_name[len("options."):]

        # Build structured header. Embedding model sees this first.
        header_parts = [f"(option {canonical_name})"]
        if description:
            header_parts.append(description)
        header = " ".join(header_parts)

        # Compose content: header + blank + original body
        content = f"{header}\n\n{body}"

        metadata: Dict[str, Any] = {
            "node_type": "binding",
            "name": canonical_name,
            "option_fqn": canonical_name,
            "option_func": func_name,
            "is_option_declaration": True,
            "nix_pattern": func_name,
            "nix_chunker": "option",
            "value_type": "apply_expression",
        }
        if description:
            metadata["option_description"] = description
        if type_text:
            metadata["option_type"] = type_text
        if default_text:
            metadata["option_default"] = default_text

        # Category from attrpath prefix
        for prefix, category in (
            ("services.", "service"),
            ("systemd.services.", "service"),
            ("systemd.", "systemd"),
            ("networking.", "networking"),
            ("boot.", "boot"),
            ("users.", "user"),
            ("environment.", "environment"),
            ("security.", "security"),
            ("hardware.", "hardware"),
        ):
            if canonical_name.startswith(prefix) or f".{prefix}" in f".{canonical_name}":
                metadata["nix_category"] = category
                break

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

    def _extract_description(self, value_node: Any, source: bytes) -> str:
        """Find the `description = "..."` binding inside the mkOption argument's attrset."""
        text = self._find_string_literal_in_attrset(value_node, source, "description")
        if text:
            return text.strip()
        return ""

    def _extract_type_text(self, value_node: Any, source: bytes) -> str:
        """Find the `type = ...` binding's RHS text inside the mkOption argument's attrset."""
        return self._find_binding_rhs_in_attrset(value_node, source, "type")

    def _extract_default_text(self, value_node: Any, source: bytes) -> str:
        """Find the `default = ...` binding's RHS text."""
        return self._find_binding_rhs_in_attrset(value_node, source, "default")

    def _find_string_literal_in_attrset(
        self, apply_node: Any, source: bytes, key: str
    ) -> Optional[str]:
        """Walk an apply_expression to find binding `<key> = "<string>";` and return the unwrapped string."""
        attrset = self._find_attrset_child(apply_node)
        if not attrset:
            return None
        for child in attrset.children:
            if child.type == "binding_set":
                for binding in child.children:
                    if binding.type != "binding":
                        continue
                    if self._binding_key_is(binding, source, key):
                        rhs = self._binding_rhs_node(binding)
                        if rhs is None:
                            continue
                        rhs_text = self.get_node_text(rhs, source).strip()
                        # Strip outer quotes if simple string literal
                        if rhs_text.startswith('"') and rhs_text.endswith('"'):
                            return rhs_text[1:-1].replace('\\"', '"')
                        if rhs_text.startswith("''") and rhs_text.endswith("''"):
                            return rhs_text[2:-2]
                        return rhs_text  # multiline / interpolated — return raw
        return None

    def _find_binding_rhs_in_attrset(
        self, apply_node: Any, source: bytes, key: str
    ) -> str:
        """Return the raw RHS text of `<key> = <value>;` inside the apply's attrset."""
        attrset = self._find_attrset_child(apply_node)
        if not attrset:
            return ""
        for child in attrset.children:
            if child.type == "binding_set":
                for binding in child.children:
                    if binding.type != "binding":
                        continue
                    if self._binding_key_is(binding, source, key):
                        rhs = self._binding_rhs_node(binding)
                        if rhs is None:
                            return ""
                        return self.get_node_text(rhs, source).strip()
        return ""

    def _find_attrset_child(self, apply_node: Any) -> Optional[Any]:
        """Locate the attrset_expression child in mkOption's apply_expression."""
        # mkOption { ... } -> apply_expression with variable_expression + attrset_expression
        for child in apply_node.children:
            if child.type == "attrset_expression":
                return child
            # Recursive call: lib.mkOption { ... } -> apply_expression(select_expression, attrset_expression)
            #   Already handled because the attrset_expression is a direct child.
        return None

    def _binding_key_is(self, binding_node: Any, source: bytes, key: str) -> bool:
        for child in binding_node.children:
            if child.type == "attrpath":
                return self.get_node_text(child, source).strip() == key
        return False

    def _binding_rhs_node(self, binding_node: Any) -> Optional[Any]:
        """Return the RHS expression node of `<key> = <value>;`."""
        seen_eq = False
        for child in binding_node.children:
            if child.type == "=":
                seen_eq = True
                continue
            if seen_eq and child.type != ";":
                return child
        return None
