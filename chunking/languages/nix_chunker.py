"""Nix-specific tree-sitter based chunker."""

from typing import Any, Dict, List, Set

from chunking.base_chunker import LanguageChunker, TreeSitterChunk

# NixOS builtin functions that indicate specific patterns
_NIXOS_OPTION_FUNCS = {"mkOption", "mkEnableOption", "mkPackageOption", "mkSinkUndeclaredOptions"}
_NIXOS_CONDITIONAL_FUNCS = {"mkIf"}
_NIXOS_MERGE_FUNCS = {"mkMerge", "mkOverride", "mkDefault", "mkForce"}
_NIXOS_ALL_KNOWN_FUNCS = _NIXOS_OPTION_FUNCS | _NIXOS_CONDITIONAL_FUNCS | _NIXOS_MERGE_FUNCS

# Attrpath prefixes that indicate NixOS categories
_CATEGORY_PREFIXES = {
    "services.": "service",
    "systemd.services.": "service",
    "systemd.": "systemd",
    "networking.": "networking",
    "boot.": "boot",
    "users.": "user",
    "environment.": "environment",
    "security.": "security",
    "hardware.": "hardware",
    "fileSystems.": "filesystem",
    "programs.": "programs",
}


class NixChunker(LanguageChunker):
    """Nix-specific chunker using tree-sitter.

    Chunks Nix files on binding nodes inside let/attrset expressions.
    Recurses into let_expression to find inner bindings, unlike the
    base chunker which stops at the first splittable node.

    Detects NixOS-specific patterns (mkOption, mkIf, mkMerge, service
    definitions, etc.) and enriches chunk metadata for better search
    ranking and contextual headers.
    """

    MIN_CHUNK_LINES = 5

    def __init__(self):
        super().__init__("nix")

    def _get_splittable_node_types(self) -> Set[str]:
        return {"binding", "let_expression"}

    def _detect_nix_patterns(self, node: Any, source: bytes, name: str = "") -> Dict[str, Any]:
        """Detect NixOS-specific patterns in a binding's value expression.

        Checks for mkOption/mkIf/mkMerge calls (apply_expression with known
        function names) and categorizes bindings by their attrpath prefix.

        Returns dict of pattern metadata to merge into the chunk metadata.
        """
        result: Dict[str, Any] = {}

        # Detect NixOS function calls in the value expression.
        # Skip structural children (attrpath, "=", ";") and check
        # the first expression child, which is the binding's value.
        for child in node.children:
            if child.type in ("attrpath", "=", ";", "comment"):
                continue
            if child.type == "apply_expression":
                func_name = self._extract_apply_func_name(child, source)
                if func_name and func_name in _NIXOS_ALL_KNOWN_FUNCS:
                    result["nix_pattern"] = func_name
                    if func_name in _NIXOS_OPTION_FUNCS:
                        result["is_option_declaration"] = True
                    elif func_name in _NIXOS_CONDITIONAL_FUNCS:
                        result["is_conditional"] = True
                    elif func_name in _NIXOS_MERGE_FUNCS:
                        result["is_merge"] = True
            break  # Found the value expression, stop

        # Categorize by attrpath prefix
        if name:
            # Check "options." prefix for option declaration blocks
            if name.startswith("options.") or name == "options":
                result["nix_category"] = "option_declaration"
            else:
                for prefix, category in _CATEGORY_PREFIXES.items():
                    if name.startswith(prefix):
                        result["nix_category"] = category
                        break

            # Detect imports list
            if name == "imports":
                result["nix_category"] = "imports"

        return result

    def _extract_apply_func_name(self, apply_node: Any, source: bytes) -> str:
        """Extract the function name from an apply_expression node.

        In tree-sitter-nix, `mkOption { ... }` parses as:
            apply_expression
              variable_expression: "mkOption"
              attrset_expression: { ... }

        Curried calls like `lib.mkIf cfg.enable { ... }` parse as:
            apply_expression                    (outer)
              apply_expression                  (inner)
                select_expression: "lib.mkIf"
                variable_expression: "cfg.enable"
              attrset_expression: { ... }

        Also handles `lib.mkOption` (select_expression).
        """
        if not apply_node.children:
            return ""

        func_node = apply_node.children[0]

        # Recurse through nested apply_expressions (curried calls)
        while func_node.type == "apply_expression" and func_node.children:
            func_node = func_node.children[0]

        if func_node.type == "variable_expression":
            return self.get_node_text(func_node, source)
        elif func_node.type == "select_expression":
            # lib.mkOption -> extract the last segment
            for child in reversed(func_node.children):
                if child.type == "attrpath" or child.type == "identifier":
                    text = self.get_node_text(child, source)
                    # attrpath may have dots; take the last segment
                    return text.split(".")[-1] if "." in text else text
            # Fallback: get the full text and take last dotted segment
            full = self.get_node_text(func_node, source)
            return full.split(".")[-1] if "." in full else full

        return ""

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        if node.type == "binding":
            name = ""
            for child in node.children:
                if child.type == "attrpath":
                    name = self.get_node_text(child, source)
                    metadata["name"] = name
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

            # Detect NixOS-specific patterns
            nix_patterns = self._detect_nix_patterns(node, source, name)
            metadata.update(nix_patterns)

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
                    end_line=len(source_code.splitlines()),
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
