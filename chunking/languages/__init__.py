"""Language-specific tree-sitter based chunkers."""

from functools import lru_cache

from chunking.languages.python_chunker import PythonChunker
from chunking.languages.javascript_chunker import JavaScriptChunker
from chunking.languages.jsx_chunker import JSXChunker
from chunking.languages.typescript_chunker import TypeScriptChunker
from chunking.languages.svelte_chunker import SvelteChunker
from chunking.languages.go_chunker import GoChunker
from chunking.languages.rust_chunker import RustChunker
from chunking.languages.java_chunker import JavaChunker
from chunking.languages.markdown_chunker import MarkdownChunker
from chunking.languages.c_chunker import CChunker
from chunking.languages.cpp_chunker import CppChunker
from chunking.languages.csharp_chunker import CSharpChunker
from chunking.languages.nix_chunker import NixChunker
from chunking.languages.toml_chunker import TomlChunker
from chunking.languages.yaml_chunker import YamlChunker
from chunking.languages.hcl_chunker import HclChunker


# Cached factory function for C++ chunker (shared across multiple extensions)
@lru_cache(maxsize=1)
def _get_cpp_chunker() -> CppChunker:
    return CppChunker()


# Map file extensions to chunker classes and language names
LANGUAGE_MAP = {
    ".py": ("python", PythonChunker),
    ".js": ("javascript", JavaScriptChunker),
    ".jsx": ("jsx", JSXChunker),
    ".ts": ("typescript", lambda: TypeScriptChunker(use_tsx=False)),
    ".tsx": ("tsx", lambda: TypeScriptChunker(use_tsx=True)),
    ".svelte": ("svelte", SvelteChunker),
    ".go": ("go", GoChunker),
    ".rs": ("rust", RustChunker),
    ".java": ("java", JavaChunker),
    ".md": ("markdown", MarkdownChunker),
    ".c": ("c", CChunker),
    ".cpp": ("cpp", _get_cpp_chunker),
    ".cc": ("cpp", _get_cpp_chunker),
    ".cxx": ("cpp", _get_cpp_chunker),
    ".c++": ("cpp", _get_cpp_chunker),
    ".cs": ("csharp", CSharpChunker),
    ".nix": ("nix", NixChunker),
    ".toml": ("toml", TomlChunker),
    ".yml": ("yaml", YamlChunker),
    ".yaml": ("yaml", YamlChunker),
    ".tf": ("hcl", HclChunker),
}
