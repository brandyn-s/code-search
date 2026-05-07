"""Unit tests for TypeScript tree-sitter chunking — CS-1 enhancements.

Pins the post-CS-1 behavior:
  - Arrow-bound `export const Foo = () => {...}` produces a chunk with
    `name="Foo"` (vs pre-CS-1 where the parent declaration had no name)
  - PascalCase arrow bindings get `is_component=True`
  - `use[A-Z]...` bindings get `is_hook=True`
  - Existing function_declaration / class_declaration chunking unchanged
"""

import pytest

from chunking.languages import TypeScriptChunker


@pytest.mark.unit
class TestTypeScriptChunkerCS1:
    """CS-1: TypeScript chunker enhancements (component/hook detection)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            self.chunker = TypeScriptChunker()
        except ValueError:
            pytest.skip("tree-sitter-typescript not installed")

    def _name_to_metadata(self, chunks):
        """Helper: build {name: metadata} dict for chunks that have a name."""
        out = {}
        for c in chunks:
            name = c.metadata.get('name')
            if name:
                out[name] = c.metadata
        return out

    def test_function_declaration_still_chunks(self):
        """Sanity: existing function_declaration handling unchanged."""
        code = '''
function normalFunction() {
    return 42;
}
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'normalFunction' in meta, (
            "function_declaration must still produce a named chunk; "
            f"got names={list(meta.keys())}"
        )

    def test_arrow_component_named_chunk(self):
        """`export const AlertToast = () => {...}` produces a chunk
        named AlertToast (pre-CS-1 the lexical_declaration had no
        chunk and the inner arrow had no name either)."""
        code = '''
export const AlertToast = () => {
    return null;
};
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'AlertToast' in meta, (
            f"arrow-bound component must produce a named chunk; got {list(meta.keys())}"
        )

    def test_arrow_component_flagged_is_component(self):
        """PascalCase arrow binding gets is_component=True."""
        code = '''
const PaymentButton = (props) => {
    return null;
};
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'PaymentButton' in meta
        assert meta['PaymentButton'].get('is_component') is True, (
            f"PascalCase arrow binding must be flagged is_component; "
            f"got metadata={meta['PaymentButton']}"
        )

    def test_lowercase_arrow_not_component(self):
        """Utility arrow binding (lowercase) is NOT a component."""
        code = '''
const formatPrice = (cents) => `$${cents / 100}`;
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'formatPrice' in meta
        assert meta['formatPrice'].get('is_component') is not True, (
            "lowercase arrow binding must NOT be flagged is_component"
        )

    def test_use_prefix_arrow_flagged_is_hook(self):
        """`useFetchData` arrow binding gets is_hook=True."""
        code = '''
const useFetchData = (url) => {
    const [data, setData] = useState(null);
    return data;
};
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'useFetchData' in meta
        assert meta['useFetchData'].get('is_hook') is True, (
            f"use[A-Z]... binding must be flagged is_hook; "
            f"got metadata={meta['useFetchData']}"
        )

    def test_function_decl_with_use_prefix_is_hook(self):
        """`function useEffect()` (declaration form) is also a hook."""
        code = '''
function useTitle(title) {
    document.title = title;
}
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'useTitle' in meta
        assert meta['useTitle'].get('is_hook') is True

    def test_lowercase_use_not_hook(self):
        """`useful` (no uppercase after `use`) is NOT a hook."""
        code = '''
const useful = (x) => x * 2;
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'useful' in meta
        assert meta['useful'].get('is_hook') is not True, (
            "`useful` (no uppercase after `use`) must NOT be flagged is_hook"
        )

    def test_class_declaration_unchanged(self):
        """Sanity: class_declaration metadata is not affected by CS-1."""
        code = '''
export class MyService {
    method() {
        return "ok";
    }
}
'''
        chunks = self.chunker.chunk_code(code)
        meta = self._name_to_metadata(chunks)
        assert 'MyService' in meta
        # Class is NOT a component (component is restricted to arrow
        # bindings — function declarations and class declarations are
        # distinguishable from React components by their syntactic shape)
        assert meta['MyService'].get('is_component') is not True
