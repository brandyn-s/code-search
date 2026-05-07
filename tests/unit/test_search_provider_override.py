"""CS-2 (2026-05-06): tests for per-search provider override on
`search_code`.

Pins the `provider` parameter routing: when the caller passes
`provider="X"`, search_code must call `get_searcher(provider="X")`
rather than the default no-arg form (which routes through
_current_provider). This is the per-search override that enables
ensemble workflows over a project indexed with multiple providers.

Strategy: stub `get_searcher` and the downstream search call so we
verify the routing dispatch without spinning up FAISS / embedders.
"""

from unittest.mock import MagicMock

import pytest

from mcp_server.code_search_server import CodeSearchServer


@pytest.mark.unit
class TestSearchProviderOverride:
    """search_code(provider=X) routing dispatch."""

    @pytest.fixture
    def server(self, tmp_path, monkeypatch):
        """Return a CodeSearchServer wired to a temp storage dir.

        We don't actually index anything; the test stubs search dispatch
        before making the search call.
        """
        monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
        s = CodeSearchServer()
        s._current_project = "/fake/project"
        s._current_provider = "voyage"
        return s

    def _stub_searcher(self, server):
        """Replace get_searcher with a Mock that records calls.

        Returns the Mock so tests can assert on call args. The mock's
        return value is itself a Mock searcher whose .search() returns
        an empty list — enough for search_code to complete without
        exercising real index/embedder code.
        """
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
        get_searcher = MagicMock(return_value=mock_searcher)
        server.get_searcher = get_searcher
        # Also stub auto-reindex paths so they don't fire
        server._indexing_job = None
        return get_searcher, mock_searcher

    def test_no_provider_uses_default_searcher(self, server):
        """search_code() (no provider) calls get_searcher() with no
        kwargs — preserves existing default-routing behavior."""
        get_searcher, _ = self._stub_searcher(server)
        server.search_code(query="alert toast", auto_reindex=False)
        # Default path takes the no-arg branch
        get_searcher.assert_called_once_with()

    def test_explicit_provider_routes_via_provider_kw(self, server):
        """search_code(provider='voyage-context') calls get_searcher
        with provider='voyage-context'."""
        get_searcher, _ = self._stub_searcher(server)
        server.search_code(
            query="alert toast",
            auto_reindex=False,
            provider="voyage-context",
        )
        get_searcher.assert_called_once_with(provider="voyage-context")

    def test_provider_routing_passes_through_to_search(self, server):
        """Smoke test: full search_code call with provider override
        produces a JSON response (not an exception)."""
        _, mock_searcher = self._stub_searcher(server)
        out = server.search_code(
            query="login handler",
            k=3,
            auto_reindex=False,
            provider="voyage",
        )
        # Output is a JSON-encoded dict; asserting it parses is
        # enough to confirm the override path didn't break
        # downstream.
        import json
        data = json.loads(out)
        assert "query" in data
        assert data["query"] == "login handler"
        # The mock searcher was queried (via the override path)
        assert mock_searcher.search.called

    def test_two_calls_with_different_providers_route_correctly(self, server):
        """Calling search_code with provider='X' then provider='Y'
        routes each call to the right provider's searcher."""
        get_searcher, _ = self._stub_searcher(server)
        server.search_code(query="q1", auto_reindex=False, provider="voyage")
        server.search_code(query="q2", auto_reindex=False, provider="voyage-context")

        calls = get_searcher.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs == {"provider": "voyage"}
        assert calls[1].kwargs == {"provider": "voyage-context"}
