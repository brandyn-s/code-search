"""Privacy helpers for logs that describe user-supplied search text."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets


# Ephemeral on purpose: log fingerprints are useful for correlating records
# within one process, but cannot be used for an offline dictionary attack
# after the process exits.
_LOG_FINGERPRINT_KEY = secrets.token_bytes(32)


def query_text_logging_enabled() -> bool:
    """Return whether the operator explicitly opted into plaintext logs."""
    return os.environ.get("CODE_SEARCH_LOG_QUERY_TEXT", "off") == "on"


def query_fingerprint(query: str) -> str:
    """Return a process-local keyed fingerprint for search text."""
    return hmac.new(
        _LOG_FINGERPRINT_KEY,
        query.encode("utf-8", errors="surrogatepass"),
        hashlib.sha256,
    ).hexdigest()


def format_query_for_log(query: str, *, label: str = "query") -> str:
    """Return plaintext only under the exact opt-in, otherwise a fingerprint."""
    if query_text_logging_enabled():
        return query
    return (
        f"<{label} redacted hmac_sha256={query_fingerprint(query)} "
        f"length={len(query)}>"
    )


def redact_query_from_message(message: str, *queries: str) -> str:
    """Replace known query strings inside an otherwise useful message."""
    if query_text_logging_enabled():
        return message
    redacted = message
    for query in sorted(set(queries), key=len, reverse=True):
        if query:
            redacted = redacted.replace(query, format_query_for_log(query))
    return redacted


def format_query_exception_for_log(exc: BaseException) -> str:
    """Avoid exception messages that may echo request text by default."""
    if query_text_logging_enabled():
        return str(exc)
    return type(exc).__name__
