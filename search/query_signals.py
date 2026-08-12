"""Oracle-blind code signals used to keep issue-style queries precise."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "been",
        "before",
        "being",
        "below",
        "bug",
        "cannot",
        "could",
        "current",
        "default",
        "does",
        "from",
        "have",
        "here",
        "into",
        "least",
        "more",
        "not",
        "only",
        "security",
        "should",
        "shown",
        "summary",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "through",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)
_CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_KNOWN_FILE_SUFFIXES = frozenset(
    {
        "c",
        "cc",
        "cpp",
        "go",
        "h",
        "hpp",
        "java",
        "js",
        "json",
        "jsx",
        "md",
        "py",
        "rb",
        "rs",
        "toml",
        "ts",
        "tsx",
        "yaml",
        "yml",
    }
)


@dataclass(frozen=True)
class QuerySignals:
    """Stable query features that do not depend on an answer oracle."""

    original_query: str
    explicit: bool
    identifiers: tuple[str, ...]
    path_hints: tuple[str, ...]
    owner_members: tuple[tuple[str, str], ...]
    lexical_terms: tuple[str, ...]


def _dedupe_append(values: list[str], value: str) -> None:
    cleaned = value.strip().strip("`'\"()[]{}<>.,:;!?*")
    if not cleaned or cleaned.casefold() in {item.casefold() for item in values}:
        return
    values.append(cleaned)


def _is_owner_member(value: str) -> bool:
    parts = value.split(".")
    return (
        len(parts) == 2
        and parts[1].casefold() not in _KNOWN_FILE_SUFFIXES
        and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts)
    )


def extract_query_signals(query: str, *, limit: int = 12) -> QuerySignals:
    """Extract paths and identifier-shaped terms from natural-language text."""

    normalized = query.replace("\r\n", "\n").replace("\r", "\n").strip()
    identifiers: list[str] = []
    paths: list[str] = []
    owner_members: list[tuple[str, str]] = []
    lexical_terms: list[str] = []

    def add_identifier(value: str) -> None:
        cleaned = value.strip().removesuffix("()")
        if not cleaned:
            return
        _dedupe_append(identifiers, cleaned)
        if _is_owner_member(cleaned):
            owner, member = cleaned.split(".", 1)
            pair = (owner.casefold(), member.casefold())
            if pair not in owner_members:
                owner_members.append(pair)
            _dedupe_append(lexical_terms, owner)
            _dedupe_append(lexical_terms, member)
        else:
            _dedupe_append(lexical_terms, cleaned)

    for match in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", normalized):
        for token in _CODE_TOKEN.findall(match.group(1)):
            add_identifier(token)

    for match in re.finditer(
        r"https://github\.com/[^\s)]+/blob/[0-9a-f]{40}/([^#\s)]+)",
        normalized,
        flags=re.IGNORECASE,
    ):
        path = match.group(1).strip(".,:;!?")
        _dedupe_append(paths, path)
        _dedupe_append(lexical_terms, path)

    without_urls = re.sub(r"https?://\S+", " ", normalized)
    without_code = re.sub(r"(?<!`)`[^`\n]+`(?!`)", " ", without_urls)
    for token in _CODE_TOKEN.findall(without_code):
        token = token.strip("._-")
        explicit = (
            "_" in token
            or "." in token
            or (token.isupper() and 2 <= len(token) <= 16)
            or bool(re.search(r"[a-z][A-Z]", token))
        )
        if explicit:
            add_identifier(token)

    has_explicit = bool(identifiers or paths)
    if has_explicit:
        title = normalized.splitlines()[0] if normalized else ""
        fallback_text = title + "\n" + without_code
        for token in _CODE_TOKEN.findall(fallback_text):
            cleaned = token.strip("._-")
            if len(cleaned) < 3 or cleaned.casefold() in _STOP_WORDS:
                continue
            if "." in cleaned and _is_owner_member(cleaned):
                for part in cleaned.split("."):
                    _dedupe_append(lexical_terms, part)
            else:
                _dedupe_append(lexical_terms, cleaned)
            if len(lexical_terms) >= limit:
                break

    return QuerySignals(
        original_query=query,
        explicit=has_explicit,
        identifiers=tuple(identifiers),
        path_hints=tuple(paths),
        owner_members=tuple(owner_members),
        lexical_terms=tuple(lexical_terms[:limit]),
    )


def build_lexical_query(query: str) -> str:
    """Return a compact BM25 query only when explicit code signals exist."""

    signals = extract_query_signals(query)
    if not signals.explicit or not signals.lexical_terms:
        return query
    return " ".join(signals.lexical_terms)


def _contains(value: str, token: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(token.casefold())}(?![A-Za-z0-9_])",
            value.casefold(),
        )
    )


def calculate_signal_boost(
    signals: QuerySignals,
    *,
    relative_path: str,
    name: str | None,
    parent_name: str | None,
    full_content: str,
) -> float:
    """Score exact path and owner/member evidence without inventing coordinates."""

    if not signals.explicit:
        return 1.0

    path = relative_path.replace("\\", "/").casefold()
    candidate_name = (name or "").casefold()
    candidate_parent = (parent_name or "").casefold()
    content = full_content.casefold()
    boost = 1.0

    for hint in signals.path_hints:
        normalized_hint = hint.replace("\\", "/").casefold()
        if path == normalized_hint or path.endswith("/" + normalized_hint):
            boost = max(boost, 3.0)
        elif PurePosixPath(path).name == PurePosixPath(normalized_hint).name:
            boost = max(boost, 1.6)

    path_stem = PurePosixPath(path).stem
    for owner, member in signals.owner_members:
        owner_match = owner in {candidate_name, candidate_parent, path_stem}
        member_match = member in candidate_name or member in content
        if owner_match and member_match:
            boost = max(boost, 1.8)
        elif member_match:
            boost = max(boost, 1.12)

    matched = 0
    for identifier in signals.identifiers:
        if _is_owner_member(identifier):
            continue
        if (
            _contains(candidate_name, identifier)
            or _contains(candidate_parent, identifier)
            or _contains(path, identifier)
            or _contains(content, identifier)
        ):
            matched += 1
            if identifier.casefold() in {candidate_name, candidate_parent, path_stem}:
                boost = max(boost, 1.4)
    if matched:
        # Identifier-shaped terms are far less ambiguous than prose. One
        # exact source occurrence should decisively beat a semantic neighbor
        # that contains none. Keep the ceiling bounded so retrieval remains a
        # blend instead of becoming a second keyword-only mode.
        boost = max(boost, 1.0 + min(matched, 1) * 2.0)

    return min(boost, 3.0)
