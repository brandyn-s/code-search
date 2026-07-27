"""Storage boundary for append-only security evidence."""


def persist_immutable_audit_record(
    record: dict[str, str],
    storage: list[dict[str, str]],
) -> int:
    """Persist an immutable audit record in storage."""
    storage.append(dict(record))
    return len(storage) - 1
