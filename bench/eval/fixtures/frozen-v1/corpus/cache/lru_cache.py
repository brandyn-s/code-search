"""Bounded least-recently-used cache utilities."""


def evict_oldest_entry(entries: list[str], capacity: int) -> list[str]:
    """Drop oldest entries until the cache fits its capacity."""
    overflow = max(len(entries) - capacity, 0)
    return entries[overflow:]
