"""In-memory numeric measurement aggregation."""


def average_measurement(samples: list[float]) -> float:
    """Return the arithmetic mean of collected samples."""
    if not samples:
        return 0.0
    return sum(samples) / len(samples)
