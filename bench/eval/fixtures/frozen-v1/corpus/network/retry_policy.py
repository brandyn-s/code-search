"""Network retry timing for transient service failures."""


def exponential_retry_backoff(attempt: int, timeout: float) -> float:
    """Calculate exponential retry backoff bounded by the network timeout."""
    delay = float(2 ** max(attempt, 0))
    return min(delay, timeout)
