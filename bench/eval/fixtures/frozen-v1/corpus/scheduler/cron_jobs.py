"""Recurring task scheduling helpers."""


def next_hourly_run(current_hour: int) -> int:
    """Return the next hour on a twenty-four-hour clock."""
    return (current_hour + 1) % 24
