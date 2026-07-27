"""Daily summary rendering for human-readable reports."""


def render_daily_summary(items: list[str]) -> str:
    """Render sorted items as a newline-delimited summary."""
    return "\n".join(sorted(items))
