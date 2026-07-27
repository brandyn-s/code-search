"""Small comma-separated row parser."""


def parse_csv_row(line: str) -> list[str]:
    """Split one comma-separated row and trim surrounding whitespace."""
    return [column.strip() for column in line.split(",")]
