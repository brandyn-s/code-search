"""Invoice arithmetic kept separate from payment processing."""


def calculate_invoice_total(
    subtotal: float,
    discount: float,
    tax: float,
) -> float:
    """Calculate invoice total after discount and tax adjustments."""
    discounted_total = subtotal - discount
    return discounted_total + tax
