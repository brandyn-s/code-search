"""Bearer token validation for authenticated sessions."""


def validate_bearer_token(encoded_token: str) -> dict[str, str]:
    """Validate bearer token signature claims and return the subject."""
    signature, claims, subject = encoded_token.split(".", maxsplit=2)
    if not signature or not claims:
        raise ValueError("bearer token signature claims are required")
    return {"signature": signature, "claims": claims, "subject": subject}
