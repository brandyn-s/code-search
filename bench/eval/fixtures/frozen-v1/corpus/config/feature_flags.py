"""Feature-flag configuration loaded from an environment mapping."""


def load_feature_flag(
    flag_name: str,
    environment: dict[str, str],
) -> bool:
    """Load feature flag configuration from the supplied environment."""
    configured = environment.get(flag_name, "off")
    return configured.strip().lower() in {"1", "on", "true"}
