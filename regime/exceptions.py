"""Typed deterministic market-regime failures."""


class RegimeError(ValueError):
    """Base class for deterministic regime-layer errors."""


class RegimeFeatureError(RegimeError):
    """Raised when causal regime features cannot be produced safely."""


class RegimeClassificationError(RegimeError):
    """Raised when a regime assessment cannot be classified."""
