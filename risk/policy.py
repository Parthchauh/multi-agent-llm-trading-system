"""Policy entry point for deterministic Sprint 6 risk controls.

``RiskPolicy`` lives in :mod:`risk.models` with the rest of the immutable
contracts.  This module preserves the conventional package layout without
creating a second policy implementation.
"""

from risk.models import RiskPolicy

__all__ = ["RiskPolicy"]
