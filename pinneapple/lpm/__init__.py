"""Native Large Physics Model campaign interfaces for PINNeAPPle.

This package is intentionally lightweight on import: solvers and training
backends are resolved only when a campaign is executed.
"""
from .schema import Campaign, CampaignValidationError

__all__ = ["Campaign", "CampaignValidationError"]
