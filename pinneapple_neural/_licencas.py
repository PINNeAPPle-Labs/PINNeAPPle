"""License guards for optional third-party integrations.

Some integrations wrap code or pretrained weights whose licenses do not allow
production or commercial use. Those integrations stay available for research,
but they must never train, serve or deliver models or services to clients.

- Normal use: a ``ResearchOnlyWarning`` is emitted every time such a component is
  instantiated, stating the license restriction.
- Commercial mode (environment variable ``PINNEAPPLE_COMMERCIAL_MODE=1``, set by
  client-facing products such as PINNeAPPle-CFD): instantiation raises
  ``ResearchOnlyError`` before anything is imported or loaded.
"""
from __future__ import annotations

import os
import warnings

NOETHER_NOTICE = (
    "Noether (emmiai-noether) and the Emmi AI pretrained weights are RESEARCH ONLY in "
    "PINNeAPPle: the Noether framework is distributed under the Emmi Noether Public License, "
    "which prohibits production use, and the AB-UPT weights are CC BY-NC 4.0 (licenses checked "
    "on 2026-09-23). Do not use this integration to train, serve or deliver models or services "
    "to clients."
)


class ResearchOnlyWarning(UserWarning):
    """A research-only (non-production / non-commercial) component was used."""


class ResearchOnlyError(RuntimeError):
    """A research-only component was requested in commercial mode."""


def commercial_mode() -> bool:
    return os.environ.get("PINNEAPPLE_COMMERCIAL_MODE", "0") == "1"


def require_research_use(component: str, notice: str = NOETHER_NOTICE) -> None:
    """Block ``component`` in commercial mode; otherwise warn that it is research only."""
    if commercial_mode():
        raise ResearchOnlyError(f"{component} is not allowed with PINNEAPPLE_COMMERCIAL_MODE=1. {notice}")
    warnings.warn(f"{component}: {notice}", ResearchOnlyWarning, stacklevel=3)
