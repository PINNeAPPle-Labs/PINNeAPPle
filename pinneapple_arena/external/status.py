"""
Cheap, no-inference status check for a model — used by the catalog list/
filter view so the UI can show an install-state badge on every card without
actually running all 71 models on every page load.
"""

from __future__ import annotations

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.schema import ModelSpec

INSTALLED = "installed"
NOT_INSTALLED = "not_installed"
MANUAL_ONLY = "manual_only"
HEAVY = "heavy"


def install_state(spec: ModelSpec) -> str:
    if spec.manual_only:
        return MANUAL_ONLY
    if spec.heavy:
        # still report whether it happens to already be installed anyway
        if spec.import_name and is_installed(spec.import_name):
            return INSTALLED
        return HEAVY
    if spec.import_name and is_installed(spec.import_name):
        return INSTALLED
    return NOT_INSTALLED
