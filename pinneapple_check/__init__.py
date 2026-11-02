"""pinneapple check: a physics linter for CI.

Fails a pull request when a simulation result or a Physics-AI model violates physics: conservation, units,
boundary and initial conditions, PDE residual, bounds, out-of-distribution use, regression against a stored
reference, and integrity of the artefacts. Command line: ``pinneapple check [paths]`` (or
``python -m pinneapple_check`` / ``python -m pinneapple.check``); GitHub Action: ``action.yml`` at the root of the
repository.

From Python::

    from pinneapple_check import run
    report = run(["cases/"])
    assert not report.errors, [o.message for o in report.errors]

The package imports only NumPy at start-up (torch only to load a ``.pt`` checkpoint), and reuses the OpenFOAM and
unit readers of ``pinneapple_data``, the training envelope of ``pinneapple_physics.physics_audit``, field metrics
of ``pinneapple_analysis``, the method catalog, ``pinneapple_security`` for hashes, signatures and checkpoint scans,
and the trust-card stages of ``pinneapple_lab``.
"""
from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v
    try:
        __version__ = _v("pinneapple")
    except PackageNotFoundError:                                     # pragma: no cover - running from a checkout
        __version__ = "0.0.0+local"
except ImportError:                                                  # pragma: no cover
    __version__ = "0.0.0+local"

from .config import Config, ConfigError, load_config  # noqa: E402
from .engine import run  # noqa: E402
from .result import Location, Outcome, Report  # noqa: E402
from .rules import RULES, Rule, get_rule, select_rules  # noqa: E402

__all__ = ["run", "Report", "Outcome", "Location", "Config", "ConfigError", "load_config", "RULES", "Rule",
           "get_rule", "select_rules", "__version__"]

# experimental (labs): see pinneapple_stability and docs/api_stability.md; nothing here changes behaviour
from pinneapple_stability import warn_experimental as _warn_experimental  # noqa: E402

_warn_experimental(__name__)
