"""What PINNeAPPle promises (issue #1127, toward 1.0): the stable **core**, and everything else.

A package of its own with no dependencies, so that a light experimental package can say what it is without
importing the heavy ``pinneapple_core`` (0.6 s and more).

* **core**: module prefixes and ``pp.*`` names covered by semantic versioning. A breaking change to them needs a
  deprecation period (``pinneapple_core.deprecation``) and a changelog entry.
* **experimental**: every other public module. Its code, tests and development are unchanged; it only carries no
  compatibility promise between versions. The packages in :data:`LABS` (far from the core in usage and test
  coverage) also say so with an :class:`ExperimentalWarning` the first time they are imported.

Nothing is removed by being experimental. A module is **promoted** to the core in any minor release once none of
its public tools is "discovered" in the Studio catalog (each is tested or verified), something outside its own
tests uses it (an example, an app, a documented task), and its names are in the API snapshot. See
``docs/api_stability.md``.

The lists below were proposed from measured usage (imports in examples, apps, templates and notebooks; ``pp.*``
uses in docs and examples; the tools the engineering app calls) and are the place to change the decision.
"""
from __future__ import annotations

import os
import warnings

__all__ = ["CORE", "CORE_PP", "LABS", "ExperimentalWarning", "stability", "is_core", "warn_experimental"]

CORE: tuple[str, ...] = (
    "pinneapple",                                       # the pp facade module itself (names: CORE_PP)
    "pinneapple_core.jsonable",
    "pinneapple_core.onboarding",
    "pinneapple_stability",
    "pinneapple_data.contracts",
    "pinneapple_data.physical_sample",
    "pinneapple_data.cae",
    "pinneapple_physics.pinn_solver",
    "pinneapple_physics.pde_environment",
    "pinneapple_physics.closed_form",
    "pinneapple_physics.solving",
    "pinneapple_neural.architectures.pinns",
    "pinneapple_neural.architectures.neural_operators",
    "pinneapple_neural.architectures.rom",
    "pinneapple_neural.trainer",
    "pinneapple_design.geometry.core",
    "pinneapple_design.geometry.io",
    "pinneapple_design.geometry.brep",
    "pinneapple_design.geometry.mesh",
    "pinneapple_analysis.validation",
    "pinneapple_analysis.trust",
    "pinneapple_presets",
    "pinneapple_tools.visualization.studio.fieldview",
)

# the pp.* names in the core; other pp.* names work as before, without the promise
CORE_PP: tuple[str, ...] = (
    "solve", "zoo", "validate", "trust_card", "datasets", "benchmark", "viz", "compare", "presets", "geometries",
    "design", "doctor", "recommend", "card", "new", "closed_form", "uncertainty", "build_model", "load_checkpoint",
    "get_preset",
    "list_presets", "list_methods", "list_models", "to_jsonable", "__version__",
)

# whole packages far from the core: low use outside their own tests and a high share of untested tools
LABS: tuple[str, ...] = (
    "pinneapple_arena", "pinneapple_worldmodel", "pinneapple_quantum", "pinneapple_app", "pinneapple_decision",
    "pinneapple_check", "pinneapple_condense", "pinneapple_reason", "pinneapple_orchestration", "pinneapple_query",
    "pinneapple_pdb", "pinneapple_autolab", "pinneapple_systems.sports",
)

QUIET_ENV = "PINNEAPPLE_QUIET_EXPERIMENTAL"


class ExperimentalWarning(UserWarning):
    """An experimental module was imported: it works, but its API may change between versions without notice."""


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def is_core(module: str, name: str | None = None) -> bool:
    """Whether ``module`` (and, for the ``pinneapple`` facade, ``name``) is in the core."""
    if module == "pinneapple":
        return name is None or name in CORE_PP
    return any(module == p or module.startswith(p + ".") for p in CORE if p != "pinneapple")


def stability(module: str, name: str | None = None) -> str:
    """``"core"`` or ``"experimental"``; for a labs package, ``"experimental (labs)"``."""
    if is_core(module, name):
        return "core"
    return "experimental (labs)" if _under(module, LABS) else "experimental"


_warned: set[str] = set()


def _importer() -> str:
    """The module that imported the labs package: the first frame outside the import machinery, this module and the
    package being initialised."""
    import sys
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__", "")
        if not (name.startswith(("importlib", "_frozen_importlib")) or name == __name__
                or f.f_code.co_filename.endswith("__init__.py") and name.startswith(LABS)):
            return name
        f = f.f_back
    return ""


def warn_experimental(package: str) -> None:
    """Called from a labs package's ``__init__``: one :class:`ExperimentalWarning` per package and process, when user
    code imports it. An import from inside the library (a core preset that uses a labs package for one case) does not
    warn: the user did not choose the experimental API.

    Silence it with ``PINNEAPPLE_QUIET_EXPERIMENTAL=1`` or
    ``warnings.filterwarnings("ignore", category=ExperimentalWarning)``."""
    if package in _warned or os.environ.get(QUIET_ENV, "").strip() not in ("", "0"):
        return
    if _importer().split(".")[0].startswith("pinneapple"):
        return
    _warned.add(package)
    warnings.warn(f"{package} is experimental: it works and is tested where its catalog says so, but its API may "
                  "change between versions without a deprecation period (see docs/api_stability.md; silence with "
                  f"{QUIET_ENV}=1)", ExperimentalWarning, stacklevel=3)
