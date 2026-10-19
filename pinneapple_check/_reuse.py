"""Load single modules of other PINNeAPPle packages without running their package ``__init__``.

Several packages (``pinneapple_data``, ``pinneapple_physics``, ``pinneapple_analysis``, ``pinneapple_catalog``)
import torch and the model zoo in their ``__init__``. The modules the checker needs from them (OpenFOAM readers,
units, the training envelope, field metrics, the method catalog) are numpy-only, so they are loaded here under a
private alias: a CI linter should start in a fraction of a second and run without torch. If the real package is
already imported, its module is used as is.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

_ALIAS = "_pinneapple_check_reuse"


def leaf(dotted: str) -> types.ModuleType:
    """``leaf("pinneapple_data.cae.foam")``: the module, imported without its parents' ``__init__``."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    top, *rest = dotted.split(".")
    if top in sys.modules:                                   # the package is already loaded: nothing to save
        return importlib.import_module(dotted)
    spec = importlib.util.find_spec(top)                     # locating a top-level package does not execute it
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"package {top} is not installed")
    base = list(spec.submodule_search_locations)[0]
    if _ALIAS not in sys.modules:
        root = types.ModuleType(_ALIAS)
        root.__path__ = []
        sys.modules[_ALIAS] = root
    name, path = _ALIAS, base
    for i, part in enumerate([top, *rest[:-1]]):
        name = f"{name}.{part}"
        if i:
            path = os.path.join(path, part)
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [path]
            pkg.__package__ = name
            sys.modules[name] = pkg
    return importlib.import_module(f"{_ALIAS}.{dotted}")


def foam():
    return leaf("pinneapple_data.cae.foam")


def cae_io():
    return leaf("pinneapple_data.cae.io")


def cae_geometry():
    return leaf("pinneapple_data.cae.geometry")


def units():
    return leaf("pinneapple_data.physical_units")


def envelope_cls():
    return leaf("pinneapple_physics.physics_audit.assumptions").Envelope


def field_metrics():
    return leaf("pinneapple_analysis.validation.field_metrics")


def spectral():
    return leaf("pinneapple_analysis.validation.spectral")


def metamorphic():
    return leaf("pinneapple_analysis.validation.metamorphic")


def dataset_audit():
    return leaf("pinneapple_data.audit")


def method_catalog():
    return leaf("pinneapple_catalog.methods")


def trust_stages():
    return leaf("pinneapple_lab.curation").STAGES


def darcy_checks():
    return leaf("pinneapple_physics.darcy_checks")


def maxwell_checks():
    return leaf("pinneapple_physics.maxwell_checks")
