"""The public namespace promised by #201 is the one that resolves today.

The map in docs/architecture/public_namespace.md is a target. This test pins the
part that exists, so a rename cannot pass silently, and pins the part that does
not, so the document cannot drift into claiming a namespace that was never built.
"""
from __future__ import annotations

import pytest

import pinneapple as pp

# Namespaces a user types today and that resolve.
RESOLVES = (
    "data",
    "models",
    "train",
    "solvers",
    "viz",
    "decision",
    "decide",
    "core",
    "cae",
    "geom",
    "env",
    "pinn",
    "symbolic",
    "presets",
    "uncertainty",
    "sensitivity",
    "inverse",
    "artifact",
    # reproduced / divergent / refused for a recorded run (#1480)
    "experiments",
    # LinearOperator, Arnoldi, POD, modal analysis, conditioning (#202)
    "linalg",
    # surrogate in the loop with a verification budget (#297)
    "optimize",
)

# Namespaces in the target map that do not exist yet. Each is a line of the
# migration, not an omission: the test fails the day one is added, so the
# document and the code change in the same pull request.
NOT_YET = (
    "physics",
    "nn",
    "optim",
    "sparse",
    "geometry",
    "mesh",
    "solver",
    "surrogate",
    "analysis",
    "workflows",
)


@pytest.mark.parametrize("name", RESOLVES)
def test_documented_namespace_resolves(name):
    assert getattr(pp, name) is not None


@pytest.mark.parametrize("name", NOT_YET)
def test_namespace_in_the_target_map_is_still_missing(name):
    with pytest.raises(AttributeError):
        getattr(pp, name)


def test_front_doors_are_method_agnostic():
    """solve/compare/validate/design are the doors; a PINN is one method inside."""
    for name in ("solve", "compare", "list_methods", "register_method"):
        assert hasattr(pp, name), name
