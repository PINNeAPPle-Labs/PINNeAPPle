"""presets.tag_geometry.tag_solve_kwargs: nothing for untagged presets, the fixture geometry for tagged ones that have
one, KeyError (no invented geometry) for the rest."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import pinneapple_physics as pp  # noqa: E402
from pinneapple_physics.pde_environment.presets.tag_geometry import (  # noqa: E402
    NOT_FIXABLE_WITHOUT_REAL_GEOMETRY,
    tag_solve_kwargs,
)


def test_untagged_preset_needs_nothing():
    assert tag_solve_kwargs("burgers_1d", pp.get_preset("burgers_1d")) == {}


def test_tagged_preset_with_a_fixture_gets_its_boundary():
    kw = tag_solve_kwargs("laplace_2d", pp.get_preset("laplace_2d"), n_bc_per_face=20)
    assert kw["x_bc"].shape[0] == 80 and kw["mask_u_boundary"].all() and "x_col" not in kw


def test_tagged_preset_without_a_fixture_is_refused():
    name = sorted(NOT_FIXABLE_WITHOUT_REAL_GEOMETRY)[0]
    with pytest.raises(KeyError):
        tag_solve_kwargs(name, pp.get_preset(name))
