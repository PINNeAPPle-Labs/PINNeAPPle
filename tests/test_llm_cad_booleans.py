"""A drafted part with a hole is built with a real boolean or refused (PINNeAPPle-apps#5): without a boolean engine
build_mesh concatenated the hole onto the part, so a washer and a plate with a hole came out with the hole's volume
added. References: the exact volumes pi (R^2 - r^2) h and a h w - pi r^2 h, within the faceting error."""
import math

import pytest

from pinneapple_design.geometry.gen import primitives as P
from pinneapple_llm.cad_draft import CadRecipeResult, measure_part

WASHER = {"boolean": {"op": "cut",
                      "base": {"name": "cylinder", "params": {"radius": 10.0, "height": 2.0, "sections": 256}},
                      "tool": {"name": "cylinder", "params": {"radius": 4.0, "height": 4.0, "sections": 256}}}}
PLATE = {"boolean": {"op": "cut",
                     "base": {"name": "box", "params": {"extents": [40.0, 30.0, 5.0]}},
                     "tool": {"name": "cylinder", "params": {"radius": 5.0, "height": 10.0, "sections": 256}}}}
EXACT = {"washer": math.pi * (10 ** 2 - 4 ** 2) * 2.0, "plate": 40 * 30 * 5 - math.pi * 5 ** 2 * 5}


@pytest.mark.parametrize("name,recipe", [("washer", WASHER), ("plate", PLATE)])
def test_parts_with_holes_have_the_requested_volume(name, recipe):
    if not P._boolean_engines_available():
        pytest.skip("no boolean engine (manifold3d, extra geometry)")
    m = measure_part(CadRecipeResult(kind="mesh_recipe", recipe=recipe))
    assert m["volume"] == pytest.approx(EXACT[name], rel=0.005) and m["watertight"]


@pytest.mark.parametrize("recipe", [WASHER, PLATE])
def test_without_a_boolean_engine_a_hole_is_refused_not_concatenated(recipe, monkeypatch):
    monkeypatch.setattr(P, "_boolean_engines_available", lambda: ())
    with pytest.raises(RuntimeError, match="boolean engine"):
        measure_part(CadRecipeResult(kind="mesh_recipe", recipe=recipe))
    with pytest.warns(UserWarning, match="concatenate"):                     # the generic factory still falls back
        P.build_mesh("box", extents=[1, 1, 1], boolean={"op": "cut", "other": {"name": "sphere", "radius": 0.2}})
