"""Planner capability for the Ahmed-body design flow (``pinneapple_flow/flows/ahmed_body_end_to_end.json``, its
``study.design`` node in :mod:`pinneapple_flow.nodes.body_study`): a goal such as "reduzir o arrasto do carro" or
"reduce the car's drag" proposes the least-drag slant and clearance of a car-like bluff body from the OpenFOAM
campaign, surrogates with calibrated intervals, and the design re-verified against the CFD (or stated unverified)."""
from __future__ import annotations

# a car or a road vehicle named in the goal ("car " / "car'" / "cars": "car" alone would also match "carga", "carbon")
_BODY = ("carro", "car ", "car'", "cars", "automóvel", "automovel", "veículo", "veiculo", "vehicle", "ahmed",
         "bluff body", "corpo rombudo")
_DRAG = ("arrasto", "drag", "aerodinâmic", "aerodinamic", "aerodynamic")


def _register_capabilities():
    from ..planner.extensions import Capability, bundled_flow, register_capability
    register_capability(Capability(
        id="ahmed.design", title="Least-drag car body (Ahmed body design, re-verified against CFD)",
        domain="aerodynamics",
        keywords={"arrasto": 2, "drag": 2, "aerodinâmic": 1, "aerodinamic": 1, "aerodynamic": 1,
                  **{k: 1 for k in _BODY if k != "ahmed"}, "ahmed": 2,
                  "reduzir": 1, "reduce": 1, "diminuir": 1, "lower": 1, "minimiz": 1, "minimis": 1, "otimiz": 1,
                  "optimi": 1, "slant": 1, "inclinação traseira": 1, "rear slant": 1, "clearance": 1,
                  "altura ao solo": 1},
        requires_all=(_BODY, _DRAG), min_score=3,
        nodes=("study.ahmed_geometry", "study.cfd_campaign", "study.grid_validation", "study.dataset",
               "study.surrogates", "study.design"),
        downstream=("study.inverse", "study.twin", "study.report"),
        build=lambda text, det: bundled_flow("ahmed_body_end_to_end"),
        inputs=("the speed (inside the sampled 20-60 m/s)", "a lift limit, if any", "the CFD campaign (recorded, or "
                "OpenFOAM to run new cases)"),
        references=("menter1994sst",),
        example="end_to_end",
        description="pinneapple_autolab.domains.ahmed_study.design through the study.design node of the Ahmed-body "
                    "flow; the optimum is re-verified by OpenFOAM or the nearest recorded run, else stated unverified"))


_register_capabilities()
