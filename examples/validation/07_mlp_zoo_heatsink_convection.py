"""Classical ML on simulation tables: the zoo's MLP for forced-convection heat sinks (``heatsink_forced_mlp``).

1. Load the model (its card states the training data, the validity range and the reference error).
2. ``pp.validate(model, query=...)``: validity range, training envelope, the zoo gate against the held-out CFD
   cases, the calibrated bound and the metamorphic relations of its card.
3. Trust card and ``pinneapple check`` outputs (``examples/validation/pinneapple-check.toml`` runs the same model
   in CI with ``zoo = "heatsink_forced_mlp"``).
4. Failure caught: a query at 0.2 m/s air velocity and 600 W, outside the range the model was trained on
   (0.5-6 m/s, 10-400 W; ``applicability.range``).

Run: ``python examples/validation/07_mlp_zoo_heatsink_convection.py``.
"""
from __future__ import annotations

import argparse
import warnings

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude

FAMILY, PHYSICS, CATCHES = "classical-ml", "heat", "applicability.range"


def run(smoke: bool = True, out_dir=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pp.zoo.load("heatsink_forced_mlp")
        test = pp.datasets.load("heatsink_forced_convection", split="test").to_pandas()
        query = test[[i["name"] for i in model.card.inputs]].head(8 if smoke else 64)
        good = pp.validate(model, query=query.to_dict("list"), level="standard")
        out = query.head(2).copy()
        out["air_velocity_m_s"], out["power_w"] = 0.2, 600.0
        bad = pp.validate(model, query=out.to_dict("list"), level="standard")
    return conclude("07_mlp_zoo_heatsink_convection", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(run().good)
