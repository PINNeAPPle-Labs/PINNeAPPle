"""Where the heat-sink surrogate fails, and what it is worth: a quality-diversity archive of a thermal model (#354).

A plate-fin heat sink is described by three numbers in this study -- fin count, fin height and approach air speed --
and the house closed-form model (:mod:`pinneapple_physics.closed_form.plate_fin_heatsink`, Teertstra forced convection
inside a duct, Bar-Cohen & Rohsenow natural convection in still air, Lee spreading resistance) says what each one does:
thermal resistance ``Rth``, fan power and mass. A surrogate is the usual way to make that model cheap enough to sit
inside a design loop, and it is trained on a *campaign* -- here, ducted forced convection, 10-16 fins, 15-35 mm fins,
1.0-2.5 m/s, which is what a fan-cooled extrusion project would sample.

The question this script answers is the one a surrogate cannot answer about itself: **where is it wrong?** The search
is a MAP-Elites archive over a behaviour space that is not the design space -- ``Rth`` against mass, i.e. how well the
part cools against how much metal it costs -- and the fitness is the surrogate's own relative error against the
closed-form model. The cells that come out above 10 % are the failure region, each one reported as a condition
("Rth in [6.0, 8.0] K/W, mass in [0.06, 0.08] kg") with the design that produced it.

Why those axes. Resistance and mass are the two things the customer pays for: watts of cooling and grams of
aluminium. Every design a thermal engineer weighs is a point on that plane, the same cell is reached by many different
fin counts, and the plane is where "this part is a different kind of part" has a meaning. The design parameters
(n, H, V) are on the other side of the map: they are what you *change*, not what the part *is*. Both axes come from
the reference model, never from the surrogate, so a cell says something true about the part even when the surrogate is
wrong about it.

The same archive with ``sense="min"`` is the design map: the least fan power that achieves each (Rth, mass) cell.

What is written
---------------
``figures/heat_sink_failure_map.png``   the archive, coloured by the surrogate's worst Rth error per cell, the failing
                                        cells marked;
``figures/heat_sink_design_map.png``    the same plane with the least fan power found in each cell;
``heat_sink_elites.csv``                one row per elite of both maps: the cell, the behaviour condition, the value
                                        and the design that produced it -- small enough to diff between runs;
``heat_sink_qd.json``                   the comparison against the baselines, the known-failure check and the budget,
                                        seed and tolerance every number here was produced with.

Run it with ``python examples/quality_diversity/heat_sink_failure_map.py`` (about a minute on a laptop CPU: most of it
is fitting the surrogate). Nothing large is written: the elites are a few dozen rows of text.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pinneapple_design.design_optimizer.quality_diversity import (
    BehaviourAxis,
    BehaviourSpace,
    EliteArchive,
    Evaluation,
    QDConfig,
    QualityDiversitySearch,
    find_failure_region,
    plot_archive,
    uniform_search,
)
from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate
from pinneapple_physics.closed_form.plate_fin_heatsink import HeatSinkGeometry, OperatingPoint
from pinneapple_physics.closed_form.plate_fin_heatsink import evaluate as sink_evaluate

# ── the part, fixed around the three design numbers ──────────────────────────────────────────────────────────────
BASE_WIDTH_M = 0.050
BASE_DEPTH_M = 0.050
BASE_THICKNESS_M = 0.004
FIN_THICKNESS_M = 0.001
MATERIAL = "al6063"
POWER_W = 25.0
T_AMBIENT_C = 25.0
SOURCE_M = 0.020

# ── the campaign the surrogate was fitted on, and the envelope a designer asks about afterwards ──────────────────
# The campaign covered the geometry the extruder offers and the air speeds the ducted rig can hold. It never went
# below 0.8 m/s: no fan curve in the project reached there, and still air was out of scope. The design questions that
# arrive afterwards do go there -- a part has to survive a fan failure, and half the catalogue is passive.
TRAIN_BOX = {"n_fins": (4.0, 20.0), "fin_height_mm": (8.0, 50.0), "velocity": (0.8, 3.0)}
SEARCH_BOX = {"n_fins": (4.0, 20.0), "fin_height_mm": (8.0, 50.0), "velocity": (0.0, 3.0)}
PARAMS = tuple(SEARCH_BOX)

T_MAX_C = 85.0                      # junction limit of the part the sink cools
RTH_MAX_K_W = (T_MAX_C - T_AMBIENT_C) / POWER_W     # 2.4 K/W: above it the part is not a usable operating point

# ── the behaviour space: cooling against metal, both from the closed-form model ─────────────────────────────────
BEHAVIOUR = BehaviourSpace(
    axes=(BehaviourAxis("Rth", 0.4, RTH_MAX_K_W, unit="K/W",
                        why="thermal resistance is the cooling the customer buys; the top of the range is where the "
                            "junction reaches its limit, above which the part is not a part"),
          BehaviourAxis("mass", 0.04, 0.17, unit="kg",
                        why="the aluminium is the other half of the cost, and it is what the fin count trades against")),
    bins=(10, 10), labels="plate-fin heat sink, 50 x 50 mm base, 25 W, aluminium 6063")

TOLERANCE = 0.10
TRAIN_SAMPLES = 2000
TRAIN_EPOCHS = 300
FAN_POWER_FLOOR_W = 1.0e-4


# ── the reference model and the case that makes a design legitimate ──────────────────────────────────────────────


def reference(row) -> dict:
    """Closed-form evaluation of one design row ``(n_fins, fin_height_mm, velocity)``.

    ``valid`` is decided by the *reference*, never by the surrogate: the case has to sit inside the ranges of the
    correlations that produced it -- the Teertstra channel Reynolds for ducted flow, the Elenbaas number for still air,
    the laminar limit on the channel Reynolds -- and the part has to be usable, i.e. the junction has to stay below
    ``T_MAX_C``. A case the physics cannot answer, or that no one would build, is not a failure of the model, and the
    archive refuses it.
    """
    n = int(round(float(row[0])))
    g = HeatSinkGeometry(base_width=BASE_WIDTH_M, base_depth=BASE_DEPTH_M, base_thickness=BASE_THICKNESS_M,
                         n_fins=max(n, 2), fin_thickness=FIN_THICKNESS_M, fin_height=float(row[1]) * 1e-3,
                         material=MATERIAL)
    op = OperatingPoint(power_w=POWER_W, t_ambient_c=T_AMBIENT_C, air_velocity_m_s=float(row[2]),
                        source_width=SOURCE_M, source_depth=SOURCE_M)
    res = sink_evaluate(g, op)
    d = res["dimensionless"]
    if res["mode"] == "natural":
        in_range = 0.1 <= d["elenbaas"] <= 1e4
    else:
        in_range = 0.26 <= d["re_star"] <= 175 and d["re_dh"] <= 2300
    rth = float(res["r_total_k_w"])
    return {"rth": rth, "fan_w": float(res["fan_power_w"]), "mass": float(res["mass_kg"]),
            "mode": res["mode"], "valid": bool(in_range and rth <= RTH_MAX_K_W)}


# ── the surrogate: an MLP of the three design numbers, fitted on the campaign ───────────────────────────────────


class ResponseSurface:
    """The surrogate under test: normalised inputs, standardised outputs, physical units at the boundary."""

    def __init__(self, core, x_lo, x_hi, y_mean, y_std) -> None:
        self.core, self.x_lo, self.x_hi, self.y_mean, self.y_std = core, x_lo, x_hi, y_mean, y_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Closed-loop prediction of ``(Rth, fan power, mass)`` for design rows."""
        Z = (np.atleast_2d(np.asarray(X, dtype=float)) - self.x_lo) / (self.x_hi - self.x_lo)
        with torch.no_grad():
            y = self.core.model(torch.tensor(Z, dtype=torch.float32)).numpy()
        return y * self.y_std + self.y_mean


def fit_surrogate(seed: int, n: int = TRAIN_SAMPLES, epochs: int = TRAIN_EPOCHS) -> ResponseSurface:
    """Fit the surrogate on the campaign box. The seed fixes the samples *and* the initial weights, so it is the run."""
    rng = np.random.default_rng(seed)
    lo = np.array([TRAIN_BOX[k][0] for k in PARAMS])
    hi = np.array([TRAIN_BOX[k][1] for k in PARAMS])
    X = lo + rng.random((n, len(PARAMS))) * (hi - lo)
    refs = [reference(x) for x in X]
    Y = np.array([[r["rth"], np.log10(max(r["fan_w"], FAN_POWER_FLOOR_W)), r["mass"]] for r in refs])
    # the network sees the whole envelope, not just the campaign, so raw inputs are scaled to it: an input scaling is
    # not information about the model's validity, and keeping it fixed keeps the surrogate's extrapolation honest
    x_lo = np.array([SEARCH_BOX[k][0] for k in PARAMS])
    x_hi = np.array([SEARCH_BOX[k][1] for k in PARAMS])
    Z = (X - x_lo) / (x_hi - x_lo)
    mean, std = Y.mean(0), Y.std(0)
    torch.manual_seed(seed)
    core = PhysicsSurrogate.build_mlp(in_dim=len(PARAMS), out_dim=Y.shape[1], hidden=64, layers=3)
    core.train_on_dataset({"theta": torch.tensor(Z, dtype=torch.float32),
                           "u": torch.tensor((Y - mean) / std, dtype=torch.float32)},
                          epochs=epochs, lr=1e-3, verbose=False)
    return ResponseSurface(core, x_lo, x_hi, mean, std)


# ── the two things the archive can be asked ─────────────────────────────────────────────────────────────────────


def failure_evaluation(surrogate: ResponseSurface, row: np.ndarray) -> Evaluation:
    """The surrogate's relative error on thermal resistance, with the reference behaviour as the descriptors."""
    ref = reference(row)
    if not ref["valid"]:
        return Evaluation(value=float("nan"), descriptors=None, valid=False, note=f"reference not applicable "
                                                                                 f"({ref['mode']})")
    pred = surrogate.predict(row)[0]
    rel_rth = abs(float(pred[0]) - ref["rth"]) / ref["rth"]
    pred_fan = 10.0 ** float(pred[1])
    rel_fan = abs(pred_fan - ref["fan_w"]) / max(ref["fan_w"], FAN_POWER_FLOOR_W)
    rel_mass = abs(float(pred[2]) - ref["mass"]) / ref["mass"]
    return Evaluation(value=rel_rth, descriptors=(ref["rth"], ref["mass"]),
                      error={"rth": rel_rth, "fan_power": rel_fan, "mass": rel_mass},
                      note=f"{ref['mode']}, Rth {ref['rth']:.3f} K/W, fan {ref['fan_w']:.4g} W")


def fan_power_evaluation(row: np.ndarray) -> Evaluation:
    """The design map's objective: the fan power that achieves this cell, to be minimised."""
    ref = reference(row)
    if not ref["valid"]:
        return Evaluation(value=float("nan"), descriptors=None, valid=False, note=f"reference not applicable "
                                                                                 f"({ref['mode']})")
    return Evaluation(value=ref["fan_w"], descriptors=(ref["rth"], ref["mass"]),
                      error={"rth": ref["rth"], "mass": ref["mass"]},
                      note=f"{ref['mode']}, n_fins {int(round(float(row[0])))}")


# ── the campaign ────────────────────────────────────────────────────────────────────────────────────────────────


def campaign(budget: int = 600, seeds: tuple[int, ...] = (0, 1, 2, 3, 4), train_seed: int = 0) -> dict:
    """One surrogate, then the failure search and the two baselines at the same budget, over the seeds given."""
    surrogate = fit_surrogate(train_seed)
    evaluate = lambda x: failure_evaluation(surrogate, x)          # noqa: E731
    runs = {"qd": [], "random": [], "latin": []}
    regions, searches = {}, None
    for seed in seeds:
        region, result = find_failure_region(SEARCH_BOX, evaluate, BEHAVIOUR, budget=budget, seed=seed,
                                             tolerance=TOLERANCE, name="surrogate Rth error",
                                             cfg=QDConfig(budget=budget, seed=seed))
        regions[seed] = region
        searches = result
        runs["qd"].append({"seed": seed, "max_error": float(result.best_seen.value), "map_max": region.map_max,
                           "cells": region.archive.cells,
                           "failing_cells": len(region.cells()), "fill": region.archive.fill,
                           "invalid": region.archive.n_invalid,
                           "median_cell_error": float(np.median([e.value for e in region.archive.elites])),
                           "spread": region.archive.diversity().design_spread,
                           "elapsed_s": result.elapsed_s})
        for kind in ("random", "latin"):
            base = uniform_search(SEARCH_BOX, evaluate, BEHAVIOUR, budget=budget, seed=seed, kind=kind, sense="max")
            runs[kind].append({"seed": seed, "max_error": float(base.best.value), "cells": base.archive.cells,
                               "failing_cells": len([e for e in base.archive.elites if e.value > TOLERANCE]),
                               "fill": base.archive.fill, "invalid": base.archive.n_invalid,
                               "median_cell_error": float(np.median([e.value for e in base.archive.elites])),
                               "spread": base.archive.diversity().design_spread, "elapsed_s": base.elapsed_s})
    # the design map: the same archive machinery with sense="min", objective = the fan power the design costs
    design_archive = EliteArchive(BEHAVIOUR, sense="min", name="fan power [W]", design_names=PARAMS)
    design_result = QualityDiversitySearch(design_archive, SEARCH_BOX, fan_power_evaluation,
                                           QDConfig(budget=budget, seed=int(seeds[0]))).run()
    return {"surrogate": surrogate, "runs": runs, "regions": regions, "design": (design_archive, design_result),
            "search": searches}


def known_failure_check(region, surrogate: ResponseSurface) -> dict:
    """The failures this study already knows about, asked of the archive without being told about them.

    ``near pass`` is the one that matters: 0.6 m/s of approach air is *not* a strange operating point -- it is the
    middle of the catalogue -- and it is below the campaign's floor of 0.8 m/s, so the surrogate has never seen it.
    ``fan off`` exercises the other half of the rule: the physics can answer it (natural convection) but the part is
    not usable there, so the archive has to refuse it instead of reporting a 90 % error that means nothing.
    """
    checks = []
    for label, row in (("near pass", np.array([14.0, 30.0, 0.6])), ("fan off", np.array([12.0, 25.0, 0.0]))):
        evaluation = failure_evaluation(surrogate, row)
        cell = region.archive.space.cell(evaluation.descriptors) if evaluation.descriptors else None
        elite = region.archive.elite_at(cell)
        checks.append({"cursor": label, "design": dict(zip(PARAMS, [float(v) for v in row], strict=True)),
                       "valid": bool(evaluation.valid),
                       "error": None if not evaluation.valid else float(evaluation.value),
                       "descriptors": [float(v) for v in evaluation.descriptors] if evaluation.descriptors else None,
                       "cell": list(cell) if cell is not None else None,
                       "cell_in_map": elite is not None,
                       "cell_elite_error": float(elite.value) if elite is not None else None,
                       "in_failing_set": bool(cell is not None and cell in {c.cell for c in region.cells()}),
                       "condition": region.archive.space.describe(cell) if cell is not None else ""})
    return {"checks": checks}


def write_elites(path: Path, archives: dict[str, EliteArchive]) -> Path:
    """One row per elite per map: the cell, its behaviour condition, the value and the design.

    The map *is* the result, and its rows are the deliverable a designer or a trust report reads -- small enough to
    diff between runs and to paste into a ticket, which is why this is a CSV and not a dump of the archive object.
    """
    rows = []
    for label, archive in archives.items():
        for elite in archive.elites:
            row = {"map": label, "cell": "-".join(str(i) for i in elite.cell), "value": elite.value,
                   "condition": archive.space.describe(elite.cell)}
            row.update({name: float(v) for name, v in zip(PARAMS, elite.design, strict=True)})
            rows.append(row)
    fields = ["map", "cell", "value", "condition", *PARAMS]
    extra = sorted({k for row in rows for k in row if k not in fields})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(out: str | None = None, budget: int = 600, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    out_dir = Path(out) if out is not None else Path(__file__).resolve().parent
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    result = campaign(budget=budget, seeds=seeds)
    surrogate, runs = result["surrogate"], result["runs"]
    region = result["regions"][int(seeds[0])]
    design_archive, _ = result["design"]
    print(region.text(8))
    print()
    for kind in ("qd", "random", "latin"):
        rows = runs[kind]
        print(f"{kind:7s} max error {np.mean([r['max_error'] for r in rows]):.1%} "
              f"(seeds {min(r['max_error'] for r in rows):.1%}-{max(r['max_error'] for r in rows):.1%}), "
              f"failing cells {np.mean([r['failing_cells'] for r in rows]):.1f}, "
              f"cells {np.mean([r['cells'] for r in rows]):.1f}, "
              f"median cell error {np.median([r['median_cell_error'] for r in rows]):.1%}, "
              f"refused {np.mean([r['invalid'] for r in rows]):.0f}")
    known = known_failure_check(region, surrogate)
    for check in known["checks"]:
        state = ("refused (not a usable operating point)" if not check["valid"] else
                 f"{check['error']:.1%} at {check['condition']} -> "
                 f"{'IN the failing set' if check['in_failing_set'] else 'NOT in the failing set'}")
        print(f"known failure [{check['cursor']}]: {state}")
    print()
    print(design_archive.summary())
    plot_archive(region.archive, out_dir / "figures" / "heat_sink_failure_map.png", tolerance=TOLERANCE,
                 title="where the heat-sink surrogate fails", unit="relative Rth error")
    plot_archive(design_archive, out_dir / "figures" / "heat_sink_design_map.png",
                 title="least fan power achieving each (Rth, mass)", unit="W")
    write_elites(out_dir / "heat_sink_elites.csv", {"failure": region.archive, "design": design_archive})
    payload = {"budget": budget, "seeds": list(seeds), "tolerance": TOLERANCE, "train_box": TRAIN_BOX,
               "search_box": SEARCH_BOX, "train_samples": TRAIN_SAMPLES, "train_epochs": TRAIN_EPOCHS,
               "runs": runs, "known_failure": known,
               "failing_cells": len(region.cells()), "map_max": region.map_max,
               "design_cells": design_archive.cells, "design_fill": design_archive.fill}
    (out_dir / "heat_sink_qd.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="where the figures and the JSON go (default: this folder)")
    parser.add_argument("--budget", type=int, default=600, help="evaluations per method per seed")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seeds")
    args = parser.parse_args()
    main(args.out, args.budget, tuple(int(s) for s in args.seeds.split(",")))
