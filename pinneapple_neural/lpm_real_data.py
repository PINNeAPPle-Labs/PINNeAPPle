"""A design space made of published, verified simulation data: the pipeline trained on someone else's runs.

The five families in :mod:`pinneapple_neural.lpm_families` generate their own targets from a reference this
repository implements. This module is the other half of the argument: it takes a **published campaign** — real
solver runs, checked by its authors against grid convergence, an independent code and a measured benchmark — and
fits the same model on it. If the methodology only works on fields this repository produced, it has not been shown
to work; here the data existed before the model did, the split was drawn by its authors, and the only thing the
pipeline may do is read it.

``pp_airfoil_rans``
    the airfoil campaign of ``pinneapple_hub.zoo``: 436 steady 2-D RANS runs (OpenFOAM ``simpleFoam``, k-omega SST,
    y+ < 1) over 100 Latin-hypercube CST shapes at four incidences plus four NACA sections at nine incidences, on a
    shared 228 x 88 O-grid. The geometry the model sees is the **wall ring** of that grid, the field is the surface
    Cp from the solved kinematic pressure, and the condition is the incidence.

Three gates stand between the dataset and the fit, and a run that fails one is not written:

* the run's own status — the campaign flags 32 runs unsteady, and a final-iteration snapshot of an oscillating
  solution is not a steady answer;
* the wall extraction must **reproduce the campaign's published Cl** from the pressure it carries, which is the only
  check that the ring, the normals and the sign convention were read correctly;
* the surface Cp must stay inside the range a RANS airfoil can produce, so a mis-indexed grid is caught instead of
  trained on.

What this does **not** do: it does not validate the RANS itself (that is the campaign's verification, quoted in the
reference), and it does not turn a 2-D section into a 3-D wing. It shows the pipeline on data nobody here produced.
"""
from __future__ import annotations

import numpy as np

from .lpm_pipeline import DesignSpec, FieldFamily, Reference

__all__ = ["RANS_DATASET", "RansAirfoil", "data_gates"]

RANS_DATASET = "airfoil_rans_cst_re4e6_fields"
#: cells around one wall-normal level of the O-grid (the grid is 228 x 88, level-major)
N_AROUND = 228
#: chord and free stream of the campaign: chord 1, U = 1, so Cp = 2 p for the kinematic pressure it stores
CHORD = 1.0
U_INF = 1.0
#: the whole grid moves with the shape, so a new geometry is new coordinates, not a new mesh
STATUS_OK = ("converged", "steady-ish")
#: the lift gate: the pressure integral against the campaign's coefficient, relative with an absolute floor
CL_TOLERANCE = 0.10
CL_FLOOR = 0.03

_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load() -> dict[str, dict[str, np.ndarray]]:
    """The campaign's train and test splits, read once per process through the public dataset loader."""
    if not _CACHE:
        import pinneapple as pp

        for split in ("train", "test"):
            data = pp.datasets.load(RANS_DATASET, split=split, offline=True)
            _CACHE[split] = {name: np.asarray(data[name]) for name in data.columns}
    return _CACHE


def _run(spec: DesignSpec) -> dict[str, np.ndarray]:
    """The stored run a spec names: the split it belongs to and its row there."""
    data = _load()[spec.split or "train"]
    index = int(spec.geometry["row"])
    return {name: values[index] for name, values in data.items()}


def _wall(spec: DesignSpec) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """The wall ring of the run's O-grid, its outward normals and the run record it came from."""
    run = _run(spec)
    x, y = run["x"][:N_AROUND], run["y"][:N_AROUND]
    points = np.stack([x, y], axis=1)
    # the ring is closed, so the tangent at each point comes from its neighbours and the normal from the tangent;
    # the sign is fixed by the enclosed area, so the normal points out of the section whatever way the ring is wound
    following = np.roll(points, -1, axis=0)
    tangent = following - points
    normals = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    area = 0.5 * np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    if area < 0.0:
        normals = -normals
    return points, normals, run


def _pressure_cl(points: np.ndarray, normals: np.ndarray, cp: np.ndarray) -> float:
    """Lift coefficient of a 2-D section from its surface Cp, per unit chord.

    ``Cp`` is already normalised by the dynamic pressure, so the integral of ``Cp`` against the outward normal is
    the coefficient itself: dividing again by ``U^2/2`` doubles it, which is exactly the mistake the lift gate below
    is there to catch.
    """
    following = np.roll(points, -1, axis=0)
    weight = np.linalg.norm(following - points, axis=1)
    return float(-np.sum(cp * normals[:, 1] * weight))


class RansAirfoil(FieldFamily):
    """Surface pressure over a design space of CST airfoils, solved by the published OpenFOAM campaign.

    The point cloud is the wall ring of the shared O-grid (228 points), the field is Cp from the run's kinematic
    pressure, and the single condition is the incidence. Nothing is drawn here: :meth:`specs` returns the campaign's
    runs exactly as they were recorded, and :meth:`published_split` hands back the 15 % of shapes its authors held
    out before publishing, so a held-out number means the same thing here as it does in the campaign.
    """

    name = "pp_airfoil_rans"
    title = "CST airfoils at Re 4e6: surface Cp over a published OpenFOAM RANS campaign"
    domain = "aerodynamics / real RANS data at Re 4e6"
    dim = 2
    field_names = ("cp",)
    field_units = ("-",)
    field_scale = (1.0,)
    characteristic_length_m = CHORD
    n_points = N_AROUND
    tolerance = 0.10
    #: the campaign's own ranges; nothing is drawn, they are here so a prediction can be normalised and read back
    operating_params = {"alpha_deg": (-2.0, 14.0)}
    geometry_params: dict[str, tuple[float, float]] = {}
    reference = Reference(
        name="steady 2-D RANS, OpenFOAM simpleFoam with k-omega SST and y+ < 1",
        citation=("PINNeAPPle airfoil campaign, dataset airfoil_rans_cst_re4e6_fields (cc-by-4.0); 436 runs on a "
                  "20064-cell O-grid, 100 Latin-hypercube CST shapes x 4 incidences and NACA 0012/2412/4412/4415 x "
                  "9 incidences, free stream U = 1 and chord 1 so Cp = 2 p"),
        assumptions=("steady RANS with a fully turbulent k-omega SST closure: no transition, no large separation "
                     "resolved in time", "2-D section at Re 4e6 and incompressible M = 0",
                     "the campaign's verification: grid study on NACA 2412 (Cl 0.6614/0.6625/0.6624 at "
                     "6840/20064/45144 cells, Cd GCI 1.9 %), XFOIL 6.99 (Cl -2.3 %, Cd +4.5 %), lift slope "
                     "0.110/deg against thin-airfoil theory, NASA TMR NACA 0012 SST (Cd 0.00812 vs 0.0081)"),
        implements="solver",
        provenance=f"pinneapple hub dataset {RANS_DATASET}, read through pinneapple.datasets.load(offline=True)",
        licence="cc-by-4.0",
    )

    # -- the campaign as a design space --------------------------------------------------------------------------
    def specs(self) -> list[DesignSpec]:
        """One spec per stored run that passed the status gate, with the campaign's own split marker."""
        data = _load()
        shapes = sorted({str(s) for split in data.values() for s in split["shape_id"]})
        shape_index = {shape: i for i, shape in enumerate(shapes)}
        specs: list[DesignSpec] = []
        for split in ("train", "test"):
            rows = data[split]
            for row in range(len(rows["run_id"])):
                if str(rows["status"][row]) not in STATUS_OK:
                    continue
                specs.append(DesignSpec(
                    geometry={"row": float(row), "shape_id": float(shape_index[str(rows["shape_id"][row])])},
                    operating={"alpha_deg": float(rows["alpha_deg"][row])},
                    geometry_id=shape_index[str(rows["shape_id"][row])],
                    label=f"{self.name}:{str(rows['run_id'][row])}",
                    split=split,
                ))
        return specs

    # -- the geometry and the field, straight from the solver ----------------------------------------------------
    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        points, normals, _ = _wall(spec)
        return points, normals

    def field(self, spec: DesignSpec) -> np.ndarray:
        _, _, run = _wall(spec)
        return (2.0 * run["p"][:N_AROUND])[:, None]

    # -- the gates -----------------------------------------------------------------------------------------------
    def validate(self, spec: DesignSpec) -> str:
        """The run must be steady, must carry a finite wall pressure, and must reproduce its own lift."""
        points, normals, run = _wall(spec)
        if str(run["status"]) not in STATUS_OK:
            return f"the campaign flags this run {run['status']!r}, not steady"
        cp = 2.0 * run["p"][:N_AROUND]
        if not np.all(np.isfinite(cp)):
            return "the stored wall pressure has non-finite values"
        if cp.min() < -30.0 or cp.max() > 1.5:
            return f"wall Cp spans [{cp.min():.2f}, {cp.max():.2f}], outside what a RANS section can produce"
        reported = float(run["Cl"])
        integrated = _pressure_cl(points, normals, cp)
        # a symmetric section at zero incidence carries almost no lift, and a relative tolerance on a coefficient of
        # 0.01 is a relative tolerance on the viscous residual: the gate needs an absolute floor as well
        if abs(integrated - reported) > CL_TOLERANCE * abs(reported) + CL_FLOOR:
            return (f"the wall pressure integrates to Cl {integrated:.4f} against the campaign's {reported:.4f}; "
                    f"the ring, the normals or the sign was read wrong")
        return ""

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A NACA reference run (never in training) at an incidence outside the trained band."""
        data = _load().get("test")
        if data is None:
            return []
        probes = []
        for row in range(len(data["run_id"])):
            if str(data["status"][row]) in STATUS_OK and abs(float(data["alpha_deg"][row])) >= 12.0:
                probes.append((f"campaign run {str(data['run_id'][row])} at alpha {float(data['alpha_deg'][row]):.1f} deg",
                               DesignSpec({"row": float(row), "shape_id": -1.0},
                                          {"alpha_deg": float(data["alpha_deg"][row])},
                                          geometry_id=-1, label=f"{self.name}:probe", split="test")))
                break
        return probes


def data_gates(family: RansAirfoil | None = None, max_runs: int | None = None) -> dict[str, object]:
    """Run every gate over the campaign and report what passed, what was refused and why.

    This is the report a dataset has to carry before a model is fitted on it: how many stored runs there are, how
    many the status gate left, how many reproduce their own lift coefficient from the pressure, and how the two
    splits are made up.
    """
    family = family or RansAirfoil()
    data = _load()
    total = {split: len(rows["run_id"]) for split, rows in data.items()}
    steady = {split: int(sum(str(s) in STATUS_OK for s in rows["status"])) for split, rows in data.items()}
    specs = family.specs()
    # the split panel describes the campaign, so it is counted before any sampling: a truncated gate run that only
    # reached the train half must not report a campaign with no test runs
    splits = {"train": sum(1 for s in specs if s.split == "train"),
              "test": sum(1 for s in specs if s.split == "test")}
    if max_runs is not None:
        specs = specs[:max_runs]
    checked = len(specs)
    failures: dict[str, int] = {}
    lift_errors: list[float] = []
    lift_absolute: list[float] = []
    for spec in specs:
        reason = family.validate(spec)
        if reason:
            key = reason.split(",")[0][:60]
            failures[key] = failures.get(key, 0) + 1
            continue
        points, normals, run = _wall(spec)
        cp = 2.0 * run["p"][:N_AROUND]
        integrated = _pressure_cl(points, normals, cp)
        lift_absolute.append(abs(integrated - float(run["Cl"])))
        if abs(float(run["Cl"])) > 1e-6:
            lift_errors.append(abs(integrated / float(run["Cl"]) - 1.0))
    return {
        "dataset": RANS_DATASET,
        "licence": family.reference.licence,
        "stored_runs": total,
        "steady_runs": steady,
        "designs_checked": checked,
        "designs_after_gates": checked - sum(failures.values()),
        "designs_refused": sum(failures.values()),
        "refusals": failures,
        "lift_gate": {"max_relative_error": max(lift_errors) if lift_errors else None,
                      "median_relative_error": float(np.median(lift_errors)) if lift_errors else None,
                      "max_absolute_error": max(lift_absolute) if lift_absolute else None,
                      "median_absolute_error": float(np.median(lift_absolute)) if lift_absolute else None,
                      "tolerance": f"{CL_TOLERANCE:.0%} of the reported Cl plus {CL_FLOOR}",
                      "note": "the wall pressure integrates to the campaign's Cl; the residual is the viscous part "
                              "of the reported coefficient, which a pressure integral does not contain"},
        "shapes": len({spec.geometry_id for spec in family.specs()}),
        "splits": splits,
    }
