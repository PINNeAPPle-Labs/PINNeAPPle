"""07 — Solver data, a surrogate on top of it, and the two drawn in 3-D: the airliner and the pole impact.

Both cases follow the same shape: the reference is a solver the library already ships, the surrogate is fitted to the
fields that solver produced, and the comparison is drawn on the geometry — reference, surrogate and the difference on
one colour scale — so "the surrogate agrees with the solver" is something you look at instead of something you take on
faith.

===============================  ======================================  ==================================
case                             reference                               surrogate
===============================  ======================================  ==================================
airliner (a320 from the library)  vortex lattice, 40 strips                ridge over the incidences
pole impact (bumper beam)         explicit crash solver, 30 designs        the three nearest designs
===============================  ======================================  ==================================

    python examples/surrogate_pipeline/07_aircraft_and_crash.py                 # both cases (~4 min)
    python examples/surrogate_pipeline/07_aircraft_and_crash.py --only aircraft
    python examples/surrogate_pipeline/07_aircraft_and_crash.py --out runs/demo

The 30-design crash campaign is the slow part; ``--reuse`` reads ``<out>/crash/pole_impact_fields.npz`` when it is
already there instead of solving again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pinneapple as pp  # noqa: E402
from pinneapple_analysis import surrogate  # noqa: E402
from pinneapple_design.aero.airliner import Airliner  # noqa: E402
from pinneapple_design.aero.vlm import solve  # noqa: E402
from pinneapple_simulation.numerical_solvers.crash_cases import DESIGN_BOUNDS  # noqa: E402

OUT = REPO / "examples" / "surrogate_pipeline" / "_out" / "cases"
BOLD, RESET = ("\033[1m", "\033[0m") if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else ("", "")


def rule(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n{'─' * len(title)}")


# ------------------------------------------------------------------------------------------------ the airliner
def aircraft(out: Path) -> dict:
    """The library's a320 through its vortex lattice, and a surrogate that learned the lift distribution."""
    g = pp.geometries.build("airliner_a320")
    p = g.params
    ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
                  sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
                  engine_station=p["engine_station"], sharklets=p["sharklets"])
    sol = solve(ac, ac.mac_le_x + 0.25 * ac.mac)
    alphas = np.deg2rad(np.linspace(-2.0, 8.0, 13))
    reference = np.array([sol.strip_cl(a, 0.0) for a in alphas])            # (13, strips)
    strips = reference.shape[1]
    print(f"vortex lattice: CL_alpha {sol.CL[1]:.2f} /rad, {strips} strips, "
          f"cl {reference.min():.3f} .. {reference.max():.3f} over {np.rad2deg(alphas[0]):.0f}..{np.rad2deg(alphas[-1]):.0f} deg")

    data = surrogate.from_arrays({"X": alphas[:, None], "Y": reference}, input_names=["alpha"],
                                 output_names=[f"cl{i}" for i in range(strips)])
    study = surrogate.run_study(data, methods=["ridge", "gp"], budget=4.0, seeds=1,
                                splits=surrogate.make_splits(data, test_frac=0.24, val_frac=0.24, shift="edge"))
    row = study.ranking()[0]
    test = np.deg2rad(3.5)
    truth = sol.strip_cl(test, 0.0)
    guess = np.asarray(study.model.predict_array(np.array([[test]]))).reshape(-1)
    l2 = float(np.linalg.norm(guess - truth) / np.linalg.norm(truth))
    print(f"surrogate: {row['method']}, held-out nrmse {row['error']:.2e}; at alpha {np.rad2deg(test):.1f} deg the "
          f"cl distribution is off by relative L2 {l2:.2e} (worst strip {np.max(np.abs(guess - truth)):.3f})")

    out.mkdir(parents=True, exist_ok=True)
    figs = [draw_aircraft(g, sol, truth, "reference · vortex lattice", out / "a320_reference"),
            draw_aircraft(g, sol, guess, "surrogate · ridge over alpha", out / "a320_surrogate")]
    return {"cl_alpha": float(sol.CL[1]), "strips": strips, "method": row["method"],
            "held_out_nrmse": float(row["error"]), "relative_l2": l2, "figures": figs}


def draw_aircraft(g, sol, cl, title: str, stem: Path):
    """The aircraft in the library's three.js viewer with the local cl painted on the wings and the tails."""
    from pinneapple_tools.visualization.studio.scene import Scene

    surfaces = g.surfaces()
    strips = {"wing": (np.abs(sol.L.yc), sol.L.strip_surf == 0), "htail": (np.abs(sol.L.yc), sol.L.strip_surf == 1)}
    for s in surfaces:
        name = (s.name or "").lower()
        kind = "wing" if "wing" in name else "htail" if "htail" in name else ""
        if not kind:
            s.fields = {}
            continue
        ys, mask = strips[kind]
        ys, cls = np.asarray(ys)[mask], np.asarray(cl)[mask]
        if len(ys) == 0:
            s.fields = {}
            continue
        order = np.argsort(ys)
        s.fields = {"cl": np.interp(np.abs(s.vertices[:, 1]), ys[order], cls[order])}
    page = pp.viz.web_viewer(Scene(surfaces, axes="z_up", title=title), str(stem), single_file=True)
    return page


# --------------------------------------------------------------------------------------------- the pole impact
def crash(out: Path, *, designs: int, reuse: bool) -> dict:
    """The pole impact through the library's explicit solver, and a surrogate over the campaign."""
    folder = out / "crash"
    npz = folder / "pole_impact_fields.npz"
    samples = _load_campaign(npz) if (reuse and npz.is_file()) else _solve_campaign(folder, designs)
    truth, guess, l2 = _field_surrogate(samples)
    print(f"{len(samples)} pole impacts, {samples[0]['points'].shape[0]} nodes: the surrogate (3 nearest designs) "
          f"reproduces the |u| field of a held-out design with relative L2 {l2:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    figs = _draw_crash(samples, out)
    return {"designs": len(samples), "relative_l2": float(l2), "figures": figs}


def _solve_campaign(folder: Path, designs: int) -> list[dict]:
    """``designs`` pole impacts of the coupon, every design sharing the mesh — the campaign the surrogate learns."""
    from pinneapple_simulation.numerical_solvers.crash_cases import pole_impact, sample_design

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"solving {designs} pole impacts (explicit FE, a few seconds each) ...")
    samples, est = [], {}
    for i, design in enumerate(sample_design(rng, n=designs)):
        started = time.perf_counter()
        res = pole_impact(design, end_time=6e-4, frames=9)
        samples.append({"id": f"s{i}", "points": np.asarray(res.mesh.nodes, float),
                        "values": np.asarray(res.displacement, float),
                        "design": np.array([[design[k] for k in DESIGN_BOUNDS]], float)})
        if not est:
            est["each"] = time.perf_counter() - started
            print(f"  one design takes {est['each']:.1f} s of solver time; {designs} of them is about "
                  f"{est['each'] * designs / 60:.1f} min")
    saved = {}
    for s in samples:
        saved[f"{s['id']}_points"] = s["points"]
        saved[f"{s['id']}_values"] = s["values"]
        saved[f"{s['id']}_extra"] = s["design"]
    np.savez_compressed(folder / "pole_impact_fields.npz", **saved)
    print(f"  campaign written to {folder / 'pole_impact_fields.npz'}")
    return samples


def _load_campaign(npz: Path) -> list[dict]:
    data = np.load(npz, allow_pickle=True)
    ids = sorted({k.split("_")[0] for k in data.files if k.startswith("s") and k[1:].split("_")[0].isdigit()},
                 key=lambda s: int(s[1:]))
    return [{"id": key, "points": np.asarray(data[f"{key}_points"], float),
             "values": np.asarray(data[f"{key}_values"], float),
             "design": np.asarray(data[f"{key}_extra"], float)} for key in ids]


def _field_surrogate(samples):
    x = np.array([np.asarray(s["design"]).ravel() for s in samples], float)
    lo, hi = x.min(0), x.max(0)
    z = (x - lo) / np.where(hi > lo, hi - lo, 1.0)
    held, train = samples[-1], samples[:-1]
    d = np.linalg.norm(z[:-1] - z[-1], axis=1)
    near = np.argsort(d)[:3]
    w = 1.0 / np.maximum(d[near], 1e-6)
    fields = np.array([np.linalg.norm(train[i]["values"][-1], axis=1) for i in near])
    guess = (fields * w[:, None]).sum(0) / w.sum()
    truth = np.linalg.norm(held["values"][-1], axis=1)
    return truth, guess, float(np.linalg.norm(guess - truth) / np.linalg.norm(truth))


def _draw_crash(samples, out: Path):
    """The coupon's own mesh, warped by the solver and by the surrogate, on one colour scale."""
    import pyvista as pv

    from pinneapple_simulation.numerical_solvers.crash_cases import pole_impact

    held = samples[-1]
    design = {k: float(v) for k, v in zip(DESIGN_BOUNDS, np.asarray(held["design"]).ravel(), strict=False)}
    res = pole_impact(design, end_time=6e-4, frames=9)
    nodes = np.asarray(res.mesh.nodes, float) * 1e3
    elements = np.asarray(res.mesh.elements, int)
    moved = nodes + np.asarray(held["values"][-1], float) * 1e3
    truth, guess, _ = _field_surrogate(samples)
    vmax = float(max(truth.max(), guess.max()) * 1e3)

    def grid(values):
        g = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: elements}, moved)
        g.point_data["|u| [mm]"] = np.asarray(values, float) * 1e3
        return g

    pv.OFF_SCREEN = True
    paths = []
    for values, title, name in ((truth, f"explicit crash solver · design {held['id']}", "crash_reference"),
                                (guess, "surrogate · three nearest designs", "crash_surrogate")):
        pl = pv.Plotter(off_screen=True, window_size=(1100, 900))
        pl.add_mesh(grid(values), scalars="|u| [mm]", cmap="jet", clim=(0.0, vmax), smooth_shading=True,
                    show_scalar_bar=True, scalar_bar_args={"title": "|u| [mm]", "color": "white", "vertical": True,
                                                           "position_x": 0.88, "position_y": 0.12, "width": 0.05,
                                                           "height": 0.7}, ambient=0.4, diffuse=0.75)
        pl.add_text(title, font_size=12, color="white", position="upper_left")
        pl.set_background("#0e1117")
        c = np.asarray([(b + a) / 2 for a, b in zip(*[iter(pl.bounds)] * 2, strict=False)], float)
        span = float(np.linalg.norm(np.asarray(pl.bounds[1::2]) - np.asarray(pl.bounds[0::2])))
        pl.camera.focal_point = tuple(c)
        pl.camera.position = tuple(c + np.array([0.95, 1.25, 0.55]) * span)
        pl.camera.up = (0.0, 0.0, 1.0)
        path = out / f"{name}.png"
        pl.screenshot(str(path))
        pl.close()
        paths.append(str(path))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="solver data, a surrogate on top, and both drawn in 3-D")
    ap.add_argument("--only", default="all", choices=["all", "aircraft", "crash"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--designs", type=int, default=30)
    ap.add_argument("--reuse", action="store_true", help="reuse the crash campaign already in --out")
    a = ap.parse_args()
    out = Path(a.out)
    started = time.perf_counter()
    summary = {}
    if a.only in ("all", "aircraft"):
        rule("airliner_a320 · vortex lattice → surrogate")
        summary["aircraft"] = aircraft(out / "aircraft")
    if a.only in ("all", "crash"):
        rule("pole impact · explicit crash solver → surrogate")
        summary["crash"] = crash(out, designs=a.designs, reuse=a.reuse)
    (out / "cases.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\ndone in {time.perf_counter() - started:.1f} s; figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
