"""Write the example flows of Flow Studio into ``pinneapple_flow/flows/`` (the flows the GUI's "Open..." lists).

Two kinds:

* reproductions of earlier PINNeAPPle results, composed node by node (heat, Burgers, Poisson, a DeepONet / FNO
  operator, an inverse problem, the El Nino forecast, the aircraft take-off);
* new problems typed as plain descriptions and planned by the offline planner (``pinneapple_flow.plan``), saved as
  the planner proposed them (the prompt is in ``meta.prompt``), plus the 2-D -> 3-D edit of one of them.

    python examples/flow_studio/build_examples.py
    python examples/flow_studio/run_examples.py          # runs them and writes results/ (metrics + thumbnails)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pinneapple_flow as pf
from pinneapple_flow.planner.build import FlowBuilder
from pinneapple_flow.templates import template

OUT = Path(pf.__file__).parent / "flows"

PROMPTS = {
    "planned_plate_2d_to_3d": "Steady heat conduction in a square plate with internal heat generation and the "
                              "edges held at 0 degrees; compare finite elements, finite differences and a PINN. "
                              "Then switch the plate from 2D to 3D.",
    "planned_digital_twin": "Monitor a heated water tank in real time from its level and temperature sensors "
                            "(a digital twin) and detect leaks.",
    "planned_hybrid_surrogate_hpo": "Build a fast surrogate of a vibrating string for many wave speeds: a mixed "
                                    "ROM + NN model with hyperparameter tuning and a sensitivity analysis.",
    "planned_pde_twin": "A 1 m metal rod cools down by heat conduction. Build a digital twin from 5 temperature "
                        "sensors along the rod, estimate the thermal diffusivity online and detect when it changes.",
}


def _problem(fb, family, dim, *, eq_params=None, geo=None, mesh_n=None, n_t=41, prefix=""):
    t = template(family, dim)
    gp = {"dimension": dim, "x_range": t["bounds"]["x"], "y_range": t["bounds"].get("y", [0, 1]),
          "z_range": t["bounds"].get("z", [0, 1]), "time_dependent": bool(t["time"]),
          "t_range": t["time"] or [0, 1], **(geo or {})}
    g = fb.add(f"{prefix}geo", "geometry.domain", gp, label="Domain")
    m = fb.add(f"{prefix}mesh", "geometry.mesh", {"n": mesh_n or {1: 201, 2: 65, 3: 25}[dim], "n_t": n_t},
               label="Grid")
    fb.connect(g, "geometry", m, "geometry")
    ep = {"template": family, "fields": t["fields"], "pdes": t["pdes"], "conditions": t["conditions"],
          "coefficients": t["coefficients"], "exact": t["exact"] or "", **(eq_params or {})}
    e = fb.add(f"{prefix}eq", "equations.system", ep, label=f"{family} equations")
    fb.connect(g, "geometry", e, "geometry")
    fb.connect(m, "mesh", e, "mesh")
    return e


def heat_1d():
    fb = FlowBuilder("Heat 1D: PINN vs finite differences vs exact",
                     "u_t = alpha u_xx on a rod with fixed ends; the PINN and the method-of-lines solution are "
                     "measured against the exact sine mode; the Lab grades and seals the result.")
    e = _problem(fb, "heat", 1)
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Finite differences")
    pinn = fb.add("pinn", "train.pinn", {"adam_steps": 3000, "lbfgs_steps": 300}, label="PINN")
    v = fb.add("validate", "validate.compare", {"reference": "exact"}, label="Validation")
    lab = fb.add("lab", "validate.lab", {"title": "Heat 1D PINN vs FD"}, label="Lab seal")
    plot = fb.add("plot", "viz.plot", {}, label="Plots")
    for s in (fd, pinn):
        fb.connect(e, "problem", s, "problem")
        fb.connect(s, "solution", v, "candidates")
        fb.connect(s, "solution", plot, "solution")
    fb.connect(e, "problem", v, "problem")
    fb.connect(v, "metrics", lab, "validation")
    return fb.flow({"kind": "reproduction"})


def burgers_1d():
    fb = FlowBuilder("Burgers 1D: PINN vs finite differences vs Cole-Hopf",
                     "u_t + u u_x = nu u_xx with nu = 0.01/pi (Raissi et al. 2019); both solutions against the "
                     "Cole-Hopf exact solution.")
    e = _problem(fb, "burgers", 1, mesh_n=513, n_t=51)
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Finite differences")
    pinn = fb.add("pinn", "train.pinn", {"adam_steps": 8000, "lbfgs_steps": 1000, "width": 32, "depth": 5,
                                         "n_collocation": 6000, "lr": 2e-3}, label="PINN")
    v = fb.add("validate", "validate.compare", {"reference": "exact", "tolerance": 0.05}, label="Validation")
    lab = fb.add("lab", "validate.lab", {"title": "Burgers PINN vs FD"}, label="Lab seal")
    r = fb.add("render", "viz.render", {"animate": False}, label="Space-time render")
    for s in (fd, pinn):
        fb.connect(e, "problem", s, "problem")
        fb.connect(s, "solution", v, "candidates")
    fb.connect(e, "problem", v, "problem")
    fb.connect(v, "metrics", lab, "validation")
    fb.connect(fd, "solution", r, "solution")
    return fb.flow({"kind": "reproduction"})


def poisson_2d():
    fb = FlowBuilder("Poisson 2D: FEM, FD and PINN with the trust card",
                     "lap(u) = -2 pi^2 sin(pi x) sin(pi y), u = 0 on the boundary; three methods against the "
                     "manufactured solution, trust card (residual, conservation, boundary), 3-D view.")
    e = _problem(fb, "poisson", 2, mesh_n=41)
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Finite differences")
    fem = fb.add("fem", "solver.classical", {"method": "fem"}, label="Finite elements")
    pinn = fb.add("pinn", "train.pinn", {"adam_steps": 3000, "lbfgs_steps": 300}, label="PINN")
    v = fb.add("validate", "validate.compare", {"reference": "exact"}, label="Validation")
    lab = fb.add("lab", "validate.lab", {"title": "Poisson 2D"}, label="Lab seal")
    view = fb.add("view3d", "viz.field3d", {}, label="3-D viewer")
    for s in (fd, fem, pinn):
        fb.connect(e, "problem", s, "problem")
        fb.connect(s, "solution", v, "candidates")
    fb.connect(e, "problem", v, "problem")
    fb.connect(v, "metrics", lab, "validation")
    fb.connect(pinn, "solution", view, "solution")
    return fb.flow({"kind": "reproduction"})


def operator_burgers():
    fb = FlowBuilder("Neural operators: DeepONet and FNO on Burgers",
                     "Random initial data -> viscous Burgers at t = 0.5 (periodic, pseudo-spectral reference); a "
                     "DeepONet and an FNO trained on the same samples, held-out test errors.")
    d = fb.add("data", "data.grf_burgers", {"n_samples": 512, "n": 128, "nu": 0.01, "t_end": 0.5},
               label="Operator set (GRF -> Burgers)")
    don = fb.add("deeponet", "surrogate.train", {"kind": "deeponet", "epochs": 2000, "lr": 1e-3,
                                                 "architecture": {"hidden": 128, "modes": 64, "depth": 3}},
                 label="DeepONet")
    fno = fb.add("fno", "surrogate.train", {"kind": "fno", "epochs": 300, "lr": 2e-3,
                                            "architecture": {"width": 32, "modes": 16, "layers": 4}}, label="FNO")
    rep = fb.add("report", "export.report", {"title": "Operators on Burgers"}, label="Report")
    for s in (don, fno):
        fb.connect(d, "data", s, "data")
        fb.connect(s, "metrics", rep, "items")
    return fb.flow({"kind": "reproduction"})


def inverse_heat():
    fb = FlowBuilder("Inverse heat: identify the diffusivity",
                     "Synthetic sensors (400 points, 1 % noise) from a finite-difference solution with alpha = 0.1; "
                     "alpha recovered by an inverse PINN and by least-squares calibration with the FD solver.")
    et = _problem(fb, "heat", 1, prefix="true_", mesh_n=101)
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Synthetic truth (FD)")
    fb.connect(et, "problem", fd, "problem")
    obs = fb.add("sensors", "data.observe", {"n_points": 400, "noise": 0.01}, label="Sensors (1% noise)")
    fb.connect(fd, "solution", obs, "solution")
    fb.connect(et, "problem", obs, "problem")
    ei = _problem(fb, "heat", 1, prefix="inv_", mesh_n=101,
                  eq_params={"coefficients": {}, "unknowns": ["alpha"], "exact": ""})
    disc = fb.add("discover", "inverse.discover", {"initial_guess": {"alpha": 0.5}, "adam_steps": 3000,
                                                   "lbfgs_steps": 200}, label="Inverse PINN")
    cal = fb.add("calibrate", "inverse.calibrate", {"bounds": {"alpha": [1e-3, 2.0]}, "initial_guess": {"alpha": 0.5},
                                                    "n": 41}, label="Calibration (FD)")
    for s in (disc, cal):
        fb.connect(ei, "problem", s, "problem")
        fb.connect(obs, "data", s, "data")
    return fb.flow({"kind": "reproduction"})


def elnino():
    fb = FlowBuilder("El Nino forecast (ENSO models)",
                     "Nino 3.4 from OISST with the WOD thermocline (warm-water volume) and ERA5 wind stress: the ENSO "
                     "library's baselines (persistence, climatology, AR), recharge and delayed oscillators and the "
                     "linear inverse model (quick settings), tuned on 1982-2011 and scored on the untouched test period "
                     "from 2012 against climatology and persistence. Left out of this example for run time on a shared "
                     "4-core CPU: the intermediate coupled model (im_nudged, over 55 minutes without finishing) and "
                     "gradient boosting (gbm, over 18 minutes); both are in the node's model list.")
    f = fb.add("forecast", "forecast.enso", {"models": ["persistence", "climatology", "ar", "ro", "delayed_oscillator",
                                                         "lim"], "quick": True},
               label="ENSO forecast")
    r = fb.add("report", "export.report", {"title": "El Nino forecast"}, label="Report")
    fb.connect(f, "metrics", r, "items")
    return fb.flow({"kind": "reproduction"})


def takeoff():
    fb = FlowBuilder("Aircraft take-off (flight library)",
                     "6-DOF take-off of the light-GA aircraft with the vortex lattice in the loop, against Raymer's "
                     "closed form (pinneapple_simulation.flight).")
    t = fb.add("takeoff", "flight.takeoff", {"aircraft": "light_ga", "kind": "light", "render": "blender",
                                             "camera": "side", "frames": 36, "samples": 8}, label="Take-off")
    r = fb.add("report", "export.report", {"title": "take-off"}, label="Report")
    fb.connect(t, "metrics", r, "items")
    return fb.flow({"kind": "reproduction"})


def lunar_rover(quick: bool = False):
    """examples/studio/07_lunar_rover_dashboard.py as a flow (``quick``: its --quick settings, 12 s and 24 frames)."""
    fb = FlowBuilder("Lunar rover dashboard" + (" (quick)" if quick else ""),
                     "NASA VIPER crossing a lunar crater on GRC-1: multibody dynamics with Wong-Reece wheels, the model "
                     "zoo's wheel-contact PINN checked against the run's physics, and the animated dashboard (planetary "
                     "render with the ruts coloured by normal stress, charts synced to the animation) with a horizon "
                     "check on the frames. The flow form of examples/studio/07_lunar_rover_dashboard.py"
                     + (" --quick (12 s traverse, 24 frames)." if quick else " (45 s traverse, 150 frames)."))
    veh = fb.add("vehicle", "vehicle.spec", {"preset": "viper", "total_mass": 450.0}, label="VIPER")
    soil = fb.add("soil", "terramechanics.soil", {"preset": "GRC-1"}, label="GRC-1 soil")
    site = fb.add("site", "scenario.site", {"scenario": "crater_site"}, label="Crater site")
    sim = fb.add("simulate", "vehicle.simulate", {"duration": 12.0 if quick else 45.0, "speed": 0.2},
                 label="Traverse")
    wr = fb.add("wheel", "terramechanics.wheel_model", {}, label="Wong-Reece wheel")
    zoo = fb.add("pinn", "terramechanics.zoo_surrogate", {"zoo_name": "rover_contact_grc1_viper"}, label="Zoo PINN")
    db = fb.add("dashboard", "vehicle.dashboard", {"frames": 24 if quick else 150, "gif_every": 1 if quick else 2},
                label="Dashboard")
    lab = fb.add("lab", "validate.lab", {"title": "Lunar rover dashboard"}, label="Lab")
    fb.connect(veh, "vehicle", sim, "vehicle")
    fb.connect(soil, "soil", sim, "soil")
    fb.connect(site, "site", sim, "site")
    fb.connect(veh, "vehicle", wr, "vehicle")
    fb.connect(soil, "soil", wr, "soil")
    fb.connect(wr, "model", zoo, "model")
    fb.connect(sim, "run", db, "run")
    fb.connect(wr, "model", db, "model")
    fb.connect(zoo, "surrogate", db, "surrogate")
    fb.connect(zoo, "metrics", lab, "validation")
    fb.connect(db, "metrics", lab, "validation")
    return fb.flow({"kind": "reproduction", "reproduces": "examples/studio/07_lunar_rover_dashboard.py"
                    + (" --quick" if quick else "")})


def planned(name: str) -> pf.Flow:
    res = pf.plan(PROMPTS[name], backend="heuristic")
    p = res["proposals"][0]
    f = pf.Flow.from_dict(p["flow"])
    f.name = p["title"]
    f.meta.update({"kind": "planned", "prompt": PROMPTS[name], "planner": res["planner"],
                   "detected": {k: res["detected"][k] for k in ("family", "dimension", "intents", "boundary")}})
    return f


def to_3d(f: pf.Flow) -> pf.Flow:
    """The user's edit: the domain node switched to 3-D (the equations follow their template)."""
    d = pf.Flow.from_dict(json.loads(f.to_json()))
    d.id, d.name = d.id + "-3d", d.name + " (edited to 3-D)"
    for n in d.nodes:
        if n.type == "geometry.domain":
            n.params["dimension"] = 3
        if n.type == "geometry.mesh":
            n.params["n"] = 21
        if n.type == "solver.classical" and n.params.get("method") == "fem":
            n.params["n"] = 11
        if n.type == "train.pinn":
            n.params.update(adam_steps=3000, n_collocation=4000)
    d.meta["edit"] = "geometry.domain dimension 2 -> 3"
    return d


def main(names=()):
    """Write the example flows (``names``: only these; the planned ones are re-planned only when asked, since a new
    plan has new decision ids)."""
    OUT.mkdir(parents=True, exist_ok=True)
    flows = {"heat_1d": heat_1d, "burgers_1d": burgers_1d, "poisson_2d": poisson_2d,
             "operator_burgers": operator_burgers, "inverse_heat": inverse_heat, "elnino_forecast": elnino,
             "aircraft_takeoff": takeoff, "lunar_rover_dashboard": lunar_rover,
             "lunar_rover_dashboard_quick": lambda: lunar_rover(quick=True)}
    for name in PROMPTS:
        flows[name] = lambda name=name: planned(name)
    flows["planned_plate_2d_to_3d_edited"] = lambda: to_3d(planned("planned_plate_2d_to_3d"))
    flows = {k: fn() for k, fn in flows.items() if not names or k in names}
    for name, f in flows.items():
        issues = [i for i in pf.validate_flow(f) if i.level == "error"]
        assert not issues, (name, issues)
        f.save(OUT / f"{name}.json")
        print(f"{name}: {len(f.nodes)} nodes")


if __name__ == "__main__":
    main(sys.argv[1:])
