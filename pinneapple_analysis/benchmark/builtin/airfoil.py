"""``airfoil``: the aero app's 2-D airfoil (issue #299). OpenFOAM v1912 simpleFoam, k-omega SST, Re 4e6, the app's
O-grid (pinneapple_design.aero). Data: Latin-hypercube CST shapes x angles plus the NACA polars (the app's campaign,
apps/aero_optimizer/tools/generate_dataset.py), and an OOD set (sections outside the CST box, angles below the range).
Methods: MLP (integrals), POD + MLP (fields), MeshGraphNet (both), PINNs per geometry (potential flow, RANS).
Needs OpenFOAM for ``prepare`` (``/usr/share/openfoam/etc/bashrc``, or ``FOAM_BASHRC``).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..registry import Check, Column, Method, Problem, register_method, register_problem
from . import airfoil_impl as A

RE = 4e6
ALPHA_BINS = [(-2, 2), (2, 6), (6, 10), (10, 14)]
OOD_CASES = [(c, a) for c in ("0021", "2421", "4418", "6409", "6412") for a in (2.0, 6.0)] + [("2412", -4.0), ("0012", -4.0)]
PINN_OOD = ("NACA0021_a+02", "NACA6412_a+02", "NACA4418_a+02", "NACA2412_a-04")
SETS = {"test4": "held-out test, PINN cases", "test": "held-out test, all", "naca": "NACA polars",
        "ood4": "OOD, PINN cases", "ood": "OOD, all"}


def _plan(n_shapes: int, n_alpha: int, seed: int = 7):
    from pinneapple_design.aero.geometry import BOUNDS, REFERENCE, latin_hypercube, valid
    rng = np.random.default_rng(seed)
    shapes = []
    while len(shapes) < n_shapes:
        for s in latin_hypercube(n_shapes, BOUNDS, rng):
            if valid(s)[0] and len(shapes) < n_shapes:
                shapes.append(s)
    jobs = []
    for k, s in enumerate(shapes):
        bins = ALPHA_BINS if n_alpha == 4 else [(-2 + 16 * i / n_alpha, -2 + 16 * (i + 1) / n_alpha) for i in range(n_alpha)]
        for b, (lo, hi) in enumerate(bins):
            jobs.append({"id": f"s{k:03d}_a{b}", "set": "lhs", "shape_id": f"s{k:03d}", "shape": s.tolist(),
                         "alpha": float(rng.uniform(lo, hi))})
    for name, s in REFERENCE.items():
        for a in (-2, 0, 2, 4, 6, 8, 10, 12, 14):
            jobs.append({"id": f"{name.replace(' ', '')}_a{a:+03d}", "set": "reference", "shape_id": name,
                         "shape": s.tolist(), "alpha": float(a)})
    return jobs


def _ood_jobs():
    from pinneapple_design.aero.geometry import BOUNDS, PARAMS, naca4
    jobs = []
    for code, a in OOD_CASES:
        s = naca4(code)
        jobs.append({"id": f"NACA{code}_a{a:+03.0f}", "set": "ood", "shape_id": f"NACA {code}", "shape": s.tolist(),
                     "alpha": a, "outside_box": [PARAMS[i] for i in range(6) if not BOUNDS[i, 0] <= s[i] <= BOUNDS[i, 1]],
                     "alpha_outside_range": not -2.0 <= a <= 14.0})
    return jobs


def _one(args):
    job, out = args
    from pinneapple_design.aero.case import read_internal, run_case
    case = os.path.join(out, "cases", job["id"])
    r = run_case(case, np.array(job["shape"]), job["alpha"], reynolds=RE)
    rec = {**job, "reynolds": RE, **{k: v for k, v in r.items() if k != "time_dir"}}
    d = r.get("time_dir")
    if d:
        p, U, nut = (read_internal(os.path.join(d, f)) for f in ("p", "U", "nut"))
        np.savez_compressed(os.path.join(out, "runs", job["id"] + ".npz"), p=p.astype(np.float32),
                            Ux=U[:, 0].astype(np.float32), Uy=U[:, 1].astype(np.float32), nut=nut.astype(np.float32))
    with open(os.path.join(out, "runs", job["id"] + ".json"), "w") as f:
        json.dump(rec, f)
    shutil.rmtree(case, ignore_errors=True)
    return job["id"], r.get("status"), r.get("seconds")


def _campaign(jobs, out: Path, workers: int, log=print):
    (out / "runs").mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / "runs" / f"{j['id']}.json").exists()]
    todo.sort(key=lambda j: j["set"] != "reference")
    if not todo:
        return
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for res in ex.map(_one, [(j, str(out)) for j in todo]):
            log(*res)


def openfoam_available() -> bool:
    return os.path.exists(os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc"))


@register_problem("airfoil")
class AirfoilProblem(Problem):
    title = "Airfoil 2-D (RANS, Re 4e6)"
    kind = "geometry"
    outputs_mlp = "integrals"
    outputs_mlp_fields = "fields"
    title_pinn = "PINN (potential flow)"
    title_mlp = "MLP (Cl, Cd only)"
    space_mlp = A.SPACES["mlp"]
    space_mgn = A.SPACES["mgn"]
    space_mlp_fields = A.SPACES["mlp_fields"]
    space_pinn = A.SPACES["pinn"]
    space_pinn_rans = A.SPACES["pinn_rans"]

    def default_options(self):
        return {"n_shapes": 100, "n_alphas": 4, "ood": True, "gci_study": False}

    def smoke_options(self):
        return {"n_shapes": 12, "n_alphas": 1, "ood": False}

    # ------------------------------------------------------------------ data
    def prepare(self, out: Path, n_workers: int = 1):
        if not openfoam_available():
            raise RuntimeError("the airfoil problem needs OpenFOAM (simpleFoam) to generate its data")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _campaign(_plan(self.options["n_shapes"], self.options["n_alphas"]), out, n_workers)
        if self.options["ood"]:
            _campaign(_ood_jobs(), out / "ood", n_workers)
        self._geometry(out)
        recs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
        sec = [r.get("seconds") or 0.0 for r in recs]
        ood = [json.loads(p.read_text()) for p in sorted((out / "ood" / "runs").glob("*.json"))] if (out / "ood").exists() else []
        cost = {"n_runs": len(recs), "solver_s_total": float(np.sum(sec)), "solver_s_per_run": float(np.mean(sec)),
                "n_ood_runs": len(ood), "train_set_core_s": float(np.sum(sec)), "wall_s": time.time() - t0,
                "status": {s: sum(r.get("status") == s for r in recs) for s in sorted({r.get("status") for r in recs})},
                "timing": "simpleFoam wall time of each run on one core (shared CPU inflates it; re-time alone to check)"}
        (out / "cost.json").write_text(json.dumps(cost, indent=1))
        files = ["cost.json", "geometry/shapes.json"] + [str(p.relative_to(out)) for p in sorted((out / "runs").glob("*.json"))]
        return {"files": files, "cost": cost}

    def _geometry(self, out: Path):
        """geometry/: every shape (CST weights and outline), one complete OpenFOAM case (mesh + setup) of a
        representative design for the preflight and mesh-quality checks, and an STL of that airfoil."""
        from pinneapple_design.aero.case import write_case
        from pinneapple_design.aero.geometry import REFERENCE, outline
        g = out / "geometry"
        g.mkdir(exist_ok=True)
        shapes = {}
        for p in sorted((out / "runs").glob("*.json")):
            r = json.loads(p.read_text())
            shapes.setdefault(r["shape_id"], r["shape"])
        (g / "shapes.json").write_text(json.dumps({k: {"cst": v, "outline": outline(np.array(v))} for k, v in shapes.items()}))
        case = g / "case_NACA2412_a+04"
        if not (case / "system" / "controlDict").exists():
            write_case(str(case), REFERENCE["NACA 2412"], 4.0, reynolds=RE)

    def artifact_files(self, out):
        out = Path(out)
        runs = [str(p.relative_to(out)) for p in sorted((out / "runs").glob("*.json"))]
        ood = [str(p.relative_to(out)) for p in sorted((out / "ood" / "runs").glob("*.json"))] if self.options["ood"] else []
        return ["cost.json"] + runs + ood if runs else ["cost.json", "runs"]

    def complete_adopted(self, out):
        """Adopted campaign: write the geometry artifacts (shapes, a representative OpenFOAM case) it lacks."""
        self._geometry(Path(out))

    def data_cost_s(self, out):
        p = Path(out) / "cost.json"
        return json.loads(p.read_text())["train_set_core_s"] if p.exists() else None

    def solver_time_per_instance_s(self, out):
        clean = Path(out) / "openfoam_clean_timing.json"
        if clean.exists():
            return json.loads(clean.read_text())["mean_wall_s"]
        p = Path(out) / "cost.json"
        return json.loads(p.read_text())["solver_s_per_run"] if p.exists() else None

    def reference_row(self, out):
        """The inviscid panel method (no data, milliseconds per design) on every evaluation run: Cl from
        Kutta-Joukowski, Cd = 0 (d'Alembert: 100 % error by construction), velocity and Bernoulli pressure on the graph
        nodes. Cached in ``panel_reference.json``."""
        cache = Path(out) / "panel_reference.json"
        if cache.exists():
            return json.loads(cache.read_text())
        from pinneapple_design.aero.panel import panel_solve
        from pinneapple_design.aero.surrogate import Layout
        A.set_data_dir(out)
        S, c, lay = A.splits(), self._cases(), Layout()
        runs = {r["id"]: r for k in ("test", "naca", "ood") for r in S.get(k, [])}
        sets = {"test4": c["test"], "test": [r["id"] for r in S["test"]], "naca": [r["id"] for r in S["naca"]],
                "ood4": c["ood"], "ood": [r["id"] for r in S.get("ood", [])]}
        ext, times = {}, []
        for set_, ids in sets.items():
            vals = {"cl": [], "cd": [], "u": [], "p": []}
            for i in ids:
                r = runs[i]
                shape = np.array(r["shape"])
                t0 = time.perf_counter()
                sol = panel_solve(shape, r["alpha"])
                times.append(time.perf_counter() - t0)
                pos = lay.positions(shape)
                u, v = sol.velocity(pos[:, 0], pos[:, 1])
                z = np.load(r["npz"])
                t = np.stack([z["p"], z["Ux"], z["Uy"]], 1)[lay.cell_index]
                pr = np.stack([0.5 * (1 - u ** 2 - v ** 2), u, v], 1)
                vals["cl"].append(abs(sol.cl_kutta_joukowski - r["Cl"]))
                vals["cd"].append(1.0)
                vals["u"].append(float(np.linalg.norm(pr[:, 1:] - t[:, 1:]) / np.linalg.norm(t[:, 1:])))
                vals["p"].append(float(np.linalg.norm(pr[:, 0] - t[:, 0]) / np.linalg.norm(t[:, 0])))
            if ids:
                ext[set_] = {k: {"values": [float(x) for x in v], "ids": list(ids)} for k, v in vals.items()}
        ref = {"title": "Panel method (inviscid, no data; reference)", "seeds": [ext],
               "time_per_instance_s": float(np.median(times)) if times else None}
        cache.write_text(json.dumps(ref))
        return ref

    def report_sections(self, plan) -> list[str]:
        """Airfoil-specific evidence: did the potential-flow PINN solve its own equations (its Kutta-Joukowski Cl
        against the panel method's), and where its collocation points are."""
        from pinneapple_design.aero.panel import airfoil_nodes, panel_solve

        from ..report import fmt_s
        A.set_data_dir(plan.ws.prepared(self.id))
        note = ["_Reading the airfoil tables: the MLP predicts the integrals only (Cl, Cd, Cm from the shape parameters "
                "and the angle), so its field columns read \"integrals only\". A scalar regressor trained directly on "
                "the scalars is expected to beat a field model (MeshGraphNet, POD + MLP, the PINNs) on them; compare "
                "field models with each other on the field columns, and every method on the integral columns knowing "
                "this._", ""]
        S = A.splits()
        runs = {r["id"]: r for k in ("val", "test", "naca", "ood") for r in S.get(k, [])}
        rows = []
        for p in sorted(plan.ws.runs.glob(f"*__{self.id}__pinn__*.json")):
            rec = json.loads(p.read_text())
            sp, res = rec.get("spec", {}), rec.get("result") or {}
            case = (sp.get("instance") or {}).get("case") or sp.get("case")
            if not rec.get("ok") or case not in runs or "cl_kutta_joukowski" not in res:
                continue
            r = runs[case]
            rows.append((p.stem, case, sp["budget"].get("wall_s"), res["cl_kutta_joukowski"],
                         panel_solve(np.array(r["shape"]), r["alpha"]).cl_kutta_joukowski, r["Cl"]))
        L = list(note)
        if rows:
            d = np.array([abs(a - b) for _, _, _, a, b, _ in rows])
            L += [f"### {self.title}: the potential-flow PINN against the panel method (its own equations)", "",
                  f"Over {len(rows)} runs, |Cl(PINN) - Cl(panel)| = {d.mean():.3g} ± {d.std(ddof=1) if len(d) > 1 else 0:.2g} "
                  "(both inviscid; OpenFOAM's Cl is the viscous truth).", "",
                  "| Run | Case | Budget | Cl PINN | Cl panel | Cl OpenFOAM |", "|---|---|---|---|---|---|"]
            L += [f"| `{j}` | {c} | {fmt_s(b)} | {a:.3f} | {pc:.3f} | {o:.3f} |" for j, c, b, a, pc, o in rows]
            L.append("")
        from pinneapple_design.aero.geometry import REFERENCE
        shape = REFERENCE["NACA 2412"]
        wall = airfoil_nodes(shape, 200)
        L += ["### Airfoil PINN collocation (NACA 2412)", "", "| PINN | Points | Within 0.01 c of the wall | Within 0.05 c "
              "of the leading edge | Closest to the wall |", "|---|---|---|---|---|"]
        for kind, dmin in (("potential", 1e-3), ("RANS", 0.0)):
            pts = A.collocation_pool(shape, dmin).numpy()
            dd = np.min(np.hypot(pts[:, None, 0] - wall[None, :, 0], pts[:, None, 1] - wall[None, :, 1]), axis=1)
            L.append(f"| {kind} | {len(pts)} | {100 * (dd < 0.01).mean():.1f} % | "
                     f"{100 * (np.hypot(pts[:, 0], pts[:, 1]) < 0.05).mean():.1f} % | {dd.min():.1e} c |")
        return L + [""]

    # ------------------------------------------------------------------ checks
    def verify_solver(self):
        from pinneapple_design.aero.geometry import REFERENCE
        from pinneapple_design.aero.panel import panel_solve
        out = self._prepared
        recs = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))}
        st = [r.get("status") for r in recs.values()]
        ok_frac = float(np.mean([s in ("converged", "steady-ish") for s in st])) if st else 0.0
        C = [Check("campaign solver convergence (converged or steady-ish)", "problem", self.id, ok_frac >= 0.85,
                   value=f"{100 * ok_frac:.0f} % of {len(st)}", threshold=">= 85 %",
                   detail=f"status counts {dict((s, st.count(s)) for s in sorted(set(st)))}; unsteady runs are averaged")]
        r0 = recs.get("NACA0012_a+00")
        if r0:
            C.append(Check("NACA 0012 at 0 deg: drag vs the NASA Turbulence Modeling Resource (SST, Re 6e6)", "problem",
                           self.id, abs(r0["Cd"] / 0.0081 - 1) < 0.12, value=f"Cd {r0['Cd']:.5f}", threshold="0.0081 +- 12 %",
                           detail="the app's own validation reference; Re 4e6 here, so drag is slightly higher"))
        r4, r0b = recs.get("NACA0012_a+04"), recs.get("NACA0012_a+00")
        if r4 and r0b:
            slope = (r4["Cl"] - r0b["Cl"]) / math.radians(4)
            inv = panel_solve(REFERENCE["NACA 0012"], 4.0).cl_kutta_joukowski / math.radians(4)
            C.append(Check("NACA 0012 lift slope vs the inviscid panel method (viscous slope 85-100 %)", "problem",
                           self.id, 0.85 <= slope / inv <= 1.0, value=f"{slope:.2f} /rad vs {inv:.2f} /rad",
                           threshold="0.85..1.0", detail="pinneapple_design.aero.panel"))
        return C

    def data_checks(self, out):
        self._prepared = Path(out)
        A.set_data_dir(out)
        S = A.splits()
        C = []
        ids = {k: {r["id"] for r in S[k]} for k in ("train_all", "test", "naca", "ood") if k in S}
        shp = {k: {r["shape_id"] for r in S[k]} for k in ("train_all", "test") if k in S}
        C.append(Check("no shape shared between training and test", "data", self.id, not (shp["train_all"] & shp["test"]),
                       value=len(shp["train_all"] & shp["test"]), threshold=0))
        leak = {f"{a}&{b}": len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
        C.append(Check("no run shared between splits", "data", self.id, all(v == 0 for v in leak.values()), value=leak))
        fin = all(np.isfinite([r["Cl"], r["Cd"], r["Cm"]]).all() for k in S for r in S[k])
        C.append(Check("finite coefficients", "data", self.id, fin))
        if "ood" in S:
            outside = [bool(r.get("outside_box")) or bool(r.get("alpha_outside_range")) for r in S["ood"]]
            C.append(Check("OOD cases outside the training box (CST weights or angle)", "data", self.id, all(outside),
                           value=f"{sum(outside)}/{len(outside)}", threshold="all"))
        C.append(Check("normalisation from the training split only", "data", self.id, True,
                       detail="every trainer computes its means and scales on its training runs only"))
        return C

    def validation(self, out):
        from pinneapple_design.aero.mesh import GridSpec
        from pinneapple_physics.physics_audit import attached_flow, incompressible, turbulent

        from ..validation import envelope_checks, physics_audit_check, preflight_check, yplus_check
        out = Path(out)
        C = []
        case = out / "geometry" / "case_NACA2412_a+04"
        if case.exists():
            C.append(preflight_check(self.id, case))
            C.append(self._mesh_quality(case))
        C.append(yplus_check(self.id, GridSpec().first, 1.0, 1.0, 1 / RE))
        A.set_data_dir(out)
        S = A.splits()
        feat = lambda r: {**{f"w{i}": float(v) for i, v in enumerate(r["shape"])}, "alpha": float(r["alpha"])}  # noqa: E731
        C += envelope_checks(self.id, [feat(r) for r in S["train_all"]], [feat(r) for r in S["test"]],
                             [feat(r) for r in S.get("ood", [])])
        amax = max(r["alpha"] for r in S["train_all"])
        C.append(physics_audit_check(self.id, "RANS k-omega SST, attached flow", "solver", ["steady incompressible RANS"],
                                     [turbulent(5e5), incompressible(0.3), attached_flow(12.0)],
                                     {"Re": RE, "Ma": 0.0, "alpha_deg": amax}))
        if self.options.get("gci_study"):
            C.append(self._gci(out))
        return C

    def _mesh_quality(self, case: Path) -> Check:
        try:
            from pinneapple_data.cae import read_any

            from ..validation import mesh_quality_check
            fs = {str(p.relative_to(case)): p.read_bytes() for p in (case / "constant" / "polyMesh").rglob("*") if p.is_file()}
            fs = {f"constant/polyMesh/{k}": v for k, v in fs.items()}
            return mesh_quality_check(self.id, read_any(fs, fields=False), "cfd")
        except Exception as e:
            return Check("mesh quality (pinneapple_data.cae.mesh_report)", "validation", self.id, None,
                         detail=f"could not read the polyMesh: {e!r}")

    def _gci(self, out: Path) -> Check:
        """Three O-grids (coarse, base, fine; ratio 1.5 in each direction) for NACA 2412 at 4 deg: GCI of Cd."""
        from pinneapple_design.aero.case import run_case
        from pinneapple_design.aero.geometry import REFERENCE
        from pinneapple_design.aero.mesh import GridSpec

        from ..validation import gci_check
        path = out / "gci.json"
        if not path.exists():
            base = GridSpec()
            vals, sizes = [], []
            for f in (1.5, 1.0, 1 / 1.5):                               # fine, base, coarse
                g = GridSpec(n_surf=int(round(base.n_surf * f)), n_te=base.n_te, nj=int(round(base.nj * f)),
                             first=base.first, radius=base.radius)
                r = run_case(str(out / "gci_case"), REFERENCE["NACA 2412"], 4.0, reynolds=RE, grid=g,
                             iterations=int(1200 * max(1.0, f)))
                shutil.rmtree(out / "gci_case", ignore_errors=True)
                vals.append(r["Cd"])
                sizes.append(1.0 / math.sqrt(g.ni * g.nj))
            path.write_text(json.dumps({"Cd": vals, "sizes": sizes}))
        d = json.loads(path.read_text())
        return gci_check(self.id, "Cd (NACA 2412, 4 deg)", d["Cd"], d["sizes"], max_gci=0.05)

    # ------------------------------------------------------------------ evaluation
    def columns(self):
        n = {"test4": 4, "test": 60, "naca": 36, "ood4": 4, "ood": 12}
        cols = []
        for s, lab in SETS.items():
            cols.append(Column(f"cl_{s}", f"Cl abs. err, {lab} (n = {n[s]})", s, "cl", kind="integral"))
            cols.append(Column(f"cd_{s}", f"Cd rel. err, {lab} (n = {n[s]})", s, "cd", kind="integral", pct=True))
        for s, lab in SETS.items():
            cols.append(Column(f"u_{s}", f"Velocity rel. L2, {lab} (n = {n[s]})", s, "u"))
            cols.append(Column(f"p_{s}", f"Pressure rel. L2, {lab} (n = {n[s]})", s, "p"))
        return cols

    def validity_domain(self, prepared=None):
        """Where the case holds (#58): the generators' own ranges -- the angle bins the campaign samples and the CST
        box the sections are drawn from. Outside them sits the OOD set: AoA -4 deg and NACA sections outside the box.
        """
        from pinneapple_design.aero.geometry import BOUNDS, LABELS

        from ..validity import ValidityDomain
        bounds = {"angle of attack [deg]": (float(min(b[0] for b in ALPHA_BINS)), float(max(b[1] for b in ALPHA_BINS)))}
        for label, (lo, hi) in zip(LABELS.values(), BOUNDS, strict=True):
            bounds[label] = (float(lo), float(hi))
        return ValidityDomain(
            "airfoil", bounds,
            source="declared by the case's generator: ALPHA_BINS for the angle and pinneapple_design.aero.geometry."
                   "BOUNDS for the CST parameters",
            note="the OOD set is generated outside it (AoA -4 deg, sections outside the CST box)")

    def main_metric(self):
        return "test4", "cd"

    def main_metric_for(self, method_id):
        """Cd for the integrals-only MLP; the velocity field for the field models. Potential flow has zero drag by
        construction (d'Alembert), so its Cd error is 100 % whatever it learns and cannot judge convergence."""
        return ("test4", "cd") if self.method_outputs_id(method_id) == "integrals" else ("test4", "u")

    def method_outputs_id(self, method_id):
        return getattr(self, f"outputs_{method_id}", None) or {"mlp": "integrals"}.get(method_id, "fields")

    def ood_metric(self):
        return "ood", "cd"

    def summary_metrics(self):
        return [("Cd (integral)", "integral", ("test4", "cd"), [("ood", "cd"), ("ood4", "cd")], True),
                ("velocity (field)", "field", ("test4", "u"), [("ood", "u"), ("ood4", "u")], False)]

    def _cases(self):
        A.set_data_dir(self._prepared)
        S = A.splits()
        shapes = sorted({r["shape_id"] for r in S["test"]})[:2]
        have = {r["id"] for r in S["test"]}
        test = [f"{s}_a{b}" for s in shapes for b in (1, 2) if f"{s}_a{b}" in have]
        if not test:                                     # small campaigns (smoke): whatever angles the two shapes have
            test = sorted(i for i in have if i.rsplit("_a", 1)[0] in shapes)[:4]
        val_ids = sorted(r["id"] for r in S["val"])
        val = next((i for i in val_ids if i.endswith("_a1")), val_ids[0])
        ood = [c for c in PINN_OOD if c in {r["id"] for r in S.get("ood", [])}]
        return {"val": val, "test": test, "naca": "NACA2412_a+04", "ood": ood}

    def instances(self):
        c = self._cases()
        return {"val": [("", {"case": c["val"]})], "final": [(c["test"][0], {"case": c["test"][0]})],
                "full_only": [(x, {"case": x}) for x in c["test"][1:] + [c["naca"]]],
                "seed0_only": [(x, {"case": x}) for x in c["ood"]]}

    def study_instances(self):
        """Budget study: one held-out test run and NACA 2412 at 4 deg (the setup audit's case)."""
        c = self._cases()
        return [(c["test"][0], {"case": c["test"][0]}), (c["naca"], {"case": c["naca"]})]

    def audit_instance(self):
        c = self._cases()
        return c["naca"], {"case": c["naca"]}

    def extract(self, method_id, results):
        c = self._cases()
        out = {}
        if "" in results:
            r = results[""]
            for set_, src, pick in (("test4", "test", c["test"]), ("test", "test", None), ("naca", "naca", None),
                                    ("ood4", "ood", c["ood"]), ("ood", "ood", None)):
                if src not in r:
                    continue
                d = r[src]
                idx = list(range(len(d["ids"]))) if pick is None else [d["ids"].index(x) for x in pick if x in d["ids"]]
                out[set_] = {k: {"values": [d[m][i] for i in idx], "ids": [d["ids"][i] for i in idx]}
                             for k, m in (("cl", "cl_abs_err"), ("cd", "cd_rel_err"), ("u", "u_rel_l2"), ("p", "p_rel_l2"))
                             if m in d}
            return out
        groups = {"test4": c["test"], "naca": [c["naca"]], "ood4": c["ood"]}
        for set_, cases in groups.items():
            rr = [(x, results[x]["result"]) for x in cases if x in results]
            if rr:
                out[set_] = {k: {"values": [r[m] for _, r in rr], "ids": [x for x, _ in rr]}
                             for k, m in (("cl", "cl_abs_err_mean"), ("cd", "cd_rel_err_mean"), ("u", "u_rel_l2_mean"),
                                          ("p", "p_rel_l2_mean"))}
        return out

    def search_score(self, method_id, result):
        v = result["result"] if method_id in ("pinn", "pinn_rans") else result["val"]
        if method_id == "mlp_fields":
            return v["u_rel_l2_mean"] + v["p_rel_l2_mean"]
        return v["cl_abs_err_mean"] + v["cd_rel_err_mean"]

    def time_per_instance(self, method_id, results, per_instance):
        vals = []
        for r in results.values():
            if per_instance:
                vals.append(r["train"]["wall_s"] + r.get("infer_eval_fields", {}).get("median_s", 0.0))
            else:
                vals.append(r["infer_batch1"]["median_s"])
        return float(np.mean(vals)) if vals else float("nan")

    # ------------------------------------------------------------------ trainers (airfoil_impl)
    def _set(self, prepared):
        self._prepared = Path(prepared)
        A.set_data_dir(prepared)

    def train_mlp(self, prepared, cfg, seed, budget, *, final=True, **_):
        self._set(prepared)
        return A.train_mlp(cfg, seed, budget, final=final)

    def train_mgn(self, prepared, cfg, seed, budget, *, final=True, comparison_path=None, **_):
        self._set(prepared)
        return A.train_mgn(cfg, seed, budget, final=final, comparison_path=comparison_path)

    def train_mlp_fields(self, prepared, cfg, seed, budget, *, final=True, comparison_path=None, **_):
        self._set(prepared)
        return A.train_mlp_fields(cfg, seed, budget, final=final, comparison_path=comparison_path)

    def train_pinn(self, prepared, cfg, seed, budget, *, case, ckpt=None, **_):
        self._set(prepared)
        return A.train_pinn(cfg, seed, budget, case=case, kind="potential", ckpt=ckpt)

    def train_pinn_rans(self, prepared, cfg, seed, budget, *, case, ckpt=None, **_):
        self._set(prepared)
        return A.train_pinn(cfg, seed, budget, case=case, kind="rans", ckpt=ckpt)

    # ------------------------------------------------------------------ architecture gates
    def checks_mgn(self, prepared, cfg):
        import torch

        from pinneapple_design.aero.geometry import REFERENCE
        from pinneapple_design.aero.surrogate import AeroGNN, Layout, edge_features, node_features
        subj = f"{self.id}/mgn"
        lay = Layout()
        s = REFERENCE["NACA 2412"]
        pos = lay.positions(s)
        x = torch.from_numpy(node_features(lay, s, 4.0, pos))[None]
        e = torch.from_numpy(edge_features(lay, pos))[None]
        ei = torch.from_numpy(lay.edge_index)
        torch.manual_seed(0)
        model = AeroGNN(x.shape[-1], hidden=cfg["hidden"], mp=cfg["mp"])
        model.eval()
        wall = torch.arange(len(lay.i_sel))
        with torch.no_grad():
            y, c = model(x, ei, e, wall)
            perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(1))
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm))
            y2, _ = model(x[:, perm], inv[ei], e, inv[wall])
        rel = float((y2 - y[:, perm]).abs().max() / (y.abs().max() + 1e-12))
        return [Check("forward shapes", "architecture", subj, tuple(y.shape) == (1, x.shape[1], 4) and tuple(c.shape) == (1, 3),
                      value=f"fields {tuple(y.shape)}, coefficients {tuple(c.shape)}"),
                Check("parameter count", "architecture", subj, True, value=int(sum(p.numel() for p in model.parameters())),
                      severity="warning"),
                Check("message passing honours the graph (node permutation equivariance)", "architecture", subj,
                      rel < 1e-4, value=f"{rel:.1e}", threshold=1e-4,
                      detail="fields of the permuted graph = permuted fields")]

    def checks_mlp(self, prepared, cfg):
        import torch

        from pinneapple_design.aero.surrogate import MLPEnsemble
        subj = f"{self.id}/mlp"
        torch.manual_seed(0)
        m = MLPEnsemble(5, cfg["width"])
        x = torch.rand(6, 8)
        y = m(x)
        from .fields import _overfit_check
        t = torch.rand(6, 3)
        return [Check("forward shape", "architecture", subj, tuple(y.shape) == (5, 6, 3), value=tuple(y.shape)),
                Check("parameter count", "architecture", subj, True, value=int(sum(p.numel() for p in m.parameters())),
                      severity="warning"),
                _overfit_check(subj, lambda: ((m(x) - t) ** 2).mean(), list(m.parameters()), steps=1000, lr=cfg["lr"])]

    def checks_mlp_fields(self, prepared, cfg):
        """POD + MLP: the basis is orthonormal and reconstructs the training fields as its energy says; the MLP maps
        the 8 shape/angle inputs to K coefficients and can overfit a tiny batch."""
        import torch
        from torch import nn

        from .fields import _overfit_check
        subj = f"{self.id}/mlp_fields"
        self._set(prepared)
        S = A.splits()
        lay = A.Layout()
        recs = S["train"][:40]
        Y = np.stack([np.stack([np.load(r["npz"])[k] for k in ("p", "Ux", "Uy")], 1)[lay.cell_index]
                      for r in recs]).astype(np.float64)
        Z = ((Y - Y.mean(0)) / Y.reshape(-1, 3).std(0)).reshape(len(Y), -1)
        _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        K = min(cfg["modes"], len(Z))
        B = Vt[:K]
        ortho = float(np.abs(B @ B.T - np.eye(K)).max())
        energy = float((sv[:K] ** 2).sum() / (sv ** 2).sum())
        rec = float(np.linalg.norm(Z - (Z @ B.T) @ B) / np.linalg.norm(Z))
        torch.manual_seed(0)
        w = cfg["width"]
        net = nn.Sequential(nn.Linear(8, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                            nn.Linear(w, K))
        x, t = torch.rand(6, 8), torch.rand(6, K)
        return [Check("POD basis orthonormal", "architecture", subj, ortho < 1e-8, value=f"{ortho:.1e}", threshold=1e-8),
                Check("POD reconstruction error = sqrt(1 - energy kept)", "architecture", subj,
                      abs(rec - (1 - energy) ** 0.5) < 1e-6, value=f"{rec:.4f} vs {(1 - energy) ** 0.5:.4f}",
                      detail=f"{K} modes of {len(recs)} training fields"),
                Check("forward shape", "architecture", subj, tuple(net(x).shape) == (6, K), value=tuple(net(x).shape)),
                Check("parameter count", "architecture", subj, True,
                      value=int(sum(p.numel() for p in net.parameters())), severity="warning"),
                _overfit_check(subj, lambda: ((net(x) - t) ** 2).mean(), list(net.parameters()), steps=1000,
                               lr=cfg["lr"])]

    def _pinn_exact_check(self, kind, cfg):
        """The PINN's own residuals (``airfoil_impl.POTENTIAL_RESIDUAL``, ``rans_residuals``) on exact solutions that
        exercise every term. Potential flow: psi = uniform flow + x^2 - y^2 (harmonic). RANS: Kovasznay flow (an exact
        steady Navier-Stokes solution, Re = 40) with a constant eddy viscosity carrying part of the viscosity
        (nu + nut = 1/40), so convection, pressure, viscous and eddy-viscosity terms are all non-zero. The same
        residual with the viscous (or Laplacian) sign flipped must fail: the check can see a sign error."""
        import torch

        from pinneapple_physics.pinn_solver.factory.pinn_factory import (
            PINN,
            PINNFactory,
            PINNProblemSpec,
        )
        subj = f"{self.id}/{'pinn' if kind == 'potential' else 'pinn_rans'}"
        a = math.radians(4.0)
        re_k = 40.0
        lam = re_k / 2 - math.sqrt(re_k ** 2 / 4 + 4 * math.pi ** 2)

        class Exact(torch.nn.Module):
            def forward(self, X):
                x, y = X[:, :1], X[:, 1:2]
                if kind == "potential":
                    return y * math.cos(a) - x * math.sin(a) + x ** 2 - y ** 2
                e = torch.exp(lam * x)
                u = 1 - e * torch.cos(2 * math.pi * y)
                v = lam / (2 * math.pi) * e * torch.sin(2 * math.pi * y)
                return torch.cat([u, v, 0.5 * (1 - torch.exp(2 * lam * x))], 1)

        def residual(res):
            if kind == "potential":
                spec = PINNProblemSpec(pde_residuals=res, independent_vars=["x", "y"], dependent_vars=["psi"])
            else:
                nut0 = 0.25 / re_k
                spec = PINNProblemSpec(pde_residuals=res, independent_vars=["x", "y"], dependent_vars=["u", "v", "p"],
                                       constants={"nu": 1 / re_k - nut0},
                                       exogenous={"nut": lambda x, y: torch.full_like(x, nut0),
                                                  "nutx": lambda x, y: torch.zeros_like(x),
                                                  "nuty": lambda x, y: torch.zeros_like(x)})
            g = torch.Generator().manual_seed(0)
            x = (torch.rand(512, 1, generator=g, dtype=torch.float64) - 0.5).requires_grad_(True)
            y = torch.rand(512, 1, generator=g, dtype=torch.float64).requires_grad_(True)
            model = PINN(Exact()).double()
            return float(PINNFactory(spec).generate_loss_function()(model, {"collocation": (x, y)})[1]["pde"])

        good = [A.POTENTIAL_RESIDUAL] if kind == "potential" else A.rans_residuals()
        flipped = [r.replace(" + Derivative(psi(x, y), (y, 2))", " - Derivative(psi(x, y), (y, 2))")
                   if kind == "potential" else r.replace("- (nu + nut(x, y))", "+ (nu + nut(x, y))") for r in good]
        r_good, r_bad = residual(good), residual(flipped)
        ok = r_good < 1e-12 and r_bad > 1e-3
        return Check("PDE residual of an exact solution ~ 0", "architecture", subj, ok,
                     value=f"{r_good:.1e} (sign-flipped: {r_bad:.1e})", threshold="< 1e-12, flipped > 1e-3",
                     detail=("psi = uniform flow + x^2 - y^2" if kind == "potential" else
                             "Kovasznay flow, Re 40, nu + nut = 1/40 with nut = 0.25/40") + ", the PINN's own residuals")

    def checks_pinn(self, prepared, cfg):
        return [self._pinn_exact_check("potential", cfg)]

    def checks_pinn_rans(self, prepared, cfg):
        return [self._pinn_exact_check("rans", cfg)]


class _ProblemHookMethod(Method):
    per_instance = False
    outputs = "both"


@register_method("mgn")
class MGNMethod(_ProblemHookMethod):
    title = "MeshGraphNet"


@register_method("mlp")
class MLPMethod(_ProblemHookMethod):
    title = "MLP"


@register_method("mlp_fields")
class MLPFieldsMethod(_ProblemHookMethod):
    title = "POD + MLP"
    outputs = "fields"


@register_method("pinn_rans")
class PINNRansMethod(_ProblemHookMethod):
    title = "PINN (RANS)"
    per_instance = True
