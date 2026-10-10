"""CalculiX through ``pinneapple_simulation.external_solvers.calculix.study``: CAD-like parts meshed by gmsh (C3D10),
structured beams (C3D8I), and the analyses an engineer runs first, each against a closed-form reference.

* ``plate_hole``       -- a plate with a central hole in tension: stress concentration against Heywood, mesh
                          convergence, equilibrium.
* ``l_bracket``        -- an L bracket with a fillet and a bolt hole, base clamped, side load on the upright: bending
                          stress in the upright against M c / I, mesh convergence of the peak stress, equilibrium.
* ``modal_cantilever`` -- eigenfrequencies and mode shapes of a cantilever against Euler-Bernoulli.
* ``buckling_column``  -- the buckling load of a column fixed at the base and free at the top against Euler.
* ``fin_heat``         -- steady conduction in an aluminium fin with convection, against the 1-D fin solution.

    python -m pinneapple_lab sweep calculix_case -g case=plate_hole,l_bracket,modal_cantilever,buckling_column,fin_heat
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from ..spec import Experiment, register

CASES = {
    "plate_hole": "Steel plate 200 x 50 x 4 mm with a 10 mm hole, 50 MPa tension (gmsh C3D10).",
    "l_bracket": "Steel L bracket 100 x 80 x 40 mm, 10 mm thick, 10 mm fillet, 12 mm bolt hole; base clamped, 2 kN "
                 "side load on the upright (gmsh C3D10).",
    "modal_cantilever": "Steel cantilever 1 m x 50 mm x 100 mm (C3D8I): first four eigenfrequencies.",
    "buckling_column": "Square steel column 2 m x 40 mm x 40 mm, fixed base, free top (C3D8I): Euler buckling.",
    "fin_heat": "Aluminium fin 100 x 20 x 4 mm, base at 100 C, convection h = 50 W/m2K to 20 C (C3D8).",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@register
class CalculixCase(Experiment):
    name = "calculix_case"
    version = "1"
    description = ("CalculiX studies built in Python (gmsh tetrahedra or structured hexahedra, node and face sets "
                   "picked by geometry): stress concentration, a filleted bracket, modes, buckling and a cooling fin, "
                   "each against its closed-form reference, with mesh convergence where it matters.")
    tags = ["structures", "fem", "calculix", "gmsh", "thermal", "modal", "3d", "dataset"]
    references = ["CalculiX (G. Dhondt), www.calculix.de", "R. B. Heywood, Designing by Photoelasticity (1952): "
                  "Kt_net = 2 + (1 - d/W)^3", "W. D. Pilkey, Peterson's Stress Concentration Factors, 3rd ed.",
                  "S. P. Timoshenko and J. M. Gere, Theory of Elastic Stability (1961)",
                  "F. P. Incropera et al., Fundamentals of Heat and Mass Transfer (fin with convective tip)"]
    limitations = {
        "plate_hole": ["Heywood's formula is a fit to plane-stress data (a few %); a 4 mm plate around a 10 mm hole is "
                       "not exactly plane stress"],
        "l_bracket": ["linear elasticity; the clamp makes the base corners singular (excluded from the checks)"],
        "modal_cantilever": ["Euler-Bernoulli neglects shear and rotary inertia (about 1 % at L/H = 10)"],
        "buckling_column": ["linear buckling (perfect column, no imperfection or plasticity)"],
        "fin_heat": ["the 1-D fin solution assumes a uniform temperature over the section (Biot number 5e-4 here)"],
    }
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "plate_hole", "render": True, "samples": 48}
    space = {"case": list(CASES)}

    def run(self, ctx):
        from pinneapple_simulation.external_solvers.calculix.study import (
            Buckle,
            FEModel,
            Frequency,
            Heat,
            Material,
            Static,
            geo_l_bracket,
            geo_plate_with_hole,
            solve,
        )
        from pinneapple_simulation.numerical_solvers.solid_fem import fea_figure
        p = ctx.params
        case = p["case"]
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        work = os.path.join(lab_root, "_cases", os.path.basename(ctx.dir))
        steel = Material("steel", 210e9, 0.3, 7850.0)
        plt = _plt()
        res = None
        title = CASES[case]

        def equilibrium(r, nodes, applied):
            R = r.reactions[nodes].sum(0)
            err = float(np.linalg.norm(R + np.asarray(applied)) / max(np.linalg.norm(applied), 1e-30))
            ctx.check("reactions_balance_loads", value=err, max=1e-4, detail="support reactions + applied load",
                      kind="physics")

        if case == "plate_hole":
            L, W, t, d, sig = 0.2, 0.05, 0.004, 0.01, 50e6
            kts = []
            for k, size in enumerate((0.004, 0.0025)):
                with ctx.stage(f"mesh_and_solve_{k}"):
                    m = FEModel.from_gmsh(geo_plate_with_hole(L, W, t, d), size=size, workdir=os.path.join(work, f"g{k}"))
                    m.material = steel
                    left = m.nodes_where(lambda X: X[:, 0] < -L / 2 + 1e-9)
                    left_bot = m.nodes_where(lambda X: (X[:, 0] < -L / 2 + 1e-9) & (X[:, 2] < 1e-9))
                    corner = m.nodes_where(lambda X: (X[:, 0] < -L / 2 + 1e-9) & (np.abs(X[:, 1] + W / 2) < 1e-9)
                                           & (X[:, 2] < 1e-9))
                    right = m.faces_where(lambda C: C[:, 0] > L / 2 - 1e-9)
                    r = solve(m, Static(fix=[(left, 1), (left_bot, 3), (corner, 2)], pressure=[(right, -sig)]),
                              os.path.join(work, f"s{k}"))
                edge = np.hypot(m.nodes[:, 0], m.nodes[:, 1]) < 0.51 * d
                kts.append(float(r.stress[edge, 0].max() / (sig * W / (W - d))))
                ctx.metric(f"elements_mesh_{k}", len(m.elements))
            res = r
            kt_ref = 2 + (1 - d / W) ** 3
            ctx.metric("Kt_net", kts[-1])
            ctx.metric("Kt_net_heywood", kt_ref)
            equilibrium(r, left, (sig * W * t, 0, 0))
            ctx.check("stress_concentration_vs_heywood", value=kts[-1], reference=kt_ref, rtol=0.05,
                      detail="peak axial stress at the hole / net-section stress", kind="reference")
            ctx.check("mesh_convergence_Kt", value=abs(kts[1] / kts[0] - 1), max=0.02,
                      detail="change of Kt from the 4 mm to the 2.5 mm mesh", kind="sanity")
        elif case == "l_bracket":
            a, b, w, t, P = 0.1, 0.08, 0.04, 0.01, 2000.0
            peaks = []
            for k, size in enumerate((0.004, 0.0025)):
                with ctx.stage(f"mesh_and_solve_{k}"):
                    m = FEModel.from_gmsh(geo_l_bracket(a, b, w, t), size=size, workdir=os.path.join(work, f"g{k}"))
                    m.material = steel
                    base = m.nodes_where(lambda X: X[:, 2] < 1e-9)
                    top = m.nodes_where(lambda X: X[:, 2] > b - 1e-9)
                    r = solve(m, Static(fix=[(base, (1, 2, 3))], loads=[(top, (P, 0, 0))]), os.path.join(work, f"s{k}"))
                fil = (m.nodes[:, 0] > t - 1e-9) & (m.nodes[:, 2] > t - 1e-9) & (m.nodes[:, 0] < t + 0.012) & \
                    (m.nodes[:, 2] < t + 0.012)
                peaks.append(float(r.von_mises[fil].max()))
                ctx.metric(f"elements_mesh_{k}", len(m.elements))
            res = r
            zc = 0.4 * b                                                 # a section of the upright below the hole
            sec = (np.abs(m.nodes[:, 2] - zc) < 0.004) & (np.abs(m.nodes[:, 1] - w / 2) < 0.006)
            outer = sec & (m.nodes[:, 0] < 1e-9)
            s_fem = float(np.abs(r.stress[outer, 2]).mean())
            s_ref = P * (b - zc) * (t / 2) / (w * t ** 3 / 12)
            ctx.metric("fillet_peak_von_mises_MPa", peaks[-1] / 1e6)
            ctx.metric("upright_bending_stress_MPa", s_fem / 1e6)
            ctx.metric("max_displacement_mm", r.max_displacement * 1e3)
            equilibrium(r, base, (P, 0, 0))
            ctx.check("upright_bending_vs_Mc_over_I", value=s_fem, reference=s_ref, rtol=0.06,
                      detail=f"axial stress on the outer face of the upright at z = {zc * 1e3:.0f} mm", kind="reference")
            ctx.check("mesh_convergence_fillet_peak", value=abs(peaks[1] / peaks[0] - 1), max=0.05,
                      detail="change of the fillet's peak von Mises stress between the two meshes", kind="sanity")
        elif case == "modal_cantilever":
            L, W, H = 1.0, 0.05, 0.1
            m = FEModel.from_box(L, W, H, 40, 2, 4, "C3D8I")
            m.material = steel
            root = m.nodes_where(lambda X: X[:, 0] < 1e-9)
            with ctx.stage("solve"):
                r = solve(m, Frequency(4, fix=[(root, (1, 2, 3))]), os.path.join(work, "f"))
            A = W * H
            f_eb = lambda I_: 1.8751 ** 2 / (2 * math.pi) * math.sqrt(210e9 * I_ / (7850 * A * L ** 4))   # noqa: E731
            ref = sorted([f_eb(H * W ** 3 / 12), f_eb(W * H ** 3 / 12)])
            for k, f in enumerate(r.frequencies[:4]):
                ctx.metric(f"f{k + 1}_Hz", float(f))
            ctx.check("first_bending_mode_vs_euler_bernoulli", value=float(r.frequencies[0]), reference=ref[0],
                      rtol=0.02, detail="weak axis", kind="reference")
            ctx.check("second_bending_mode_vs_euler_bernoulli", value=float(r.frequencies[1]), reference=ref[1],
                      rtol=0.02, detail="strong axis (shear and rotary inertia lower it by about 1 %)", kind="reference")
            res = r
        elif case == "buckling_column":
            L, a = 2.0, 0.04
            m = FEModel.from_box(L, a, a, 50, 3, 3, "C3D8I")
            m.material = steel
            base = m.nodes_where(lambda X: X[:, 0] < 1e-9)
            top = m.nodes_where(lambda X: X[:, 0] > L - 1e-9)
            with ctx.stage("solve"):
                r = solve(m, Buckle(2, fix=[(base, (1, 2, 3))], loads=[(top, (-1.0, 0, 0))]), os.path.join(work, "b"))
            Pcr = math.pi ** 2 * 210e9 * (a ** 4 / 12) / (4 * L ** 2)
            ctx.metric("buckling_load_kN", float(r.buckling_factors[0]) / 1e3)
            ctx.metric("euler_load_kN", Pcr / 1e3)
            ctx.check("buckling_load_vs_euler", value=float(r.buckling_factors[0]), reference=Pcr, rtol=0.03,
                      detail="fixed-free column, K = 2", kind="reference")
            res = r
        else:
            L, w, t, k_al, h, Tb, Ti = 0.1, 0.02, 0.004, 200.0, 50.0, 100.0, 20.0
            m = FEModel.from_box(L, w, t, 50, 4, 2, "C3D8")
            m.material = Material("aluminium", 70e9, 0.33, 2700.0, conductivity=k_al)
            base = m.nodes_where(lambda X: X[:, 0] < 1e-9)
            skin = m.faces_where(lambda C: C[:, 0] > 1e-9)
            with ctx.stage("solve"):
                r = solve(m, Heat(temperature=[(base, Tb)], film=[(skin, (Ti, h))], initial=Ti), os.path.join(work, "h"))
            Pm, A = 2 * (w + t), w * t
            mm = math.sqrt(h * Pm / (k_al * A))
            hk = h / (mm * k_al)
            x = m.nodes[:, 0]
            th = (np.cosh(mm * (L - x)) + hk * np.sinh(mm * (L - x))) / (np.cosh(mm * L) + hk * np.sinh(mm * L))
            T_ref = Ti + (Tb - Ti) * th
            err = float(np.linalg.norm(r.temperature - T_ref) / np.linalg.norm(T_ref - Ti))
            ctx.metric("tip_temperature_C", float(r.temperature[x > L - 1e-9].mean()))
            ctx.metric("tip_temperature_1d_C", float(T_ref[x > L - 1e-9].mean()))
            ctx.check("temperature_vs_1d_fin_solution", value=err, max=0.01,
                      detail="relative L2 of T - T_inf against the fin with a convective tip", kind="reference")
            res = r
            fig, ax = plt.subplots(figsize=(7, 3.4))
            o = np.argsort(x)
            ax.plot(x[o] * 1e3, r.temperature[o], ".", ms=2, alpha=0.4, label="CalculiX nodes")
            xx = np.linspace(0, L, 100)
            ax.plot(xx * 1e3, Ti + (Tb - Ti) * (np.cosh(mm * (L - xx)) + hk * np.sinh(mm * (L - xx))) /
                    (np.cosh(mm * L) + hk * np.sinh(mm * L)), "k", lw=1, label="1-D fin")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("T (C)")
            ax.legend(frameon=False)
            fig.tight_layout()
            ctx.figure("fin_temperature_profile", fig)
        # figures
        if res.u is not None and res.stress is not None:
            fig, sc = fea_figure(res, title=title, view=(24, -55) if case != "l_bracket" else (20, -125))
            ctx.figure("von_mises_fea", fig, dpi=130)
            plt.close(fig)
        if res.modes:
            for k_, md in enumerate(res.modes[:2]):
                lab = (f"Mode {k_ + 1}: {res.frequencies[k_]:.2f} Hz" if res.frequencies is not None else
                       f"Buckling mode {k_ + 1}: load factor {res.buckling_factors[k_]:.4g}")
                fig, _ = fea_figure(res, values=np.linalg.norm(md, axis=1) / np.abs(md).max(), u=md, title=title,
                                    label=lab + "\nU, Magnitude (normalised)")
                ctx.figure(f"mode_{k_ + 1}", fig, dpi=130)
                plt.close(fig)
        if res.temperature is not None:
            fig, _ = fea_figure(res, values=res.temperature, u=np.zeros_like(m.nodes), title=title,
                                label="NT, Temperature (C)", view=(28, -60))
            ctx.figure("temperature_fea", fig, dpi=130)
            plt.close(fig)
        ds = ctx.dataset("fields", description="CalculiX mesh and nodal results", units={"nodes": "m", "u": "m",
                                                                                          "stress": "Pa"})
        row = {"nodes": m.nodes.astype(np.float32), "elements": m.elements.astype(np.int32), "element": m.element,
               "case": case}
        for k_, v in (("u", res.u), ("stress", res.stress), ("temperature", res.temperature)):
            if v is not None:
                row[k_] = np.asarray(v, np.float32)
        if res.frequencies is not None:
            row["frequencies"] = np.asarray(res.frequencies, np.float32)
        if res.buckling_factors is not None:
            row["buckling_factors"] = np.asarray(res.buckling_factors, np.float32)
        ds.add(**row)
        if p["render"] and case in ("l_bracket", "plate_hole") and res.stress is not None:
            with ctx.stage("render"):
                try:
                    import pinneapple as pp
                    from pinneapple_tools.visualization.studio.scene import Scene, Surface
                    F = res.model.outward_faces()
                    used = np.unique(F)
                    remap = -np.ones(len(m.nodes), int)
                    remap[used] = np.arange(len(used))
                    T = remap[F]
                    if T.shape[1] == 4:
                        T = np.vstack([T[:, [0, 1, 2]], T[:, [0, 2, 3]]])
                    s = 0.08 * np.ptp(m.nodes, axis=0).max() / max(res.max_displacement, 1e-30)
                    V = m.nodes[used] + s * res.u[used]
                    sc_ = Scene([Surface("part", V, T, "steel", {"von_mises": res.von_mises[used] / 1e6})], axes="z_up",
                                title=title)
                    sc_.labels["von_mises"] = "von Mises stress (MPa)"
                    pp.viz.web_viewer(sc_, os.path.join(ctx.dir, "viewer"))
                    out = ctx.path("figures", "von_mises_render.jpg")
                    pp.viz.render(sc_, out, field="von_mises", view=(-1.0, -1.3, 0.8) if case == "plate_hole" else
                                  (1.2, -1.0, 0.7), samples=int(p["samples"]), size=(1400, 860), distance=1.1)
                    ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
                except Exception as exc:                          # noqa: BLE001 - the render is optional (bpy)
                    ctx.log(f"render skipped: {exc}")
        shutil.rmtree(work, ignore_errors=True)
