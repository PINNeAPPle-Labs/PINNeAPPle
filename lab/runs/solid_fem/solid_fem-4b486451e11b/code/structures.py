"""3-D structural mechanics with the solid finite-element solver (``pinneapple_simulation.numerical_solvers.solid_fem``,
8-node hexahedra with incompatible modes), checked against closed-form beam and torsion theory and against CalculiX
on the same mesh, drawn the way a post-processor draws it (deformed mesh, von Mises in discrete bands, legend).

* ``cantilever``       -- a steel beam clamped at the root with a load on its tip face: tip deflection against
                          Timoshenko beam theory, bending stress against M c / I.
* ``simply_supported`` -- a wide beam on two line supports under pressure on its top face: mid-span deflection and
                          bending stress against beam theory.
* ``torsion``          -- a square bar clamped at one end, twisted at the other: rate of twist and the largest shear
                          stress against Saint-Venant torsion of a square section.

    python -m pinneapple_lab sweep solid_fem -g case=cantilever,simply_supported,torsion
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from ..spec import Experiment, register

CASES = {
    "cantilever": "Steel cantilever 2 m x 0.1 m x 0.2 m, clamped root, 10 kN on the tip face.",
    "simply_supported": "Steel beam 2 m x 0.3 m x 0.2 m on two line supports, 2 MPa on the top face.",
    "torsion": "Square steel bar 1 m x 0.1 m x 0.1 m, clamped at one end, 10 kN m torque on the other.",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def build(case: str, refine: int = 1):
    """Mesh, model and the closed-form references of one case."""
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM, box_mesh
    E, nu = 210e9, 0.3
    G = E / (2 * (1 + nu))
    if case == "cantilever":
        L, W, H, P = 2.0, 0.1, 0.2, 1e4
        m = box_mesh(L, W, H, 60 * refine, 4 * refine, 8 * refine)
        f = SolidFEM(m, E, nu)
        f.fix(m.nodes_on(x=0.0))
        f.load_face(m.face_nodes("x+"), (0.0, 0.0, -P))
        Iy = W * H ** 3 / 12
        ref = {"deflection": P * L ** 3 / (3 * E * Iy) + P * L / (5 / 6 * G * W * H),
               "stress": P * (L - L / 4) * (H / 2) / Iy, "x_stress": L / 4}
        return m, f, ref, {"L": L, "W": W, "H": H, "load_N": P, "E": E, "nu": nu}
    if case == "simply_supported":
        L, W, H, p = 2.0, 0.3, 0.2, 2e6
        m = box_mesh(L, W, H, 60 * refine, 8 * refine, 8 * refine)
        f = SolidFEM(m, E, nu)
        bot = m.nodes_on(z=-H / 2)
        left, right = np.intersect1d(bot, m.nodes_on(x=0.0)), np.intersect1d(bot, m.nodes_on(x=L))
        f.fix(left, (2,)).fix(right, (2,))
        f.fix(np.intersect1d(left, m.nodes_on(y=-W / 2)), (0, 1)).fix(np.intersect1d(right, m.nodes_on(y=-W / 2)), (1,))
        f.pressure(m.face_nodes("z+"), p)
        q = p * W
        Iy = W * H ** 3 / 12
        ref = {"deflection": 5 * q * L ** 4 / (384 * E * Iy) + q * L ** 2 / (8 * 5 / 6 * G * W * H),
               "stress": q * L ** 2 / 8 * (H / 2) / Iy, "x_stress": L / 2}
        return m, f, ref, {"L": L, "W": W, "H": H, "pressure_Pa": p, "E": E, "nu": nu}
    L, a, T = 1.0, 0.1, 1e4
    m = box_mesh(L, a, a, 40 * refine, 12 * refine, 12 * refine)
    f = SolidFEM(m, E, nu)
    f.fix(m.nodes_on(x=0.0))
    end = m.face_nodes("x+")
    r = m.nodes[end][:, 1:]
    tang = np.stack([-r[:, 1], r[:, 0]], 1)                     # e_x x r
    scale = T / (r ** 2).sum()
    for k, n in enumerate(end):
        f._f[3 * n + 1] += scale * tang[k, 0]
        f._f[3 * n + 2] += scale * tang[k, 1]
    beta, alpha = 0.1406, 0.208                                  # Saint-Venant, square section (Timoshenko & Goodier)
    ref = {"twist_rate": T / (beta * G * a ** 4), "shear": T / (alpha * a ** 3)}
    return m, f, ref, {"L": L, "a": a, "torque_Nm": T, "E": E, "nu": nu}


@register
class SolidFEMExperiment(Experiment):
    name = "solid_fem"
    version = "1"
    description = ("3-D linear elasticity with hexahedra with incompatible modes (C3D8I): a cantilever, a simply "
                   "supported beam and a bar in torsion, checked against beam and Saint-Venant theory and against "
                   "CalculiX on the same mesh, drawn as a post-processor draws them (deformed mesh, von Mises bands).")
    tags = ["structures", "fem", "solid-mechanics", "3d", "dataset"]
    references = ["S. P. Timoshenko and J. N. Goodier, Theory of Elasticity, 3rd ed. (1970)",
                  "E. L. Wilson, R. L. Taylor et al., incompatible displacement models (1973); R. L. Taylor, P. J. "
                  "Beresford and E. L. Wilson, Int. J. Numer. Meth. Eng. 10 (1976) 1211",
                  "CalculiX (G. Dhondt), C3D8I element"]
    limitations = {
        "cantilever": ["linear elasticity, small displacements", "the clamp restrains warping: stresses at the root "
                       "are local (excluded from the bending check)"],
        "simply_supported": ["line supports concentrate the stress at the supports (singular); the bending check "
                             "is read at mid-span", "linear elasticity, small displacements"],
        "torsion": ["the torque enters as linearly distributed nodal forces; the clamped end restrains warping, so the "
                    "twist rate is read in the middle of the bar"],
    }
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "cantilever", "refine": 1, "calculix": True, "render": True, "samples": 64}
    space = {"case": list(CASES)}

    def run(self, ctx):
        from pinneapple_simulation.numerical_solvers.solid_fem import fea_figure
        p = ctx.params
        case = p["case"]
        mesh, fem, ref, model = build(case, int(p["refine"]))
        ctx.input("model", model)
        with ctx.stage("solve"):
            res = fem.solve()
        ctx.metric("elements", res.info["elements"])
        ctx.metric("dofs", res.info["dofs"])
        ctx.metric("max_displacement_mm", res.max_displacement * 1e3)
        ctx.metric("max_von_mises_MPa", float(res.von_mises.max()) / 1e6)
        # equilibrium: the reactions balance the applied loads
        F = fem._f.reshape(-1, 3).sum(0)
        R = res.reactions.sum(0)
        ctx.check("reactions_balance_loads", value=float(np.linalg.norm(F + R) / np.abs(fem._f).sum()),
                  max=1e-8, detail="|sum of reactions + sum of applied forces| / sum of |nodal forces|", kind="physics")
        if case in ("cantilever", "simply_supported"):
            L, H = model["L"], model["H"]
            if case == "cantilever":
                d = -res.u[mesh.nodes_on(x=L, z=0.0) if len(mesh.nodes_on(x=L, z=0.0)) else mesh.face_nodes("x+"), 2].mean()
            else:
                d = -res.u[mesh.nodes_on(x=L / 2, z=0.0), 2].mean()
            xs = ref["x_stress"]
            fibre = mesh.nodes_on(x=xs, z=-H / 2) if case == "simply_supported" else mesh.nodes_on(x=xs, z=H / 2)
            s = abs(float(res.stress[fibre, 0].mean()))
            ctx.metric("deflection_mm", d * 1e3)
            ctx.metric("bending_stress_MPa", s / 1e6)
            ctx.check("deflection_vs_beam_theory", value=d, reference=ref["deflection"], rtol=0.03,
                      detail="Timoshenko beam (bending + shear, k = 5/6)", kind="reference")
            ctx.check("bending_stress_vs_Mc_over_I", value=s, reference=ref["stress"], rtol=0.03,
                      detail=f"axial stress on the outer fibre at x = {xs:g} m", kind="reference")
        else:
            L, a = model["L"], model["a"]
            xs = mesh.nodes[:, 0]
            ang = []
            for x in np.unique(xs):
                nd = mesh.nodes_on(x=x)
                r = mesh.nodes[nd][:, 1:]
                uu = res.u[nd][:, 1:]
                ang.append((x, float(((r[:, 0] * uu[:, 1] - r[:, 1] * uu[:, 0]) / ((r ** 2).sum(1) + 1e-30))[
                    (r ** 2).sum(1) > 0].mean())))
            ang = np.array(ang)
            m_ = (ang[:, 0] > 0.3 * L) & (ang[:, 0] < 0.7 * L)
            rate = float(np.polyfit(ang[m_, 0], ang[m_, 1], 1)[0])
            mid = mesh.nodes_on(x=float(np.unique(xs)[len(np.unique(xs)) // 2]), y=a / 2, z=0.0)
            tau = float(np.abs(res.stress[mid, 5]).mean())
            ctx.metric("twist_rate_deg_per_m", math.degrees(rate))
            ctx.metric("max_shear_MPa", tau / 1e6)
            ctx.check("twist_rate_vs_saint_venant", value=rate, reference=ref["twist_rate"], rtol=0.03,
                      detail="theta' = T / (beta G a^4), beta = 0.1406", kind="reference")
            ctx.check("max_shear_vs_saint_venant", value=tau, reference=ref["shear"], rtol=0.06,
                      detail="tau = T / (alpha a^3) at the middle of a side, alpha = 0.208", kind="reference")
        if p["calculix"]:
            lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
            work = os.path.join(lab_root, "_cases", os.path.basename(ctx.dir))
            with ctx.stage("calculix"):
                U = fem.calculix(work)
            shutil.rmtree(work, ignore_errors=True)
            if U is not None:
                err = float(np.abs(U - res.u).max() / np.abs(res.u).max())
                ctx.metric("calculix_max_rel_difference", err)
                ctx.check("matches_calculix_same_mesh", value=err, max=1e-3,
                          detail="largest nodal displacement difference / largest displacement, CalculiX C3D8I",
                          kind="baseline")
            else:
                ctx.log("ccx not installed: CalculiX cross-check not run")
        with ctx.stage("dataset"):
            ds = ctx.dataset("fields", description="Mesh (nodes, hexahedra) with nodal displacements and stresses "
                             "(Voigt xx, yy, zz, xy, yz, zx) and von Mises", units={"nodes": "m", "u": "m",
                                                                                   "stress": "Pa", "von_mises": "Pa"})
            ds.add(nodes=mesh.nodes.astype(np.float32), elements=mesh.elements.astype(np.int32),
                   u=res.u.astype(np.float32), stress=res.stress.astype(np.float32),
                   von_mises=res.von_mises.astype(np.float32), case=case)
        plt = _plt()
        view = {"cantilever": (22, -58), "simply_supported": (22, -58), "torsion": (24, -50)}[case]
        fig, scale = fea_figure(res, title=CASES[case], view=view)
        ctx.figure("von_mises_fea", fig, dpi=140)
        plt.close(fig)
        fig, _ = fea_figure(res, field="displacement", title=CASES[case], view=view, scale=scale)
        ctx.figure("displacement_fea", fig, dpi=140)
        plt.close(fig)
        frames = []
        vr = (float(res.von_mises.min()) / 1e6, float(res.von_mises.max()) / 1e6)
        for t in np.r_[np.linspace(0, 1, 12), np.linspace(1, 0, 12)[1:-1]]:
            from pinneapple_simulation.numerical_solvers.solid_fem import FEMResult
            rt = FEMResult(mesh, res.u * t, res.stress * t, res.reactions, res.info)
            fig, _ = fea_figure(rt, title=CASES[case], view=view, scale=scale,
                                vrange=(vr[0], vr[1]))
            fig.set_size_inches(8, 5)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0)
            plt.close(fig)
        ctx.gif("loading", frames, duration_ms=90)
        if p["render"]:
            with ctx.stage("render"):
                try:
                    import pinneapple as pp
                    from pinneapple_tools.visualization.studio.scene import Scene, Surface
                    V, T, fl = res.surface(scale)
                    fl = {"von_mises": fl["von_mises"] / 1e6, "displacement": fl["displacement"] * 1e3}
                    sc = Scene([Surface("part", V, T, "steel", fl)], axes="z_up", title=CASES[case])
                    sc.labels["von_mises"] = "von Mises stress (MPa)"
                    sc.labels["displacement"] = "Displacement (mm)"
                    pp.viz.web_viewer(sc, os.path.join(ctx.dir, "viewer"))
                    out = ctx.path("figures", "von_mises_render.jpg")
                    pp.viz.render(sc, out, field="von_mises", view=(-0.9, -1.5, 0.8), samples=int(p["samples"]),
                                  size=(1400, 800), distance=1.3)
                    ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
                except Exception as exc:                          # noqa: BLE001 - the render is optional (bpy)
                    ctx.log(f"render skipped: {exc}")
