"""Internal flow in pipes and static mixers (OpenFOAM, ``pinneapple_simulation.numerical_solvers.internal_flow``).

``pipe_flow`` runs one case of the structured O-grid pipe solver:

* ``laminar_pipe`` -- Hagen-Poiseuille flow at Re 500: friction factor against 64/Re, centreline speed against 2 U.
* ``bend_90``      -- turbulent flow (Re 5e4, k-omega SST) through a 90 degree bend of radius 2 D: the friction factor
                     downstream against Colebrook, the bend loss coefficient against Ito's correlation, the Dean
                     vortices (secondary flow) after the bend.
* ``kenics_mixer`` -- laminar flow (Re 50) through Kenics static-mixer elements cut into the pipe by snappyHexMesh,
                     with a passive scalar entering on half of the inlet; an empty pipe of the same mesh is the
                     baseline. Mixing (coefficient of variation along the pipe), the pressure-drop ratio Z and the
                     scalar balance.

    python -m pinneapple_lab sweep pipe_flow -g case=laminar_pipe,bend_90,kenics_mixer
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from ..spec import Experiment, register

CASES = {
    "laminar_pipe": "Hagen-Poiseuille flow at Re 500 in a straight pipe of 20 diameters (developed inlet profile).",
    "bend_90": "Turbulent flow (Re 5e4) through a smooth 90 degree bend of radius 2 D between straight runs of 15 D "
               "and 30 D: friction, bend loss and Dean vortices.",
    "kenics_mixer": "Laminar flow (Re 50) through six Kenics static-mixer elements with two streams entering side by "
                    "side, against the same pipe without elements.",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _flow(case: str, p: dict, mixer: int | None = None):
    from pinneapple_simulation.numerical_solvers.internal_flow import InternalFlow, Route
    D = p["D"]
    if case == "laminar_pipe":
        return InternalFlow(Route(D=D).straight(20), Re=500, iterations=p["iterations"], title="Laminar pipe, Re 500")
    if case == "bend_90":
        return InternalFlow(Route(D=D).straight(15).bend(2.0, 90).straight(30), Re=5e4, n_radial=8, wall_grading=0.5,
                            axial_cell_D=0.25, iterations=p["iterations"], title="90 degree bend, Re 50 000")
    n = p["elements"] if mixer is None else mixer
    return InternalFlow(Route(D=D).straight(18), Re=50, mixer=n, mixer_start_D=3.0, scalar=True, n_core=10,
                        n_radial=8, wall_grading=0.6, axial_cell_D=0.1, iterations=p["iterations"],
                        title=f"Kenics static mixer, {n} elements, Re 50" if n else "Empty pipe, Re 50")


@register
class PipeFlow(Experiment):
    name = "pipe_flow"
    version = "1"
    description = ("Internal flow in OpenFOAM on a structured O-grid swept along the pipe: laminar pipe against "
                   "Hagen-Poiseuille, a turbulent 90 degree bend against Colebrook and Ito, a Kenics static mixer "
                   "against an empty pipe (mixing of a passive scalar, pressure-drop ratio).")
    tags = ["cfd", "openfoam", "internal-flow", "pipes", "mixing", "3d", "dataset"]
    references = ["Hagen-Poiseuille: f = 64 / Re", "C. F. Colebrook, J. ICE 11 (1939) 133",
                  "H. Ito, J. Basic Eng. 82 (1960) 131 (pressure losses in smooth pipe bends)",
                  "W. R. Dean, Phil. Mag. 4 (1927) 208", "D. M. Hobbs and F. J. Muzzio, Chem. Eng. J. 67 (1997) 153 "
                  "(Kenics mixer, laminar)"]
    limitations = {
        "laminar_pipe": ["developed inlet profile imposed (no entrance length)"],
        "bend_90": ["RANS k-omega SST with wall functions, y+ about 30: the separation at the inner wall and the "
                    "loss are model-dependent", "1/7 power-law inlet profile instead of a fully developed one",
                    "Ito's correlation is a fit to smooth-pipe data with its own scatter (about 10 %)"],
        "kenics_mixer": ["the scalar is carried with numerical diffusion only (no molecular diffusivity): striations "
                         "thinner than a cell are smeared, so the coefficient of variation falls faster than in a "
                         "real laminar mixer", "plates cut by snappyHexMesh on a two-level refinement (about 4 cells "
                         "across the plate thickness)", "a single Reynolds number"],
    }
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "laminar_pipe", "D": 0.05, "elements": 6, "iterations": 1500, "procs": 2, "samples": 64,
              "render": True, "keep_case": False}
    space = {"case": list(CASES)}

    def run(self, ctx):
        p = dict(ctx.params)
        case = p["case"]
        if case not in CASES:
            raise ValueError(f"case must be one of {list(CASES)}")
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        work = os.path.join(lab_root, "_cases", os.path.basename(ctx.dir))
        flow = _flow(case, p)
        D = p["D"]
        slices = {"laminar_pipe": [10.0], "bend_90": [10.0, 15 + math.pi / 2, 15 + math.pi + 1, 15 + math.pi + 4,
                                                       15 + math.pi + 12],
                  "kenics_mixer": [2.9] + [3.0 + 1.5 * k for k in range(1, p["elements"] + 1)] + [16.0]}[case]
        ctx.input("route", {"D": D, "segments": flow.route.segments, "Re": flow.Re, "turbulent": flow.turbulent,
                            "mixer_elements": flow.mixer})
        with ctx.stage("cfd"):
            res = flow.solve(os.path.join(work, "main"), procs=int(p["procs"]), log=ctx.log, slice_at_D=slices)
        base = None
        if case == "kenics_mixer":
            with ctx.stage("cfd_empty_pipe"):
                base = _flow(case, p, mixer=0).solve(os.path.join(work, "empty"), procs=int(p["procs"]), log=ctx.log,
                                                    slice_at_D=[16.0])
        L = flow.route.length / D
        sec = res.sections
        sD = sec["s"] / D
        ctx.metric("cells", res.info["cells"])
        ctx.metric("U_mean_m_s", res.U)
        ctx.metric("Re", flow.Re)
        # mass: the bulk velocity is the same on every section
        clear = (sD > 1) & (sD < L - 1)
        if case == "kenics_mixer":                      # the plates take part of the section inside the mixer
            clear &= (sD < 2.5) | (sD > 3.0 + 1.5 * p["elements"] + 0.5)      # plus the refined cells around
        um = sec["u_mean"][clear]
        area_err = float(np.abs(um / res.U - 1).max())
        ctx.check("bulk_velocity_constant_along_pipe", value=area_err, max=0.03,
                  detail="volume-averaged axial velocity on every open section / inlet bulk velocity (mass "
                         "conservation and section averaging on the O-grid)", kind="physics")
        plt = _plt()
        if case == "laminar_pipe":
            f = res.friction_factor(3, 17)
            prof = res.profiles["0.5"]
            ctx.metric("friction_factor", f)
            ctx.metric("centreline_speed_over_U", float(prof[1].max()))
            ctx.check("friction_factor_vs_64_over_Re", value=f, reference=64 / flow.Re, rtol=0.04,
                      detail="Hagen-Poiseuille", kind="reference")
            ctx.check("centreline_speed_vs_2U", value=float(prof[1].max()), reference=2.0, rtol=0.02,
                      detail="parabolic profile u = 2 U (1 - r^2 / R^2)", kind="reference")
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
            rr = np.linspace(0, 1, 50)
            ax[0].plot(prof[0], prof[1], ".", ms=3, alpha=0.5, label="OpenFOAM cells")
            ax[0].plot(rr, 2 * (1 - rr ** 2), "k", lw=1, label="2 (1 - r²/R²)")
            ax[0].set_xlabel("r / R")
            ax[0].set_ylabel("u / U")
            ax[0].legend(frameon=False)
            ax[1].plot(sD, sec["p"] / (0.5 * res.U ** 2), color="#1b9e77")
            ax[1].set_xlabel("x / D")
            ax[1].set_ylabel("p / (½ U²)")
            ax[1].set_title(f"f = {f:.4f}  (64/Re = {64 / flow.Re:.4f})", fontsize=9)
            fig.tight_layout()
            ctx.figure("profile_and_pressure", fig)
        elif case == "bend_90":
            from pinneapple_simulation.numerical_solvers.internal_flow import (
                colebrook,
                ito_bend_loss,
            )
            f = res.friction_factor(L - 12, L - 2)
            G = res.gradient(L - 12, L - 2)
            ia, ib = int(np.argmin(abs(sD - 13))), int(np.argmin(abs(sD - (L - 2))))
            K_excess = (sec["p"][ia] - sec["p"][ib] - G * (sec["s"][ib] - sec["s"][ia])) / (0.5 * res.U ** 2)
            K_total = K_excess + f * 2.0 * math.pi / 2                       # + friction over the bend length (R = 2D)
            K_ito = ito_bend_loss(flow.Re, 4.0, 90)
            sec_max = max(float(np.nanmax(s["grids"]["secondary speed"])) for s in res.slices
                          if float(s["name"].split()[1]) > 15 + math.pi)
            ctx.metric("friction_factor_downstream", f)
            ctx.metric("colebrook", colebrook(flow.Re))
            ctx.metric("K_bend_total", K_total)
            ctx.metric("K_bend_excess", K_excess)
            ctx.metric("K_ito", K_ito)
            ctx.metric("secondary_speed_max_over_U", sec_max)
            ctx.check("friction_factor_vs_colebrook", value=f, reference=colebrook(flow.Re), rtol=0.08,
                      detail="smooth pipe, 18-28 D after the bend", kind="reference")
            ctx.check("bend_loss_vs_ito", value=K_total, reference=K_ito, rtol=0.25,
                      detail="total loss of the bend (excess loss over the developed gradient + friction of the "
                             "bend length) against Ito (1960), R / r = 4", kind="reference")
            ctx.check("dean_vortices_after_bend", value=sec_max, min=0.1,
                      detail="secondary speed / U on the sections after the bend", kind="physics")
            fig, ax = plt.subplots(1, 1, figsize=(7, 3.4))
            ax.plot(sD, sec["p"] / (0.5 * res.U ** 2), color="#d95f02", label="OpenFOAM, section mean")
            ax.axvspan(15, 15 + math.pi, color="0.9", label="bend")
            ax.plot(sD, (sec["p"][ib] + G * (sec["s"][ib] - sec["s"])) / (0.5 * res.U ** 2), "k--", lw=0.8,
                    label="developed gradient")
            ax.set_xlabel("arc length / D")
            ax.set_ylabel("p / (½ U²)")
            ax.set_title(f"K = {K_total:.3f} (Ito {K_ito:.3f}); f = {f:.4f} (Colebrook {colebrook(flow.Re):.4f})",
                         fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            ctx.figure("pressure_along_bend", fig)
            fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
            for a, s in zip(axs, [x for x in res.slices if float(x["name"].split()[1]) > 15][:4], strict=False):
                im = a.imshow(s["grids"]["axial speed"], origin="lower", cmap="jet", extent=(-1, 1, -1, 1))
                a.set_title(f"{float(s['name'].split()[1]) - 15 - math.pi:+.1f} D from the bend exit", fontsize=9)
                a.set_xticks([])
                a.set_yticks([])
                plt.colorbar(im, ax=a, fraction=0.046)
            fig.suptitle("Axial speed / U on cross-sections: the fast core is thrown to the outer wall (Dean flow)",
                         fontsize=10)
            fig.tight_layout()
            ctx.figure("dean_sections", fig)
        else:
            cov, cm = sec["cov"], sec["c_mean"]
            bcov = base.sections["cov"]
            end = int(np.argmin(abs(sD - 16.5)))
            ctx.metric("cov_outlet", float(cov[end]))
            ctx.metric("cov_outlet_empty_pipe", float(bcov[end]))
            ctx.metric("mixing_gain", float(bcov[end] / max(cov[end], 1e-9)))
            dp_mix = float(sec["p"][int(np.argmin(abs(sD - 2.5)))] - sec["p"][end])
            dp_empty = float(base.sections["p"][int(np.argmin(abs(sD - 2.5)))] - base.sections["p"][end])
            ctx.metric("pressure_drop_ratio_Z", dp_mix / dp_empty)
            per = []
            for k in range(1, p["elements"] + 1):
                i = int(np.argmin(abs(sD - (3.0 + 1.5 * k))))
                per.append(float(cov[i]))
            rates = [b / a for a, b in zip(per, per[1:], strict=False) if a > 0]
            ctx.metric("cov_reduction_per_element", float(np.exp(np.mean(np.log(rates)))) if rates else float("nan"))
            ctx.output("cov_after_each_element", per)
            ctx.check("scalar_flux_conserved", value=float(cm[end]), reference=0.5, rtol=0.02,
                      detail="flux-weighted mean concentration at the outlet = inlet mixture (half the inlet at 1)",
                      kind="physics")
            ctx.check("mixer_beats_empty_pipe", value=float(bcov[end] / max(cov[end], 1e-9)), min=5.0,
                      detail="coefficient of variation at the outlet, empty pipe / mixer", kind="baseline")
            ctx.check("mixing_decays_through_elements", passed=bool(all(b < a for a, b in zip(per, per[1:],
                                                                                              strict=False))),
                      detail="CoV after each element smaller than after the previous one", kind="physics")
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
            ax[0].semilogy(sD, cov, color="#7570b3", label=f"Kenics, {p['elements']} elements")
            ax[0].semilogy(base.sections["s"] / D, bcov, color="0.5", label="empty pipe")
            for k in range(p["elements"] + 1):
                ax[0].axvline(3.0 + 1.5 * k, color="0.85", lw=0.8)
            ax[0].set_xlabel("x / D")
            ax[0].set_ylabel("coefficient of variation")
            ax[0].legend(frameon=False)
            ax[1].plot(sD, sec["p"] / (0.5 * res.U ** 2), color="#7570b3", label="Kenics")
            ax[1].plot(base.sections["s"] / D, base.sections["p"] / (0.5 * res.U ** 2), color="0.5", label="empty")
            ax[1].set_xlabel("x / D")
            ax[1].set_ylabel("p / (½ U²)")
            ax[1].set_title(f"Z = Δp mixer / Δp pipe = {dp_mix / dp_empty:.2f}", fontsize=9)
            ax[1].legend(frameon=False)
            fig.tight_layout()
            ctx.figure("mixing_and_pressure", fig)
            shown = [s for s in res.slices if "concentration" in s["grids"]]
            fig, axs = plt.subplots(1, len(shown), figsize=(2.1 * len(shown), 2.5))
            for a, s in zip(np.atleast_1d(axs), shown, strict=True):
                a.imshow(s["grids"]["concentration"], origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
                x = float(s["name"].split()[1])
                k = int(round((x - 3.0) / 1.5))
                a.set_title("inlet" if x < 3 else (f"after {k}" if x < 3.0 + 1.5 * p["elements"] + 0.1 else "outlet"),
                            fontsize=9)
                a.set_axis_off()
            fig.suptitle("Concentration on cross-sections: the striations double at each element", fontsize=10)
            fig.tight_layout()
            ctx.figure("striations", fig)
        with ctx.stage("datasets"):
            ds = ctx.dataset("sections", description="Section averages along the pipe: arc length, pressure "
                             "(kinematic), bulk velocity; coefficient of variation and mean of the scalar (mixer)",
                             units={"s": "m", "p": "m2/s2", "u_mean": "m/s"})
            ds.add(**{k: np.asarray(v, np.float32) for k, v in sec.items()}, case=case, Re=flow.Re, D=D)
            sl = ctx.dataset("cross_sections", description="Cross-section grids (72 x 72, NaN outside the fluid): "
                             "axial and secondary speed / U, concentration")
            for s in res.slices:
                sl.add(station_D=float(s["name"].split()[1]), origin=np.asarray(s["origin"], np.float32),
                       u=np.asarray(s["u"], np.float32), v=np.asarray(s["v"], np.float32),
                       **{k.replace(" ", "_"): np.asarray(g, np.float32) for k, g in s["grids"].items()})
            if len(res.wall.get("xyz", [])):
                w = ctx.dataset("wall", description="Wall faces: centre, wall shear stress (kinematic), pressure",
                                units={"xyz": "m", "tau": "m2/s2", "p": "m2/s2"})
                w.add(xyz=res.wall["xyz"].astype(np.float32), tau=res.wall["tau"].astype(np.float32),
                      p=res.wall["p"].astype(np.float32))
        if p["render"]:
            with ctx.stage("render"):
                self._render(ctx, case, res, flow, p)
        if not p["keep_case"]:
            shutil.rmtree(work, ignore_errors=True)

    def _render(self, ctx, case, res, flow, p):
        import pinneapple as pp
        from pinneapple_tools.visualization.studio.scene import Surface
        extra = []
        if case == "kenics_mixer":
            from pinneapple_simulation.numerical_solvers.internal_flow import kenics_elements
            V, F = kenics_elements(flow.route.D, flow.mixer, flow.mixer_start_D)
            extra = [Surface("elements", V, F, "steel")]
        window = {"laminar_pipe": (4, 16), "bend_90": (11, 24), "kenics_mixer": (1.5, 13.5)}[case]
        sc = res.to_scene(extra_surfaces=extra, window_D=window)
        full = res.to_scene(extra_surfaces=extra)
        glb = os.path.join(ctx.dir, "outputs", "scene.glb")
        os.makedirs(os.path.dirname(glb), exist_ok=True)
        full.save(glb)
        pp.viz.web_viewer(full, os.path.join(ctx.dir, "viewer"))
        n = int(p["samples"])
        names = [s.name for s in sc.slices]
        view = {"laminar_pipe": (-0.5, -1.0, 0.45), "bend_90": (-0.3, -0.4, 1.0), "kenics_mixer": (-0.45, -1.0, 0.5)}[case]
        renders = [("streamlines", dict(lines=True, view=view, ghost=True))]
        pick = {"laminar_pipe": "axial speed", "bend_90": "axial speed", "kenics_mixer": "concentration"}[case]
        for nm in names:
            if nm.endswith(pick) and (case != "bend_90" or "16.5708" in nm or "19.1416" in nm):
                renders.append((f"section_{nm.split()[1].replace('.', 'p')}", dict(slice=nm, view=view)))
        for name, kw in renders[:5]:
            out = os.path.join(ctx.dir, "figures", f"{name}.jpg")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:
                pp.viz.render(sc, out, samples=n, size=(1400, 800), **kw)
                ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
            except Exception as exc:                              # noqa: BLE001 - a missing render is logged
                ctx.log(f"render {name} failed: {exc}")
