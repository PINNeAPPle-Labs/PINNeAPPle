"""Figures of a design package (matplotlib, Agg). Every plotted number comes from a model run of the package; each
figure says which model in its title or legend."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# categorical slots in fixed order (validated reference palette of the dataviz method), ink and grid
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 2.0})
    return plt


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def pareto_figure(ctx: dict[str, Any], x: str, y: str, xlabel: str, ylabel: str, out_dir: str) -> str | None:
    """Every reduced-model evaluation of the optimisation: feasible / infeasible, and the chosen design."""
    opt = ctx.get("optimisation")
    cand = ctx.get("candidate")
    if opt is None or opt.study is None:
        return None
    ev = getattr(opt.study, "evaluations", None) or getattr(opt.study, "reference_evals", None)
    if ev is None or x not in ev.Y or y not in ev.Y:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    f = ev.feasible
    ax.scatter(ev.Y[x][~f], ev.Y[y][~f], s=10, color="#b9b8b3", label="violates a failure margin", lw=0)
    ax.scatter(ev.Y[x][f], ev.Y[y][f], s=12, color=C1, label="feasible", lw=0)
    ax.scatter([cand.outputs[x]], [cand.outputs[y]], s=90, facecolor="none", edgecolor=C2, lw=2, label="chosen")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Optimisation on the reduced model ({len(ev)} evaluations)", loc="left", fontsize=10)
    ax.legend(loc="best")
    return _save(fig, out_dir, "optimisation.png")


# ============================================================================================ loop car
def loop_figures(t, spec, d: dict[str, Any], ctx: dict[str, Any], out_dir: str) -> dict[str, str]:
    from .physics import loop_track as LT
    from .templates.loop_car import CAR, _rho_air

    plt = _plt()
    out: dict[str, str] = {}
    R, Lc, v0, m = d["radius"], d["clothoid_ratio"] * d["radius"], d["v0"], d["mass"]
    tr = t.track(d)
    p = tr.sample(1500)
    h, A = 0.5 * CAR["height"], CAR["width"] * CAR["height"]
    worst = LT.along_track(R, Lc, v0 * (1 - spec["launch_scatter"]), m, crr=spec["crr_max"], cd=spec["cd_range"],
                           area=A, com_height=h, wheel_inertia=t.wheel_inertia(), rho=_rho_air(), run_out=0.3)
    same = LT.along_track(R, Lc, v0, m, crr=0.0, cd=0.0, com_height=CAR["wheel_r"], run_out=0.3)
    # ---- side view coloured by speed (worst case), with the clothoid / arc pieces
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ws = worst.profile
    v_on_track = np.interp(p["s"], ws["s"], ws["v"])
    sc = ax.scatter(p["x"] * 100, p["z"] * 100, c=v_on_track, cmap="Blues", s=6, vmin=0, lw=0)
    fig.colorbar(sc, ax=ax, label="speed, worst case\n[m/s] (1-D model)", shrink=0.9, pad=0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    k0 = tr.loop_span()
    ax.set_title(f"Track: R = {R * 100:.1f} cm arc, clothoids {Lc * 100:.1f} cm, loop {tr.height() * 100:.1f} cm "
                 "tall", loc="left", fontsize=10)
    for s_mark, lab in ((k0[0], "entry clothoid"), (k0[0] + Lc, "arc")):
        j = np.searchsorted(p["s"], s_mark)
        ax.plot(p["x"][j] * 100, p["z"][j] * 100, "o", color=C2, ms=5)
        ax.annotate(lab, (p["x"][j] * 100, p["z"][j] * 100), textcoords="offset points", xytext=(6, -12),
                    color=MUTED, fontsize=8)
    out["track"] = _save(fig, out_dir, "track_side_view.png")
    # ---- profiles along the track: speed and normal load, nominal and worst case, Newton if it ran
    fig, axs = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True)
    for run, lab, col in ((same, "1-D model, no friction, no air (Newton's conditions)", C1),
                          (worst, "1-D model, worst-case envelope (design point)", C2)):
        pr = run.profile
        axs[0].plot(pr["s"], pr["v"], color=col, label=lab)
        axs[1].plot(pr["s"], pr["n"], color=col, label=lab)
    ver = ctx.get("verification")
    if ver is not None and ver.ran and ver.result is not None and "runs" in ver.result.artifacts:
        for run, ls in zip(ver.result.artifacts["runs"], (":", "-"), strict=False):
            keep = run.s < p["s"][-1] - 0.02
            axs[0].plot(run.s[keep], np.linalg.norm(run.vel, axis=1)[keep], color=C3, lw=1.5, ls=ls,
                        label=f"Newton XPBD, no friction, dt = {run.dt:g} s")
    axs[1].axhline(0, color=INK, lw=1)
    for a in axs:
        a.axvspan(*k0, color="#f2f1ee", zorder=0)
    axs[0].set_ylabel("speed [m/s]")
    axs[1].set_ylabel("N / (m g)")
    axs[1].set_title("normal load from the centre-of-mass path: N = m (v^2 / (R - h) + g cos theta) on the arc, "
                     f"h = centre-of-mass height ({1e3 * h:.1f} mm worst case, {1e3 * CAR['wheel_r']:.0f} mm "
                     "Newton's sphere)", loc="left", fontsize=8)
    axs[1].set_xlabel("distance along the track [m] (shaded: the loop)")
    axs[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axs[0].set_title("Speed and normal load along the track", loc="left", fontsize=10)
    out["profiles"] = _save(fig, out_dir, "profiles.png")
    pf = pareto_figure(ctx, "launch_speed", "g_peak", "launch speed [m/s]", "peak load on the car [g]", out_dir)
    if pf:
        out["optimisation"] = pf
    if ver is not None and ver.ran and ver.result is not None and "trajectory" in ver.result.artifacts:
        try:
            out["gif"] = loop_gif(t, d, ver.result.artifacts["trajectory"], out_dir)
        except Exception as e:  # noqa: BLE001 - a figure must not stop the package
            out["gif_error"] = str(e)
    out.pop("gif_error", None)
    return out


def loop_gif(t, d: dict[str, Any], traj, out_dir: str, *, n_frames: int = 48, size=(720, 480)) -> str:
    """The car going round, from the Newton trajectory (software renderer, one fixed camera)."""
    from pinneapple_design.geometry_library.core import Body
    from pinneapple_design.geometry_library.kit import merge
    from pinneapple_design.geometry_library.render import render_geometry
    from pinneapple_tools.visualization.dashboard.static import write_gif

    from .package import make_geometry
    from .physics import loop_track as LT
    from .templates.loop_car import CAR

    tr = t.track(d)
    parts = LT.track_meshes(tr, thickness=d["thickness"], rail=0.008, n=400)
    track_bodies = [Body(f"track_{k}", V, F, "plastic") for k, (V, F) in parts.items()]
    body, wheels = LT.car_mesh(CAR["length"], CAR["width"], CAR["height"], CAR["wheel_r"])
    car = merge([body] + wheels)
    Vc0 = car[0] - [0, 0, CAR["wheel_r"]]               # car frame: origin at the wheel centres' height
    lo = np.min([b.vertices.min(0) for b in track_bodies], 0) - 0.03
    hi = np.max([b.vertices.max(0) for b in track_bodies], 0) + 0.03
    # frames: uniform in time until the car leaves the end of the run-out
    keep = np.flatnonzero(traj.s < tr.length() - 0.02)
    idx = keep[np.linspace(0, len(keep) - 1, n_frames).astype(int)]
    frames = []
    p = tr.sample(4000)
    for i in idx:
        # position from Newton; attitude from the track frame at the car (the car rides on the running surface)
        j = int(np.searchsorted(p["s"], traj.s[i]).clip(0, len(p["s"]) - 1))
        th = p["theta"][j]
        T = np.array([np.cos(th), 0.0, np.sin(th)])
        N = np.array([p["nx"][j], 0.0, p["nz"][j]])
        Rf = np.c_[T, np.cross(N, T), N]
        Vc = Vc0 @ Rf.T + traj.pos[i]
        bodies = track_bodies + [Body("car", Vc, car[1], "paint")]
        g = make_geometry("loop_frame", "loop", bodies)
        v = float(np.linalg.norm(traj.vel[i]))
        im = render_geometry(g, size=size, azimuth=270.0, elevation=12.0, ssaa=1, frame=(lo, hi),
                             label=f"Newton, t = {traj.t[i]:.2f} s, v = {v:.2f} m/s")
        frames.append(np.asarray(im))
    os.makedirs(out_dir, exist_ok=True)
    return write_gif(os.path.join(out_dir, "loop_newton.gif"), frames, 16)


# ============================================================================================ bracket
def bracket_figures(t, spec, d: dict[str, Any], ctx: dict[str, Any], out_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    ver = ctx.get("verification")
    pf = pareto_figure(ctx, "mass_g", "margin_yield", "mass [g]", "yield margin (>= 0 needed)", out_dir)
    if pf:
        out["optimisation"] = pf
    if ver is None or not ver.ran or ver.result is None or "model" not in ver.result.artifacts:
        return out
    plt = _plt()
    m, r = ver.result.artifacts["model"], ver.result.artifacts["result"]
    X = m.nodes
    vm = r.von_mises / 1e6
    scale = 0.15 * max(np.ptp(X, axis=0)) / max(np.abs(r.u).max(), 1e-12)
    Xd = X + scale * r.u
    surf = _tet_surface(m.elements)
    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(projection="3d")
    from matplotlib import cm
    from matplotlib.colors import Normalize
    peak = float(ver.result.outputs["sigma_peak_mpa"])
    norm = Normalize(0, peak)                     # the scale reaches the reported peak (a nodal value)
    tri = Xd[surf] * 1e3
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    col = Poly3DCollection(tri, facecolors=cm.Blues(norm(vm[surf].mean(1))), edgecolor="none")
    ax.add_collection3d(col)
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    span = (hi - lo).max()
    c = 0.5 * (lo + hi)
    for setlim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setlim(c[k] - span / 2, c[k] + span / 2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(18, -60)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="Blues"), ax=ax, shrink=0.65, label="von Mises [MPa]")
    fig.text(0.02, 0.02, "CalculiX nodal von Mises (extrapolated and averaged at the nodes); each surface triangle "
             "shows the mean of its three corners,\nso the brightest face sits a little below the peak node value "
             "in the title. Peak taken away from the clamped and the loaded faces.", fontsize=7, color=MUTED)
    ax.set_title(f"CalculiX: von Mises on the deformed shape (x{scale:.0f}); peak "
                 f"{ver.result.outputs['sigma_peak_mpa']:.0f} MPa", loc="left", fontsize=10)
    out["stress"] = _save(fig, out_dir, "bracket_von_mises.png")
    # 3-D view file with fields (glTF, fields as vertex attributes) for the field viewer
    try:
        out["field_view_glb"] = _field_glb(m, r, out_dir)
    except Exception:  # noqa: BLE001 - optional (needs scipy)
        pass
    return out


def _tet_surface(el: np.ndarray) -> np.ndarray:
    """Boundary triangles (corner nodes) of a tetrahedral mesh (4- or 10-node)."""
    c = el[:, :4]
    faces = np.vstack([c[:, [0, 2, 1]], c[:, [0, 1, 3]], c[:, [1, 2, 3]], c[:, [0, 3, 2]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


def _field_glb(m, r, out_dir: str) -> str:
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_tools.visualization.studio.fieldview import field_view

    tets = m.elements[:, :4]
    mesh = Mesh(points=m.nodes, blocks=[CellBlock("tetra", tets)])
    fv = field_view(mesh, [{"name": "von_mises", "values": r.von_mises / 1e6, "label": "von Mises", "unit": "MPa"},
                           {"name": "displacement", "values": r.u * 1e3, "label": "displacement", "unit": "mm"}],
                    deform={"displacement": r.u}, title="bracket (CalculiX)", length_unit="m")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "bracket_fields.glb")
    with open(p, "wb") as f:
        f.write(fv.glb())
    return p


# ============================================================================================ heat sink
def heatsink_figures(t, spec, d: dict[str, Any], ctx: dict[str, Any], out_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pf = pareto_figure(ctx, "mass_g", "t_source_c", "mass [g]", "source temperature, worst case [°C]", out_dir)
    if pf:
        out["optimisation"] = pf
    ver = ctx.get("verification")
    if ver is None or not ver.ran or ver.result is None:
        return out
    eng = ver.result.artifacts.get("engine", {})
    tm = eng.get("temperature_map")
    if not tm:
        return out
    plt = _plt()
    T = np.asarray(tm["values_c"], float)                 # rows along the depth, columns along the width
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(T, origin="lower", cmap="Blues", extent=(0, d["width_mm"], 0, d["depth_mm"]))
    fig.colorbar(im, ax=ax, label="base temperature [°C]")
    ax.set_xlabel("width [mm]")
    ax.set_ylabel("depth [mm]")
    ax.grid(False)
    tsrc = ver.result.outputs["t_source_c"]
    ax.set_title(f"Finite-volume temperature of the base underside: map peak {np.nanmax(T):.1f} °C\n"
                 f"(source / case temperature {tsrc:.1f} °C = base peak + the interface resistance's rise)",
                 loc="left", fontsize=9)
    out["base_temperature"] = _save(fig, out_dir, "heatsink_base_temperature.png")
    return out


# ============================================================================================ glider wing
def wing_figures(t, spec, d: dict[str, Any], ctx: dict[str, Any], out_dir: str) -> dict[str, str]:
    from .templates.glider_wing import CLOSED_FORM, REDUCED
    plt = _plt()
    out = {}
    o = t.design_options(spec)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for lvl, lab, col in ((CLOSED_FORM, "elliptic lifting line", C2), (REDUCED, "vortex lattice", C1)):
        pol = t._polar(d, spec, lvl, mass_scale=1.0, rho_scale=1.0, cd0_scale=1.0)
        ax.plot(pol["V"], pol["sink"], color=col, label=lab)
    cand = ctx["candidate"]
    ax.axvline(cand.outputs["speed_m_s"], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("flight speed [m/s]")
    ax.set_ylabel("sink rate, wing alone [m/s]")
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.annotate("design speed", (cand.outputs["speed_m_s"], 0.02), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(4, 0), color=MUTED, fontsize=8)
    ax.set_title("Wing polar at nominal mass and sea-level density", loc="left", fontsize=10)
    ax.text(0.0, -0.2, f"Design speed chosen at the worst corner: mass x{o['mass_scale']:g}, density "
                       f"x{o['rho_scale']:g}, profile drag x{o['cd0_scale']:g}", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.legend()
    out["polar"] = _save(fig, out_dir, "wing_polar.png")
    pf = pareto_figure(ctx, "area_m2", "sink_m_s", "wing area [m²]", "sink rate, worst case [m/s]", out_dir)
    if pf:
        out["optimisation"] = pf
    return out
