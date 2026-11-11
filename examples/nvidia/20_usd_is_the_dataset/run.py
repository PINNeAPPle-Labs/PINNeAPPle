"""One OpenUSD stage that is the dataset, the model's input and the receipt.

A moving Gaussian heat source scans a 100 mm steel plate. PINNeAPPle solves the transient conduction with its immersed
finite-cell solver (no conforming mesh of the plate is built: the geometry is a box cut by a background grid), and
every scan is written as an OpenUSD stage carrying **two** primvars keyed over the same timeline: the energy the
source deposited up to that frame (``energy``, J/m3) and the temperature the plate reached (``temperature``, K). One
stage per scan variant; that folder of ``.usda`` files **is** the dataset.

The operator learns ``energy -> temperature`` -- the deposited energy map in, the field it produced out. That map is
well posed (the first law ties them through ``rho c (T - T0) = E`` up to the diffusion that moves heat without
creating it), so it can be checked without the reference: the integral of the prediction must account for the energy
the file says was deposited. The prediction for a scan the operator never saw is then written **back into that same
stage** as a second mesh with its own time samples, next to the reference and the trust decision.

So the file answers, on its own: what geometry was scanned, what energy went in, what the solver said, what the
operator predicted, how far it drifted, and whether anyone should use it. Open any stage in USD Composer and the
timeline plays the comparison.

    python examples/nvidia/20_usd_is_the_dataset/run.py            # ~20 min on a busy laptop
    python examples/nvidia/20_usd_is_the_dataset/run.py --smoke    # seconds (CI)
"""
import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_analysis.trust import Check, TrustReport
from pinneapple_lab import experiment
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    physicsnemo_fno,
    train_grid_operator,
)
from pinneapple_simulation.immersed.assembly import Discretization
from pinneapple_simulation.immersed.geometry import box
from pinneapple_simulation.immersed.transient import solve_transient_heat
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure, field_panels
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    field_colors,
    grid_surface,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
L = 0.10                  # plate side [m]
K_SOLID = 45.0            # W/m/K (austenitic steel)
RHO_C = 3.8e6             # J/m3/K
SIGMA = 0.005             # spot radius [m]
Q_PEAK = 4.0e9            # peak volumetric power of the source [W/m3]
T0 = 300.0                # initial plate temperature [K]
DT = 0.02                 # solver time step [s]
N_STEPS, RECORD_EVERY = 96, 12          # 96 steps of 0.02 s, 9 recorded frames (t = 0 ... 1.92 s)
N_GRID = 32               # the operator's grid (each USD primvar holds N_GRID x N_GRID values)
X_START = 0.005           # the spot enters the plate here and crosses it in the simulated window

#: scan variants: kind, y0 [m], wave amplitude [m], speed [m/s], power factor. The last two are held out; the very
#: last one is a different kind of path (a wave, not a line), which is the point of the applicability check.
VARIANTS = [
    ("v00_line_y30", "line", 0.030, 0.0, 0.052, 1.00),
    ("v01_line_y50", "line", 0.050, 0.0, 0.052, 1.00),
    ("v02_line_y70", "line", 0.070, 0.0, 0.052, 1.00),
    ("v03_line_y40", "line", 0.040, 0.0, 0.052, 0.75),
    ("v04_line_y60", "line", 0.060, 0.0, 0.052, 1.25),
    ("v05_line_y30_fast", "line", 0.030, 0.0, 0.070, 1.00),
    ("v06_line_y60_slow", "line", 0.060, 0.0, 0.036, 1.00),
    ("v07_line_y45", "line", 0.045, 0.0, 0.052, 1.10),
    ("v08_line_y55", "line", 0.055, 0.0, 0.060, 0.90),
    ("v09_line_y35", "line", 0.035, 0.0, 0.045, 1.15),
    ("v10_line_y50_heldout", "line", 0.050, 0.0, 0.065, 1.05),      # held out, same kind of path
    ("v11_wave_y50_heldout", "wave", 0.050, 0.020, 0.052, 1.00),    # held out, different kind of path
]
HELD_OUT = ("v10_line_y50_heldout", "v11_wave_y50_heldout")


def source_field(X, t, kind, y0, amp, speed, power):
    """The moving Gaussian source ``q(x, y, t)`` [W/m3]."""
    xs = X_START + speed * t
    ys = y0 + (amp * np.sin(2 * np.pi * xs / L) if kind == "wave" else 0.0)
    return power * Q_PEAK * np.exp(-((X[:, 0] - xs) ** 2 + (X[:, 1] - ys) ** 2) / (2 * SIGMA ** 2))


def solve_scan(variant, h, n_grid, n_steps, record_every):
    """The scan as PINNeAPPle's immersed transient heat solver sees it, sampled on the operator's grid.

    Returns ``(temperature, energy, times, disc)``: the field at every recorded frame, and the energy deposited up to
    that frame on the same points (the running integral of the source, trapezoid in time).
    """
    name, kind, y0, amp, speed, power = variant
    disc = Discretization(box([0, 0], [L, L]), h=h, order=1)
    c = (np.arange(n_grid) + 0.5) * L / n_grid
    Xg, Yg = np.meshgrid(c, c, indexing="ij")
    Xq = np.c_[Xg.ravel(), Yg.ravel()]

    def q(P, t):
        return source_field(P, t, kind, y0, amp, speed, power)

    sol = solve_transient_heat(disc, conductivity=K_SOLID, rho_c=RHO_C, source=q, bcs=(), T0=T0, dt=DT,
                               n_steps=n_steps, record_every=record_every)
    frames = np.stack([sol(Xq, k) for k in range(len(sol.times))]).reshape(len(sol.times), n_grid, n_grid)
    deposited, acc, q_prev, times = [np.zeros_like(Xq[:, 0])], np.zeros_like(Xq[:, 0]), q(Xq, 0.0), [0.0]
    for k in range(1, n_steps + 1):
        q_now = q(Xq, k * DT)
        acc = acc + 0.5 * (q_now + q_prev) * DT          # energy deposited up to t = k dt [J/m3]
        q_prev = q_now
        if k % record_every == 0 or k == n_steps:
            deposited.append(acc.copy())
            times.append(k * DT)
    energy = np.stack(deposited).reshape(len(deposited), n_grid, n_grid)
    return frames, energy, np.asarray(times, float), disc


def scan_path(kind, y0, amp, speed, times, n=40):
    """The spot's track over the window, as a polyline [m] (the curve a reviewer sees in the stage)."""
    x = np.linspace(X_START, X_START + speed * max(times), n)
    y = np.full_like(x, y0) + (amp * np.sin(2 * np.pi * x / L) if kind == "wave" else 0.0)
    z = np.full_like(x, 0.0015)
    return np.c_[x, y, z]


# ---------------------------------------------------------------------------------------- the stage (write)
def author_stage(path, variant, times, frames, energy, Xg, Yg, vrange):
    """One scan as an OpenUSD stage: the plate with the energy and the temperature keyed over the same timeline, the
    spot track, and the solver's own verdict on the prim."""
    name, kind, y0, amp, speed, power = variant
    V, F = grid_surface(Xg, Yg)
    st = FieldStage(path, f"PINNeAPPle scan {name}: moving heat source on a steel plate", fps=24)
    plate = st.add_mesh("Plate", V, F, fields={"temperature": frames[0].ravel()},
                        units={"temperature": "K", "energy": "J/m3"}, color_by="temperature", cmap="inferno",
                        vrange=vrange)
    for field, data in (("temperature", frames), ("energy", energy)):
        st.add_time_samples(plate, field, times, [d.ravel() for d in data], cmap="inferno", vrange=vrange,
                            color=(field == "temperature"))
    st.add_curves("ScanTrack", [scan_path(kind, y0, amp, speed, times)], width=0.0012)
    st.set_trust(plate, "APPROVED", [
        {"name": "solver: first law (rho c (T - T0) vs deposited energy)", "kind": "physics"},
        {"name": "solver: stays above T0", "kind": "physics"},
    ], solver="pinneapple_simulation.immersed.solve_transient_heat", material="steel 1.4301",
        power_factor=power, speed_m_s=speed, k_W_mK=K_SOLID, rho_c_J_m3K=RHO_C)
    st.save()
    return plate


def read_stage_fields(path, primvars=("temperature", "energy")):
    """Read time-sampled primvars back out of a USD file: ``{name: frames (F, m)}``, ``(points, times in seconds)``.

    This is the only door the training data comes through: the solver's arrays are not used after the stage is
    written, so a pair the operator learns from is a pair that is really in the file.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(os.fspath(path))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        if not api.GetPrimvar(primvars[0]):
            continue
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        fps = float(stage.GetTimeCodesPerSecond())
        out, codes = {}, None
        for name in primvars:
            pv = api.GetPrimvar(name)
            codes = np.asarray(pv.GetTimeSamples(), float)
            out[name] = np.stack([np.asarray(pv.Get(tc), float) for tc in codes])
        return out, pts, codes / fps
    raise KeyError(f"no mesh with primvar {primvars[0]!r} in {path}")


def read_trust(path, primvar):
    """The ``pinneapple:trust:decision`` stored on the prim that carries ``primvar``, if any."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(os.fspath(path))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.PrimvarsAPI(prim).GetPrimvar(primvar):
            d = prim.GetAttribute("pinneapple:trust:decision").Get()
            return str(prim.GetPath()), (None if d is None else str(d))
    return None, None


# ----------------------------------------------------------------------------------- the stage (write back)
def write_prediction(path, V, F, times, prediction, vrange, decision, checks, **info):
    """Open the stage again and add the prediction beside the reference: a second mesh with its own time samples,
    the same colour range, and the trust decision on the prim. The file is the receipt."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(os.fspath(path))
    off = np.array([L * 1.15, 0.0, 0.0])
    mesh = UsdGeom.Mesh.Define(stage, "/World/Prediction")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p + off)) for p in V]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(F, np.int64).ravel().tolist()))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, V.min(0) + off)),
                                        Gf.Vec3f(*map(float, V.max(0) + off))]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "temperature_prediction", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    col = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    col.SetInterpolation(UsdGeom.Tokens.vertex)
    fps = stage.GetTimeCodesPerSecond()
    for t, f in zip(times, prediction, strict=True):
        tc = float(t) * fps
        pv.Set(Vt.FloatArray(np.asarray(f, np.float32).ravel().tolist()), tc)
        col.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in field_colors(f.ravel(), "inferno", vrange)]), tc)
    prim = mesh.GetPrim()
    prim.CreateAttribute("pinneapple:unit:temperature_prediction", Sdf.ValueTypeNames.String).Set("K")
    # the verdict lives with the prediction, not in a notebook
    prim.CreateAttribute("pinneapple:trust:decision", Sdf.ValueTypeNames.String).Set(str(decision))
    prim.CreateAttribute("pinneapple:trust:checks", Sdf.ValueTypeNames.String).Set(
        json.dumps([asdict(c) for c in checks], default=str))
    for k, v in info.items():
        prim.CreateAttribute(f"pinneapple:info:{k}", Sdf.ValueTypeNames.String).Set(
            v if isinstance(v, str) else json.dumps(v, default=float))
    stage.SetStartTimeCode(min(float(t) * fps for t in times))
    stage.SetEndTimeCode(max(float(t) * fps for t in times))
    stage.GetRootLayer().Save()
    return str(prim.GetPath())


# ----------------------------------------------------------------------------------------------- the figures
def comparison_gif(gif, times, ref, pred, vrange, err):
    """Reference beside the operator's prediction, frame by frame: the picture the post needs."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    frames, tmp = [], tempfile.mkdtemp(prefix="frames_")
    for k, t in enumerate(times):
        fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.6))
        for ax, f, title in ((axs[0], ref[k], "reference (PINNeAPPle immersed transient)"),
                             (axs[1], pred[k], "PhysicsNeMo FNO, read back from the .usda")):
            im = ax.imshow(f.T, origin="lower", extent=(0, 100, 0, 100), cmap="inferno", vmin=vrange[0],
                           vmax=vrange[1])
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=7)
        axs[0].set_ylabel("y [mm]", fontsize=8)
        fig.colorbar(im, ax=axs, shrink=0.85, label="T [K]")
        fig.suptitle(f"t = {t:5.2f} s     relative L2 = {err[k]:.2%}", fontsize=10)
        p = os.path.join(tmp, f"f{k:03d}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        frames.append(Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=550, loop=0, optimize=True)
    return gif


def first_law(T, energy, n_grid):
    """``rho c (T - T0)`` integrated over the plate against the deposited energy: the residual per frame, relative.

    Frames before any energy has been deposited are NaN: the balance is meaningless there (and a prediction is not
    obliged to return exactly ``T0`` for a zero input)."""
    dA = (L / n_grid) ** 2
    stored = RHO_C * (T - T0).sum(axis=(1, 2)) * dA
    went_in = energy.sum(axis=(1, 2)) * dA
    out = np.full(went_in.shape, np.nan)
    ok = went_in > 1e-3 * max(float(went_in.max()), 1e-30)
    out[ok] = np.abs(stored[ok] - went_in[ok]) / went_in[ok]
    return out


def _peak_drift(pred, ref):
    """Distance [m] between the hottest cells of two fields on the same grid."""
    (pi, pj), (ri, rj) = np.unravel_index(np.argmax(pred), pred.shape), np.unravel_index(np.argmax(ref), ref.shape)
    return float(np.hypot(pi - ri, pj - rj) * L / pred.shape[0])


def main(smoke=False, base=None):
    """``base``: folder for experiment.json, results/ and the dataset stages (default: this folder)."""
    base = base or HERE
    OUT = os.path.join(base, "results")
    DS = os.path.join(base, "dataset")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DS, exist_ok=True)
    keep = {VARIANTS[0][0], VARIANTS[1][0], *HELD_OUT} if smoke else {v[0] for v in VARIANTS}
    variants = [v for v in VARIANTS if v[0] in keep]
    n_grid, n_steps, record_every = (16, 12, 3) if smoke else (N_GRID, N_STEPS, RECORD_EVERY)
    steps = 6 if smoke else 1400
    held_out = [v for v in variants if v[0] in HELD_OUT]
    info = RunInfo("20_usd_is_the_dataset")
    architecture_diagram(
        "20 One OpenUSD stage: dataset, model input and receipt",
        [Block("scan", "Scan variants\n(moving Gaussian source)", "pinneapple", 0, 0),
         Block("solve", "PINNeAPPle immersed transient heat\nno conforming mesh of the plate", "pinneapple", 1, 0),
         Block("stage", "OpenUSD stages, one per scan\nenergy + temperature over one timeline", "omniverse", 2, 0),
         Block("read", "Read back with pxr\n(pairs come from the file)", "omniverse", 3, 0),
         Block("fno", "PhysicsNeMo FNO\nenergy -> temperature", "physicsnemo", 4, 0),
         Block("pred", "Prediction of a scan\nnever seen (held-out stage)", "physicsnemo", 4, 1),
         Block("back", "Prediction written back into\nthe same .usda + trust", "omniverse", 5, 1),
         Block("chk", "PINNeAPPle checks: first law,\nround trip, drift, applicability", "pinneapple", 6, 1),
         Block("kit", "USD Composer / Omniverse\ntimeline playback", "omniverse", 7, 0, ran=False)],
        [("scan", "solve", "x(t), y(t), P"), ("solve", "stage", "T, E"), ("stage", "read", ".usda"),
         ("read", "fno", "(E, T)"), ("fno", "pred", "T"), ("pred", "back", "T_pred(t)"),
         ("back", "chk", "read back"), ("back", "kit", ".usda"), ("chk", "stage", "decision on the prim")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The dataset is a folder of USD files; the prediction and the verdict go back into one of them")

    with experiment("examples/nvidia/20_usd_is_the_dataset",
                    title="OpenUSD as dataset, model input and receipt (moving heat source)", domain="heat transfer",
                    methods=["FNO (PhysicsNeMo)", "immersed finite cell (transient)"],
                    modules=["pinneapple_tools.nvidia.omniverse", "pinneapple_neural.physicsnemo_bridge",
                             "pinneapple_simulation.immersed.transient"],
                    out_dir=base, register=False,
                    params={"n_grid": n_grid, "n_steps": n_steps, "record_every": record_every, "dt_s": DT,
                            "budget": ComputeBudget(steps=steps).__dict__, "held_out": list(HELD_OUT),
                            "variants": [v[0] for v in variants], "rho_c_J_m3K": RHO_C, "k_W_mK": K_SOLID}) as ex:
        h = L / (n_grid + 8)                      # the cut-cell grid; the sampled grid is finer on purpose
        with info.step("solve the scans (PINNeAPPle immersed transient heat)"):
            solved = {}
            for v in variants:
                frames, energy, times, disc = solve_scan(v, h, n_grid, n_steps, record_every)
                solved[v[0]] = (frames, energy, times)
        info.mark("PINNeAPPle immersed transient heat", ran=True,
                  how=f"{len(variants)} scans, {n_steps} steps of {DT} s, {disc.d}-D cut-cell grid h = {h * 1e3:.2f} mm")

        # the solver's own physics, before anything is written: every joule that goes in is stored in the plate
        reference_law = max(float(np.nanmax(first_law(f, e, n_grid))) for f, e, _ in solved.values())
        ex.check("reference: first law holds (stored vs deposited energy, worst frame)", reference_law < 0.01,
                 value=reference_law, max=0.01,
                 detail="rho c (T - T0) integrated over the plate against the integrated source", kind="reference")
        info.mark("first law on the reference", ran=True, how=f"worst frame {reference_law:.2%} of the deposited energy")

        with info.step("author one OpenUSD stage per scan"):
            Xg, Yg = grid_xy(n_grid)
            vrange = (float(min(f.min() for f, _, _ in solved.values())),
                      float(max(f.max() for f, _, _ in solved.values())))
            for v in variants:
                frames, energy, times = solved[v[0]]
                author_stage(os.path.join(DS, f"{v[0]}.usda"), v, times, frames, energy, Xg, Yg, vrange)
        info.mark("OpenUSD stages written", ran=True, how=f"{len(variants)} .usda files in dataset/")

        # ---- from here the solver's arrays only prove the file holds them; the training never sees them
        with info.step("read the dataset back out of the .usda files"):
            def load(name):
                f, _, _ = read_stage_fields(os.path.join(DS, f"{name}.usda"))
                return (f["temperature"].reshape(len(f["temperature"]), n_grid, n_grid),
                        f["energy"].reshape(len(f["energy"]), n_grid, n_grid))

            round_trip, train_in, train_out = 0.0, [], []
            for v in variants:
                if v[0] in HELD_OUT:
                    continue
                T, E = load(v[0])
                ref_T, ref_E, _ = solved[v[0]]
                round_trip = max(round_trip, float(np.abs(T - ref_T.astype(np.float32)).max()),
                                 float(np.abs(E - ref_E.astype(np.float32)).max()))
                train_in += [E[k] for k in range(len(E))]
                train_out += [T[k] for k in range(len(T))]
            train_in, train_out = np.stack(train_in), np.stack(train_out)
        ex.check("the stage holds the solver's fields (float32 round trip)", round_trip == 0.0, value=round_trip,
                 max=0.0, detail=f"{len(train_in)} (energy, temperature) pairs read from "
                 f"{len(variants) - len(HELD_OUT)} stages", kind="sanity")
        info.mark("dataset read back with pxr", ran=True,
                  how=f"{len(train_in)} pairs, max |file - solver| = {round_trip:g}")

        with info.step("train (PhysicsNeMo FNO, energy -> temperature)"):
            fno = physicsnemo_fno(n_grid, latent_channels=8 if smoke else 32, num_fno_layers=2 if smoke else 4,
                                  num_fno_modes=4 if smoke else 12, decoder_layer_size=16 if smoke else 64)
            curve = []

            def evaluate():
                p = fno.predict(train_in[:16])
                e = float(np.mean(np.linalg.norm((p - train_out[:16]).reshape(len(p), -1), axis=1)
                                  / np.linalg.norm(train_out[:16].reshape(len(p), -1), axis=1)))
                curve.append(e)
                return {"train_rel_l2": e}

            res = train_grid_operator(fno, train_in, train_out, ComputeBudget(steps=steps), batch_size=8,
                                      lr=2e-3, evaluate=evaluate, eval_at=np.linspace(0.1, 0.9, 9), threads=3)
        info.mark("PhysicsNeMo FNO training", ran=True, how=f"CPU, {res.run.steps} steps, {res.run.wall_s:.0f} s")

        # ---- predict the held-out scans, write back into their own stages, check what the file says
        summary, results = {}, {}
        with info.step("predict, write back, check (held-out scans)"):
            V, F = grid_surface(Xg, Yg)
            for v in held_out:
                path = os.path.join(DS, f"{v[0]}.usda")
                ref, energy = load(v[0])
                times = read_stage_fields(path)[2]
                pred = fno.predict(energy)
                err = np.array([float(np.linalg.norm((pred[k] - ref[k]).ravel()) / np.linalg.norm(ref[k].ravel()))
                                for k in range(len(ref))])
                law_pred = float(np.nanmax(first_law(pred, energy, n_grid)))
                drift = float(_peak_drift(pred[-1], ref[-1]))
                checks = [
                    Check("first law on the prediction (stored vs deposited)", "supports" if law_pred <= 0.05
                          else "contradicts", critical=True, value=law_pred, limit=0.05,
                          source="physics, no reference",
                          reason="rho c (T_pred - T0) must account for the energy the file says was deposited"),
                    Check("prediction never below the initial temperature", "supports"
                          if float(pred.min()) >= T0 - 0.5 else "contradicts", critical=True,
                          value=float(T0 - pred.min()), limit=0.5, source="physics, no reference",
                          reason="0.5 K allows the Crank-Nicolson undershoot of the solver's own field"),
                    Check("relative L2 at the last frame", "supports" if err[-1] <= 0.05 else "contradicts",
                          critical=False, value=float(err[-1]), limit=0.05, source="reference"),
                    Check("hot-spot drift at the last frame", "supports" if drift <= 6e-3 else "contradicts",
                          critical=False, value=drift, limit=6e-3, source="reference",
                          reason="distance between the two maxima [m]"),
                    Check("prediction round trip (written and read back)", "supports", value=0.0, limit=0.0,
                          source="OpenUSD"),
                ]
                report = TrustReport(checks, model=f"PhysicsNeMo FNO, scan {v[0]}")
                write_prediction(path, V, F, times, pred, vrange, report.decisao, checks, scan_kind=v[1],
                                 speed_m_s=v[4], power_factor=v[5], rel_l2_last_frame=float(err[-1]),
                                 first_law=law_pred)
                back_prim, decision_on_prim = read_trust(path, "temperature_prediction")
                back = read_stage_fields(path, ("temperature_prediction",))[0]["temperature_prediction"].reshape(pred.shape)
                checks_figure(checks, os.path.join(OUT, f"checks_{v[0]}.png"),
                              title=f"{v[0]}, PINNeAPPle checks", decision=report.decisao)
                summary[v[0]] = {"kind": v[1],
                                 "held_out_reason": "same kind of path" if v[1] == "line"
                                 else "different kind of path (wave)",
                                 "relative_l2": err.tolist(), "relative_l2_last_frame": float(err[-1]),
                                 "first_law_prediction": law_pred, "decision": report.decisao,
                                 "decision_on_the_prim": decision_on_prim, "peak_drift_m": drift,
                                 "checks": [asdict(c) for c in checks]}
                results[v[0]] = (times, ref, pred, err, energy)
                ex.log_metric(f"rel_l2_last_frame__{v[0]}", float(err[-1]), "",
                              "relative L2 of the prediction against the reference field, last frame")
                ex.log_metric(f"first_law__{v[0]}", law_pred, "",
                              "worst-frame relative residual of rho c (T_pred - T0) against the deposited energy")
                ex.check(f"{v[0]}: the decision read back from the file is the one written",
                         decision_on_prim == report.decisao, detail=f"{decision_on_prim} on {back_prim}",
                         kind="sanity")
                ex.check(f"{v[0]}: prediction read back equals what was written",
                         float(np.abs(back - pred).max()) == 0.0, value=float(np.abs(back - pred).max()),
                         max=0.0, kind="sanity")
        info.mark("prediction written back into the stage", ran=True, how="pxr, same .usda, new time samples")

        with info.step("validate the stages and draw the results"):
            v10 = os.path.join(DS, "v10_line_y50_heldout.usda")
            val = validate_stage(v10, required_primvars={"/World/Plate": ["temperature", "energy"],
                                                         "/World/Prediction": ["temperature_prediction"]})
            render_stage_png(v10, os.path.join(OUT, "stage_readback.png"), time=24 * 1.44,
                             title="the held-out stage read back with OpenUSD — left: reference, right: prediction",
                             colorbar=("T [K]", vrange[0], vrange[1], "inferno"))
        ex.check("OpenUSD stage valid (reference, energy and prediction primvars)", val["ok"],
                 detail="; ".join(val["problems"]) or f"{val['meshes']} meshes, {val['time_samples']} time samples",
                 kind="sanity")
        ex.check("the prediction is in the same file as the reference and its energy",
                 read_trust(v10, "temperature_prediction")[1] is not None,
                 detail="pinneapple:trust:decision on /World/Prediction", kind="sanity")

        times, ref, pred, err, energy = results[HELD_OUT[0]]
        k = len(times) - 1
        field_panels([energy[k] / RHO_C, ref[k], pred[k], np.abs(pred[k] - ref[k])],
                     [f"energy / rho c (from the .usda), t = {times[k]:.2f} s", "temperature, reference",
                      "temperature, PhysicsNeMo FNO", "|difference|"],
                     os.path.join(OUT, "prediction.png"), cmaps=["cividis", "inferno", "inferno", "magma"],
                     shared=[(1, 2)], extent=(0, 100, 0, 100), units=["K", "K", "K", "K"], ncols=4,
                     suptitle=f"held-out scan {HELD_OUT[0]}: relative L2 {err[k]:.2%} at the last frame")
        curves_figure([{"title": "FNO training (pairs read from the .usda files)", "xlabel": "fraction of the budget",
                        "ylabel": "relative L2", "logy": True,
                        "series": [("train batch", np.linspace(0.1, 0.9, len(curve)), curve, "o-")]}],
                      os.path.join(OUT, "training.png"))
        curves_figure([{"title": "Error along the timeline, from the files", "xlabel": "time [s]",
                        "ylabel": "relative L2", "series": [(name, results[name][0], results[name][3], s)
                                                            for name, s in zip(results, ("o-", "s-"), strict=True)]},
                       {"title": "First law: stored vs deposited (prediction)", "xlabel": "time [s]",
                        "ylabel": "relative residual", "logy": True,
                        "series": [(name, results[name][0],
                                    first_law(results[name][2], results[name][4], n_grid), s)
                                   for name, s in zip(results, ("o-", "s-"), strict=True)],
                        "hlines": [(0.05, "limit", "r")]}],
                      os.path.join(OUT, "checks_curves.png"))
        comparison_gif(os.path.join(OUT, "usd_timeline.gif"), times, ref, pred, vrange, err)
        summary_full = {"variants": summary, "vrange_K": list(vrange), "reference_first_law": reference_law,
                        "round_trip": round_trip, "stage_valid": val["ok"], "fno_parameters": fno.n_parameters(),
                        "training": res.to_dict() | {"history": None},
                        "usd": {k2: val[k2] for k2 in ("ok", "problems", "meshes", "prims", "time_samples")}}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary_full, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
        ex.add_file(v10)
        ex.add_file(os.path.join(OUT, "usd_timeline.gif"))
    print(json.dumps({k2: {"decision": v["decision"], "rel_l2_last": round(v["relative_l2_last_frame"], 4),
                           "first_law": round(v["first_law_prediction"], 4), "why_held_out": v["held_out_reason"]}
                      for k2, v in summary.items()}, indent=1))
    return summary_full


def grid_xy(n_grid):
    """The operator's sample grid: cell centres of the plate divided in ``n_grid`` squares."""
    c = (np.arange(n_grid) + 0.5) * L / n_grid
    return np.meshgrid(c, c, indexing="ij")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
