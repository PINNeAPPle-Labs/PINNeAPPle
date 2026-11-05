"""A surrogate of the aerodynamics of a whole 3-D aircraft, trained, used and graded in one call.

    from pinneapple_design.aero.aircraft_surrogate import aircraft_surrogate_study
    r = aircraft_surrogate_study("out/")     # out/aircraft_surrogate.mp4, out/aircraft_field.glb + .json, figures

What happens inside, every number measured in the call:

1. **Data from the solver.** Designs of the library's light aircraft (wing area, aspect ratio, taper, leading-edge
   sweep, tip twist) are drawn by a Latin hypercube; each one is solved once by the vortex lattice of wing +
   horizontal tail (:func:`~pinneapple_design.aero.vlm.solve`), and the solution, linear in the angle of attack,
   gives the pressure jump ``dCp`` on every one of its 200 panels at several angles. Some designs train the model,
   the others are held out and never seen in training.
2. **The surrogate.** A bagged ensemble of networks maps the five design parameters and the angle of attack to the
   whole ``dCp`` field; the spread of the ensemble is its uncertainty. The panel layout is the same for every
   design, so the field is a fixed-size vector.
3. **Physics-consistent coefficients.** CL and Cm (Kutta-Joukowski on the bound vortices) and CDi (Trefftz plane)
   are computed from the *predicted* field with the solver's own post-processing
   (:func:`~pinneapple_design.aero.vlm.loads_from_circulation`), so the coefficients and the field never disagree.
4. **Inference against the data.** On the held-out designs, the prediction is compared with the solver and with
   the nearest training design (a lookup baseline), and one design outside the trained ranges is checked for an
   out-of-distribution flag.
5. **VeriPhysics grade.** The checks below decide APPROVED / REVIEW / REJECT with the library's grading policy
   (:func:`pinneapple_veriphysics.policy.decide`): ``grade.reference`` (field error on held-out designs),
   ``grade.applicability`` (in-range designs accepted, the out-of-range one flagged), ``grade.robustness``
   (smooth response to small input changes), and the aerodynamic checks ``aero.lift_from_field``,
   ``aero.linearity_alpha`` (the lattice is linear in the angle of attack), ``aero.symmetry`` (left and right
   wing carry the same load) and ``aero.munk_bound`` (span efficiency at most 1).

The vortex lattice is inviscid, small-angle and has no fuselage: the surrogate learns that model, not a viscous
flow. Needs torch and matplotlib; the MP4 needs ffmpeg (on the PATH, or through ``imageio-ffmpeg``).
"""
from __future__ import annotations

import dataclasses
import math
import os
import shutil
import time
from typing import Any

import numpy as np

from .aircraft3d import airframe_from, baseline_x
from .vlm import Lattice, build_lattice, loads_from_circulation, solve

__all__ = ["DESIGN", "BOUNDS", "AircraftSurrogate", "FieldEnsemble", "aircraft_surrogate_study", "dataset",
           "load_surrogate", "panel_dcp", "panel_quads"]

MODEL_FORMAT = "pinneapple.aero.aircraft_surrogate.FieldEnsemble/1"

DESIGN = ("wing_area", "aspect_ratio", "taper", "sweep_le", "twist", "alpha")
BOUNDS = {"wing_area": (12.0, 20.0), "aspect_ratio": (6.0, 10.0), "taper": (0.45, 1.0), "sweep_le": (0.0, 10.0),
          "twist": (-5.0, 0.0), "alpha": (-2.0, 8.0)}
UNITS = {"wing_area": "m²", "aspect_ratio": "-", "taper": "-", "sweep_le": "deg", "twist": "deg", "alpha": "deg"}
PROBE = {"wing_area": 16.0, "aspect_ratio": 12.5, "taper": 0.6, "sweep_le": 16.0, "twist": -2.0, "alpha": 4.0}
NC, NS_WING, NS_TAIL = 5, 14, 6

BG, FG, MUTED, GRID = "#0b0f17", "#e8edf5", "#8a96a8", "#1c2433"
AMBER, TEAL, RED, VIOLET = "#ffb547", "#2ee6c5", "#ff5d6c", "#9d8cff"


# ------------------------------------------------------------------------------------------------- the data
def _airframe(row: dict[str, float]):
    x = baseline_x()
    af = airframe_from(x)
    return dataclasses.replace(af, **{k: float(row[k]) for k in DESIGN if k != "alpha"})


def panel_dcp(L: Lattice, gamma: np.ndarray) -> np.ndarray:
    """Pressure jump of each panel for its circulation (per unit free-stream speed): ``2 Gamma / dx``, ``dx`` the
    panel's chordwise length."""
    dx = L.chord[L.strip] / NC
    return 2.0 * np.asarray(gamma, float) / dx


def _gamma(L: Lattice, dcp: np.ndarray) -> np.ndarray:
    return np.asarray(dcp, float) * (L.chord[L.strip] / NC) / 2.0


def panel_quads(L: Lattice) -> np.ndarray:
    """(N, 4, 3) panel corners for drawing: the bound leg is at the quarter of each panel's chordwise length."""
    dx = (L.chord[L.strip] / NC)[:, None]
    e = np.array([1.0, 0.0, 0.0])
    a, b = L.P1 - 0.25 * dx * e, L.P2 - 0.25 * dx * e
    return np.stack([a, b, b + dx * e, a + dx * e], axis=1)


@dataclasses.dataclass
class Case:
    """One geometry: its parameters, lattice, reference point and the solver's circulation for [1, alpha, i_t]."""
    row: dict[str, float]
    L: Lattice
    G: np.ndarray
    s_ref: float
    c_ref: float
    x_ref: float
    b_ref: float

    def dcp(self, alpha_deg: float) -> np.ndarray:
        return panel_dcp(self.L, self.G.T @ np.array([1.0, math.radians(alpha_deg), 0.0]))

    def loads(self, dcp: np.ndarray) -> dict[str, float]:
        return loads_from_circulation(self.L, _gamma(self.L, dcp), self.s_ref, self.c_ref, self.x_ref)


def _case(row: dict[str, float]) -> Case:
    af = _airframe(row)
    x_ref = af.mac_le_x + 0.25 * af.mac
    sol = solve(af, x_ref, nc=NC, ns_wing=NS_WING, ns_tail=NS_TAIL)
    return Case(dict(row), sol.L, sol.G, sol.s_ref, sol.c_ref, sol.x_ref, sol.b_ref)


def dataset(n_designs: int, n_alpha: int, seed: int = 0, bounds: dict[str, tuple[float, float]] | None = None
            ) -> tuple[list[Case], np.ndarray, np.ndarray, np.ndarray]:
    """``n_designs`` geometries by a Latin hypercube over ``bounds`` (default :data:`BOUNDS`; alpha excluded), each
    solved once and
    sampled at ``n_alpha`` angles of attack spread over the alpha range. Returns the cases, the inputs ``X``
    (n, 6) in :data:`DESIGN` order, the fields ``Y`` (n, 200) and the case index of each row."""
    from scipy.stats import qmc

    B = bounds or BOUNDS
    geo = [k for k in DESIGN if k != "alpha"]
    lo = np.array([B[k][0] for k in geo])
    hi = np.array([B[k][1] for k in geo])
    U = qmc.LatinHypercube(d=len(geo), seed=seed).random(n_designs)
    rng = np.random.default_rng(seed + 1)
    a_lo, a_hi = B["alpha"]
    cases, X, Y, idx = [], [], [], []
    for i, u in enumerate(U):
        row = dict(zip(geo, (lo + u * (hi - lo)).tolist(), strict=True))
        c = _case(row)
        cases.append(c)
        # stratified angles: one per equal slice of the range, jittered
        edges = np.linspace(a_lo, a_hi, n_alpha + 1)
        for a in edges[:-1] + rng.random(n_alpha) * (edges[1] - edges[0]):
            X.append([row[k] for k in geo] + [float(a)])
            Y.append(c.dcp(float(a)))
            idx.append(i)
    return cases, np.array(X), np.array(Y), np.array(idx)


# ------------------------------------------------------------------------------------------------- the model
def _norm_in(X: np.ndarray, bounds: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    B = bounds or BOUNDS
    lo = np.array([B[k][0] for k in DESIGN])
    hi = np.array([B[k][1] for k in DESIGN])
    return 2.0 * (np.asarray(X, float) - lo) / (hi - lo) - 1.0


class FieldEnsemble:
    """A bagged ensemble of MLPs from the six inputs to the panel field (standardised); ``predict`` returns the
    mean and the ensemble spread in field units."""

    def __init__(self, n_out: int, width: int = 160, depth: int = 3, heads: int = 5, seed: int = 0,
                 bounds: dict[str, tuple[float, float]] | None = None):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.heads = nn.ModuleList()
        for _ in range(heads):
            layers, d = [], len(DESIGN)
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.GELU()]
                d = width
            layers.append(nn.Linear(d, n_out))
            self.heads.append(nn.Sequential(*layers))
        self.mu = np.zeros(n_out)
        self.sd = np.ones(n_out)
        self.Xtrain = np.zeros((0, len(DESIGN)))
        self.bounds = dict(bounds or BOUNDS)
        self.config = {"n_out": int(n_out), "width": int(width), "depth": int(depth), "heads": int(heads),
                       "seed": int(seed)}

    def save(self, path: str) -> str:
        """Write the ensemble to one file: weights, architecture, output scaling, the declared design space and the
        training inputs (the applicability score is a distance to them). Tensors and plain types only, so
        :meth:`load` reads it with ``torch.load(weights_only=True)``."""
        import torch
        torch.save({"format": MODEL_FORMAT, "design": list(DESIGN), "config": dict(self.config),
                    "bounds": {k: (float(a), float(b)) for k, (a, b) in self.bounds.items()},
                    "mu": torch.tensor(self.mu), "sd": torch.tensor(self.sd), "Xtrain": torch.tensor(self.Xtrain),
                    "state": self.heads.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path: str) -> FieldEnsemble:
        """The ensemble :meth:`save` wrote (read with ``weights_only=True``: no code is executed from the file)."""
        import torch
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d.get("format") != MODEL_FORMAT or list(d.get("design", [])) != list(DESIGN):
            raise ValueError(f"{path} is not a {MODEL_FORMAT} file for inputs {DESIGN}")
        m = cls(**d["config"], bounds={k: tuple(v) for k, v in d["bounds"].items()})
        m.heads.load_state_dict(d["state"])
        m.mu, m.sd, m.Xtrain = d["mu"].numpy(), d["sd"].numpy(), d["Xtrain"].numpy()
        return m

    def _forward(self, Z):
        import torch
        return torch.stack([h(Z) for h in self.heads])           # (H, n, out)

    def fit(self, X, Y, *, epochs: int, lr: float = 3e-3, seed: int = 0, Xval=None, Yval=None,
            snapshots=(), on_snapshot=None) -> dict[str, list[float]]:
        """Full-batch training; at each epoch in ``snapshots`` the training and validation errors are recorded and
        ``on_snapshot(epoch, history)`` is called."""
        import torch

        g = torch.Generator().manual_seed(seed)
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-8
        self.Xtrain = np.asarray(X, float)
        Z = torch.tensor(_norm_in(X, self.bounds), dtype=torch.float32)
        T = torch.tensor((Y - self.mu) / self.sd, dtype=torch.float32)
        n, H = len(Z), len(self.heads)
        counts = torch.stack([torch.bincount(torch.randint(0, n, (n,), generator=g), minlength=n) for _ in range(H)])
        w = (counts.float() / counts.sum(1, keepdim=True))[:, :, None]           # (H, n, 1) bootstrap weights
        opt = torch.optim.Adam(self.heads.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs, pct_start=0.15)
        hist: dict[str, list[float]] = {"epoch": [], "train": [], "val": []}
        snaps = set(int(e) for e in snapshots)
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            err = (self._forward(Z) - T) ** 2                                      # (H, n, out)
            loss = (w * err.mean(2, keepdim=True)).sum() / H
            loss.backward()
            opt.step()
            sched.step()
            if ep in snaps:
                hist["epoch"].append(ep)
                hist["train"].append(self.rel_error(X, Y))
                hist["val"].append(self.rel_error(Xval, Yval) if Xval is not None else float("nan"))
                if on_snapshot is not None:
                    on_snapshot(ep, hist)
        return hist

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        import torch

        with torch.no_grad():
            P = self._forward(torch.tensor(_norm_in(np.atleast_2d(X), self.bounds), dtype=torch.float32)).numpy()
        P = P * self.sd + self.mu
        return P.mean(0), P.std(0)

    def rel_error(self, X, Y) -> float:
        m, _ = self.predict(X)
        return float(np.linalg.norm(m - Y) / max(np.linalg.norm(Y), 1e-300))

    def ood(self, x, factor: float = 2.0) -> dict[str, Any]:
        """Distance of ``x`` to the nearest training geometry against the spread of the training set (95th percentile
        of nearest-neighbour distances), and any input outside the declared design space :data:`BOUNDS` (the box the
        training designs were drawn from: the model's validity domain)."""
        Zt = _norm_in(self.Xtrain, self.bounds)
        z = _norm_in(np.atleast_2d(x), self.bounds)[0]
        geo = slice(0, len(DESIGN) - 1)                     # geometry inputs (each training geometry has many alphas)
        G = np.unique(np.round(Zt[:, geo], 9), axis=0)
        d = float(np.min(np.linalg.norm(G - z[geo], axis=1)))
        D = np.linalg.norm(G[:, None] - G[None], axis=2)
        np.fill_diagonal(D, np.inf)
        ref = float(np.quantile(D.min(1), 0.95))
        reasons = []
        score = d / max(ref, 1e-12)
        if score > factor:
            reasons.append(f"geometry {score:.2f}x farther from the training designs than usual")
        xv = np.atleast_2d(x)[0]
        B = self.bounds
        out = [k for i, k in enumerate(DESIGN) if not (B[k][0] - 1e-9 <= xv[i] <= B[k][1] + 1e-9)]
        if out:
            reasons.append("outside the declared design space: " + ", ".join(out))
        return {"ood": bool(reasons), "score": score, "reasons": reasons}


# ------------------------------------------------------------------------------------------------- drawing
def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def _airframe_mesh(row: dict[str, float], detail: str = "low") -> list[tuple[np.ndarray, np.ndarray, str]]:
    return [(p.vertices, p.faces, p.name) for p in _airframe(row).build(detail) if p.aero]


def _style(ax3, lim, azim: float = -128.0) -> None:
    ax3.set_facecolor(BG)
    ax3.set_axis_off()
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax3.set_xlim(x0, x1)
    ax3.set_ylim(y0, y1)
    ax3.set_zlim(z0, z1)
    ax3.set_box_aspect((x1 - x0, y1 - y0, (z1 - z0)), zoom=1.22)
    ax3.set_proj_type("persp", focal_length=0.45)
    ax3.view_init(elev=26, azim=azim)


def _draw_aircraft(ax3, mesh, quads, values, norm, cmap, lim, azim=-128.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for V, F, name in mesh:
        if name.startswith(("wing", "htail")):
            continue                      # the lifting surfaces are drawn as their panels, coloured by the field
        pc = Poly3DCollection(V[F], facecolor="#3a4558", edgecolor="none", alpha=0.55)
        ax3.add_collection3d(pc)
    pc = Poly3DCollection(quads + np.array([0, 0, 0.012]), facecolor=cmap(norm(values)), edgecolor="#0b0f17",
                          linewidth=0.15)
    ax3.add_collection3d(pc)
    _style(ax3, lim, azim)


def _limits(mesh) -> tuple:
    P = np.concatenate([V for V, _, _ in mesh])
    lo, hi = P.min(0), P.max(0)
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo) * 1.0
    return ((c[0] - r[0], c[0] + r[0]), (c[1] - r[1], c[1] + r[1]), (c[2] - r[2], c[2] + r[2]))


# ------------------------------------------------------------------------------------------------- the study
def aircraft_surrogate_study(out_dir: str = "aircraft_surrogate", n_train: int = 48, n_test: int = 12,
                             n_alpha: int = 6, epochs: int = 1500, frames: int = 80, fps: int = 10,
                             width_px: int = 1280, seed: int = 0, bounds: dict[str, tuple[float, float]] | None = None,
                             target: dict[str, float] | None = None, probe: dict[str, float] | None = None
                             ) -> dict[str, Any]:
    """Train a surrogate of the 3-D aircraft's aerodynamics on vortex-lattice data, compare its inferences with the
    solver and the training data, and grade it with VeriPhysics.

    Parameters
    ----------
    out_dir : str
        Folder for the video, the 3-D field view and the figures (a job runner passes its own output folder).
    n_train : int
        Aircraft geometries in the training set (each solved once, sampled at ``n_alpha`` angles).
    n_test : int
        Held-out geometries, never seen in training.
    n_alpha : int
        Angles of attack per geometry, stratified over -2..8 deg.
    epochs : int
        Training epochs of the ensemble (full batch).
    frames : int
        Frames of the training video (snapshots spaced geometrically over the epochs).
    fps : int [1/s]
        Frames per second of the video.
    width_px : int [px]
        Width of the video and figures.
    seed : int
        Seed of the designs, the split and the networks.
    bounds : dict, optional
        The design space (default :data:`BOUNDS`): the box the designs are drawn from, the model's declared validity.
    target : dict, optional
        The user's design (the five geometry parameters and ``alpha``), shown in the video and the 3-D view; by
        default the first held-out design at 6 deg.
    probe : dict, optional
        A request checked for applicability (default :data:`PROBE`, outside the default design space).

    Returns
    -------
    dict
        ``outputs`` (the measured numbers), ``checks`` (code, status, detail, value), ``decision`` (VeriPhysics
        verdict and reasons), ``files`` (names written in ``out_dir``).
    """
    return _study(out_dir, n_train, n_test, n_alpha, epochs, frames, fps, width_px, seed, bounds, target, probe)[0]


def _study(out_dir: str, n_train: int, n_test: int, n_alpha: int, epochs: int, frames: int, fps: int,
           width_px: int, seed: int, bounds: dict[str, tuple[float, float]] | None,
           target: dict[str, float] | None, probe: dict[str, float] | None,
           headline: str = "training a 3-D aircraft aerodynamics surrogate") -> tuple[dict[str, Any], Any, Any, float]:
    """:func:`aircraft_surrogate_study`, also returning the trained model and the target case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    from pinneapple_veriphysics.policy import decide, load_policy

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    files: list[str] = []
    B = dict(bounds or BOUNDS)
    PR = dict(probe or PROBE)

    # 1. data from the solver --------------------------------------------------------------------------------
    t = time.perf_counter()
    cases, X, Y, idx = dataset(n_train + n_test, n_alpha, seed, B)
    gen_s = time.perf_counter() - t
    order = np.random.default_rng(seed + 2).permutation(len(cases))
    test_ids, train_ids = set(order[:n_test].tolist()), set(order[n_test:].tolist())
    tr = np.array([i in train_ids for i in idx])
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[~tr], Y[~tr]

    # the showcase: the user's design if given, else the first held-out design, at 6 deg
    if target is not None:
        show = _case({k: float(target[k]) for k in DESIGN if k != "alpha"})
        show_a = float(target.get("alpha", 6.0))
        show_label = "your design"
    else:
        show = cases[sorted(test_ids)[0]]
        show_a = 6.0
        show_label = "held-out design (never seen in training)"
    show_x = np.array([[show.row[k] for k in DESIGN if k != "alpha"] + [show_a]])
    show_truth = show.dcp(show_a)
    mesh = _airframe_mesh(show.row)
    quads = panel_quads(show.L)
    lim = _limits(mesh)
    vmax = float(np.percentile(np.abs(Y), 99.5))
    norm = mcolors.Normalize(vmin=-0.15 * vmax, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    # 2. training, with snapshots for the video --------------------------------------------------------------
    model = FieldEnsemble(Y.shape[1], seed=seed, bounds=B)
    frame_dir = os.path.join(out_dir, "_frames")
    os.makedirs(frame_dir, exist_ok=True)
    # snapshots spaced geometrically: the field forms in the first epochs, the video should show it
    snaps = sorted(set(np.unique(np.round(np.geomspace(1, epochs, max(frames, 2))).astype(int)).tolist()) | {epochs})
    W = width_px / 100
    fig = plt.figure(figsize=(W, W * 9 / 16), dpi=100, facecolor=BG)
    snap_n = [0]
    ystrip = show.L.yc
    wing = show.L.strip_surf == 0

    def strip_load(dcp):
        g = np.bincount(show.L.strip, weights=_gamma(show.L, dcp), minlength=len(show.L.chord))
        return 2 * g / show.c_ref                                   # c_l * c / c_ref

    def on_snapshot(ep, hist):
        fig.clf()
        pred, _ = model.predict(show_x)
        pred = pred[0]
        azim = -150.0 + 32.0 * snap_n[0] / max(len(snaps) - 1, 1)
        a1 = fig.add_axes([-0.03, 0.27, 0.54, 0.62], projection="3d")
        _draw_aircraft(a1, mesh, quads, pred, norm, cmap, lim, azim)
        a2 = fig.add_axes([0.49, 0.27, 0.54, 0.62], projection="3d")
        _draw_aircraft(a2, mesh, quads, show_truth, norm, cmap, lim, azim)
        fig.text(0.02, 0.955, f"PINNeAPPle · {headline}", color=FG, fontsize=17,
                 weight="bold")
        fig.text(0.98, 0.885, f"{show_label} · α = {show_a:g}°", color=MUTED, fontsize=12, ha="right")
        lp, lt = show.loads(pred), show.loads(show_truth)
        fig.text(0.03, 0.885, f"SURROGATE · epoch {ep}", color=AMBER, fontsize=14, weight="bold")
        fig.text(0.03, 0.855, f"CL {lp['CL']:.3f}   CDi {lp['CDi']:.4f}   Cm {lp['Cm']:.3f}", color=FG, fontsize=12,
                 family="monospace")
        fig.text(0.53, 0.885, "VORTEX LATTICE · truth", color=TEAL, fontsize=14, weight="bold")
        fig.text(0.53, 0.855, f"CL {lt['CL']:.3f}   CDi {lt['CDi']:.4f}   Cm {lt['Cm']:.3f}", color=FG, fontsize=12,
                 family="monospace")
        a3 = fig.add_axes([0.06, 0.07, 0.40, 0.2], facecolor=BG)
        a3.loglog(hist["epoch"], hist["train"], color=AMBER, lw=2, label="training designs")
        a3.loglog(hist["epoch"], hist["val"], color=TEAL, lw=2, label="held-out designs")
        a3.set_xlim(1, epochs)
        a3.set_ylim(min(min(hist["train"] + hist["val"]) * 0.7, 0.02), 1.5)
        a3.set_xlabel("epoch (log scale)", color=MUTED)
        a3.set_ylabel("relative L2 error of ΔCp", color=MUTED)
        a3.legend(frameon=False, labelcolor=FG, loc="upper right")
        a4 = fig.add_axes([0.56, 0.07, 0.40, 0.2], facecolor=BG)
        a4.plot(ystrip[wing], strip_load(show_truth)[wing], color=FG, lw=2.2, label="truth")
        a4.plot(ystrip[wing], strip_load(pred)[wing], color=AMBER, lw=2, ls="--", label="surrogate")
        a4.set_xlabel("span position y (m)", color=MUTED)
        a4.set_ylabel("wing loading c_l c / c_ref", color=MUTED)
        a4.legend(frameon=False, labelcolor=FG, loc="lower center")
        for a in (a3, a4):
            a.tick_params(colors=MUTED)
            for s in a.spines.values():
                s.set_color(GRID)
            a.grid(color=GRID, lw=0.6)
        cax = fig.add_axes([0.493, 0.38, 0.011, 0.40])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label("ΔCp", color=FG)
        cb.ax.tick_params(colors=MUTED)
        fig.savefig(os.path.join(frame_dir, f"f{snap_n[0]:04d}.png"), facecolor=BG)
        snap_n[0] += 1

    t = time.perf_counter()
    hist = model.fit(Xtr, Ytr, epochs=epochs, seed=seed, Xval=Xte, Yval=Yte, snapshots=snaps,
                     on_snapshot=on_snapshot)
    train_s = time.perf_counter() - t
    plt.close(fig)
    poster = "aircraft_surrogate_poster.png"
    shutil.copy(os.path.join(frame_dir, f"f{snap_n[0] - 1:04d}.png"), os.path.join(out_dir, poster))
    files.append(poster)
    exe = _ffmpeg()
    if exe:
        import subprocess

        # hold the last frame for two seconds
        last = os.path.join(frame_dir, f"f{snap_n[0] - 1:04d}.png")
        for k in range(2 * fps):
            shutil.copy(last, os.path.join(frame_dir, f"f{snap_n[0] + k:04d}.png"))
        subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        os.path.join(frame_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart",
                        os.path.join(out_dir, "aircraft_surrogate.mp4")], check=True)
        files.insert(0, "aircraft_surrogate.mp4")
    shutil.rmtree(frame_dir, ignore_errors=True)

    # 3. inference on the held-out designs, against the solver and the training data --------------------------
    t = time.perf_counter()
    Pte, Ste = model.predict(Xte)
    infer_ms = (time.perf_counter() - t) / len(Xte) * 1e3
    Ptr, _ = model.predict(Xtr)
    rel = lambda P, T: np.linalg.norm(P - T, axis=1) / np.maximum(np.linalg.norm(T, axis=1), 1e-300)  # noqa: E731
    err_tr, err_te = rel(Ptr, Ytr), rel(Pte, Yte)
    # lookup baseline: the field of the nearest training sample in input space
    Ztr, Zte = _norm_in(Xtr, B), _norm_in(Xte, B)
    nn_i = np.argmin(np.linalg.norm(Zte[:, None] - Ztr[None], axis=2), axis=1)
    err_nn = rel(Ytr[nn_i], Yte)
    coef = {"truth": [], "pred": [], "set": []}
    for Y_, P_, name in ((Ytr, Ptr, "train"), (Yte, Pte, "test")):
        ids = idx[tr] if name == "train" else idx[~tr]
        for i, (y, p) in enumerate(zip(Y_, P_, strict=True)):
            c = cases[ids[i]]
            coef["truth"].append(c.loads(y))
            coef["pred"].append(c.loads(p))
            coef["set"].append(name)
    sets = np.array(coef["set"])
    get = lambda which, k: np.array([d[k] for d in coef[which]])  # noqa: E731
    cl_t, cl_p = get("truth", "CL"), get("pred", "CL")
    cdi_t, cdi_p = get("truth", "CDi"), get("pred", "CDi")
    cm_t, cm_p = get("truth", "Cm"), get("pred", "Cm")
    te = sets == "test"
    cl_err = float(np.max(np.abs(cl_p[te] - cl_t[te])))
    cdi_rel = float(np.median(np.abs(cdi_p[te] - cdi_t[te]) / np.maximum(np.abs(cdi_t[te]), 1e-4)))

    # 4. VeriPhysics grade --------------------------------------------------------------------------------
    pol = load_policy(None)
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        checks.append({"code": code, "status": "pass" if ok else "fail", "detail": detail, "value": value})

    field_err = float(np.linalg.norm(Pte - Yte) / np.linalg.norm(Yte))
    chk("grade.reference", field_err <= pol.reference_tol,
        f"relative L2 error of the ΔCp field on {n_test} held-out designs × {n_alpha} angles: {field_err:.3f} "
        f"(allowed {pol.reference_tol:g}); nearest-training-sample lookup: {float(np.median(err_nn)):.3f} median",
        field_err)
    test_geo = {}
    for x in Xte:                                            # one flag per held-out geometry (its angles share it)
        test_geo.setdefault(tuple(np.round(x[:-1], 9)), x)
    in_flags = [model.ood(x)["ood"] for x in test_geo.values()]
    probe_x = np.array([[PR[k] for k in DESIGN]])
    pflag = model.ood(probe_x[0])
    probe_case = _case(PR)
    probe_err = float(rel(model.predict(probe_x)[0], probe_case.dcp(PR["alpha"])[None])[0])
    probe_outside = any(not (B[k][0] - 1e-9 <= PR[k] <= B[k][1] + 1e-9) for k in DESIGN)
    # the probe must be flagged exactly when it is outside the declared space
    ok_app = (not any(in_flags)) and (pflag["ood"] or not probe_outside)
    chk("grade.applicability", ok_app,
        f"{sum(in_flags)} of {len(in_flags)} held-out geometries flagged (a flag: outside the declared design space, or "
        f"more than 2x the usual distance between training designs); the request with aspect ratio "
        f"{PR['aspect_ratio']:g} and sweep {PR['sweep_le']:g}° ({'outside' if probe_outside else 'inside'} the "
        "declared design space) " + ("flagged: " + "; ".join(pflag["reasons"]) if pflag["ood"] else "not flagged")
        + f" (its field error against the solver {probe_err:.3f})", pflag["score"])
    rng = np.random.default_rng(seed + 3)
    span = np.array([B[k][1] - B[k][0] for k in DESIGN])
    eps = 1e-3
    base = model.predict(Xte)[0]
    d1 = np.abs(model.predict(Xte + eps * span * rng.standard_normal(Xte.shape))[0] - base).mean()
    d2 = np.abs(model.predict(Xte + 0.1 * eps * span * rng.standard_normal(Xte.shape))[0] - base).mean()
    ratio = float(d1 / max(d2, 1e-300))
    chk("grade.robustness", pol.robustness_band[0] <= ratio <= pol.robustness_band[1],
        f"output change for a {eps:g}-span perturbation is {ratio:.1f}x the change for a 10x smaller one "
        "(about 10 for a smooth model)", ratio)
    chk("aero.lift_from_field", cl_err <= 0.02,
        f"CL from the predicted field (Kutta-Joukowski) vs the solver on the held-out designs: max |ΔCL| "
        f"{cl_err:.4f} (allowed 0.02)", cl_err)
    # linearity: second difference of CL over alpha relative to its first difference, held-out geometries
    lin = []
    for cid in sorted(test_ids):
        c = cases[cid]
        a = np.linspace(B["alpha"][0], B["alpha"][1], 9)
        Xa = np.array([[c.row[k] for k in DESIGN if k != "alpha"] + [ai] for ai in a])
        cl = np.array([c.loads(p)["CL"] for p in model.predict(Xa)[0]])
        d1a = np.diff(cl)
        lin.append(float(np.max(np.abs(np.diff(d1a))) / max(np.mean(np.abs(d1a)), 1e-12)))
    lin_w = float(max(lin))
    chk("aero.linearity_alpha", lin_w <= 0.05,
        f"largest second difference of CL(α) over its mean first difference: {lin_w:.3f} (allowed 0.05; the "
        "lattice is exactly linear in α)", lin_w)
    # symmetry: panels mirrored in y on the same surface
    Lc = show.L
    mirror = np.array([np.argmin(np.linalg.norm(Lc.C - Lc.C[i] * [1, -1, 1], axis=1)) for i in range(len(Lc.C))])
    sym = float(np.linalg.norm(Pte - Pte[:, mirror]) / np.linalg.norm(Pte))
    chk("aero.symmetry", sym <= 0.02, f"left-right difference of the predicted field: {sym:.4f} of its norm "
                                      "(allowed 0.02)", sym)
    sel = te & (cl_t > 0.3)
    ar_sel = np.array([cases[i].row["aspect_ratio"] for i in idx[~tr]])[cl_t[te] > 0.3]
    e_pred = (cl_p[sel] ** 2) / (math.pi * ar_sel * np.maximum(cdi_p[sel], 1e-9)) if sel.any() else np.array([0.0])
    e_max = float(np.max(e_pred))
    chk("aero.munk_bound", e_max <= 1.02,
        f"span efficiency CL²/(π AR CDi) of the predicted loads, held-out designs with CL > 0.3: max {e_max:.3f} "
        "(Munk: at most 1; allowed 1.02)", e_max)
    decision, reasons = decide(checks, pol)

    # 5. figures and the 3-D field view ---------------------------------------------------------------------
    plt.rcParams.update({"axes.facecolor": BG, "figure.facecolor": BG, "text.color": FG, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    fig, axs = plt.subplots(1, 3, figsize=(W, W * 0.34))
    for ax, (tv, pv, name) in zip(axs, ((cl_t, cl_p, "CL"), (cdi_t, cdi_p, "CDi"), (cm_t, cm_p, "Cm")), strict=True):
        for s_, col in (("train", AMBER), ("test", TEAL)):
            m = sets == s_
            ax.scatter(tv[m], pv[m], s=14, color=col, alpha=0.8, label=("training" if s_ == "train" else "held-out"))
        lo_, hi_ = float(min(tv.min(), pv.min())), float(max(tv.max(), pv.max()))
        ax.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=1)
        ax.set_xlabel(f"{name}, vortex lattice")
        ax.set_ylabel(f"{name}, surrogate")
        ax.grid(color=GRID, lw=0.6)
    axs[0].legend(frameon=False, labelcolor=FG)
    fig.suptitle("Inference vs solver: training designs and held-out designs", color=FG, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "parity.png"), dpi=100)
    plt.close(fig)
    files.append("parity.png")

    fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.3))
    bins = np.logspace(np.log10(max(min(err_tr.min(), err_te.min()) * 0.8, 1e-5)),
                       np.log10(max(err_nn.max(), err_te.max()) * 1.2), 40)
    ax.hist(err_tr, bins=bins, color=AMBER, alpha=0.75, label=f"training designs (median {np.median(err_tr):.3f})")
    ax.hist(err_te, bins=bins, color=TEAL, alpha=0.75, label=f"held-out designs (median {np.median(err_te):.3f})")
    ax.hist(err_nn, bins=bins, color=RED, alpha=0.45,
            label=f"nearest training sample as answer (median {np.median(err_nn):.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("relative L2 error of the ΔCp field (log scale)")
    ax.set_ylabel("samples")
    ax.legend(frameon=False, labelcolor=FG)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("Surrogate vs training data", color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "errors_vs_training.png"), dpi=100)
    plt.close(fig)
    files.append("errors_vs_training.png")

    fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.3))
    ax.set_axis_off()
    col = {"APPROVED": TEAL, "REVIEW": AMBER, "REJECT": RED}[decision]
    ax.text(0.0, 0.92, "VeriPhysics grade", color=MUTED, fontsize=15, transform=ax.transAxes)
    ax.text(0.0, 0.70, decision, color=col, fontsize=34, weight="bold", transform=ax.transAxes)
    for k, c in enumerate(checks):
        mark = "✔" if c["status"] == "pass" else "✘"
        ax.text(0.0, 0.52 - 0.075 * k, f"{mark}  {c['code']}", color=TEAL if c["status"] == "pass" else RED,
                fontsize=12, transform=ax.transAxes, family="monospace")
    fig.savefig(os.path.join(out_dir, "veriphysics_grade.png"), dpi=100, facecolor=BG)
    plt.close(fig)
    files.append("veriphysics_grade.png")

    try:
        from pinneapple_data.cae.model import CellBlock, Mesh
        from pinneapple_tools.visualization.studio.fieldview import field_view

        Vs, Fs, cell_vals = [], [], {k: [] for k in ("truth", "pred", "err", "std")}
        full = [(p.vertices, p.faces, p.name) for p in _airframe(show.row).build("cfd") if p.aero]
        lift = [(V, F, nme) for V, F, nme in full if nme.startswith(("wing", "htail"))]
        pred6, std6 = model.predict(show_x)
        off = 0
        for V, F, _ in lift:
            cent = V[F].mean(1)
            near = np.argmin(np.linalg.norm(cent[:, None, :2] - show.L.C[None, :, :2], axis=2)
                             + 50.0 * (np.abs(cent[:, None, 2] - show.L.C[None, :, 2]) > 0.6), axis=1)
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
            cell_vals["truth"].append(show_truth[near])
            cell_vals["pred"].append(pred6[0][near])
            cell_vals["err"].append(np.abs(pred6[0] - show_truth)[near])
            cell_vals["std"].append(std6[0][near])
        mesh3 = Mesh(points=np.concatenate(Vs), blocks=[CellBlock("triangle", np.concatenate(Fs))])
        fv = field_view(mesh3, [
            {"name": "truth", "values": np.concatenate(cell_vals["truth"]), "label": "ΔCp, vortex lattice",
             "group": "dcp", "where": "cell"},
            {"name": "pred", "values": np.concatenate(cell_vals["pred"]), "label": "ΔCp, surrogate", "group": "dcp",
             "where": "cell"},
            {"name": "err", "values": np.concatenate(cell_vals["err"]), "label": "|surrogate − lattice|",
             "kind": "error", "where": "cell"},
            {"name": "std", "values": np.concatenate(cell_vals["std"]), "label": "ensemble spread",
             "kind": "error", "where": "cell"}],
            title=f"{show_label.capitalize()} at α = {show_a:g}°: AR {show.row['aspect_ratio']:.2f}, "
                  f"sweep {show.row['sweep_le']:.1f}°",
            notes=("lifting surfaces only: the vortex lattice carries no fuselage load",), length_unit="m")
        with open(os.path.join(out_dir, "aircraft_field.glb"), "wb") as fh:
            fh.write(fv.glb())
        import json

        with open(os.path.join(out_dir, "aircraft_field.json"), "w") as fh:
            json.dump(fv.meta, fh)
        files += ["aircraft_field.glb", "aircraft_field.json"]
    except Exception as e:  # noqa: BLE001 - the view is an extra; the study stands without it
        checks.append({"code": "view.field", "status": "inconclusive", "detail": f"3-D field view not written: {e}",
                       "value": None})

    outputs = {
        "decision": decision,
        "field_error_held_out": field_err,
        "field_error_training": float(np.linalg.norm(Ptr - Ytr) / np.linalg.norm(Ytr)),
        "lookup_error_held_out_median": float(np.median(err_nn)),
        "CL_max_abs_error_held_out": cl_err,
        "CDi_median_rel_error_held_out": cdi_rel,
        "ood_probe_flagged": bool(pflag["ood"]),
        "ood_probe_field_error": probe_err,
        "target_field_error": float(rel(model.predict(show_x)[0], show_truth[None])[0]),
        "target_flagged": bool(model.ood(show_x[0])["ood"]),
        "n_train_designs": n_train, "n_test_designs": n_test, "n_alpha": n_alpha, "panels": int(Y.shape[1]),
        "solver_seconds_per_design": gen_s / len(cases),
        "surrogate_ms_per_inference": infer_ms,
        "speedup_vs_solver": (gen_s / len(cases)) / max(infer_ms / 1e3, 1e-12),
        "training_seconds": train_s,
        "total_seconds": time.perf_counter() - t0,
    }
    assumptions = ["vortex lattice of wing + horizontal tail: inviscid, small angles, no fuselage load",
                   f"design space: {', '.join(f'{k} {B[k][0]:g}–{B[k][1]:g} {UNITS[k]}' for k in DESIGN)}",
                   "the library's light aircraft (NACA 2412 section, its fuselage and tail) for everything "
                   "not in the design vector"]
    files += _write_artifacts(out_dir, model, cases, X, Y, idx, sorted(train_ids), sorted(test_ids), checks,
                              {"verdict": decision, "reasons": reasons, "policy": pol.name}, outputs, B,
                              {"n_train": n_train, "n_test": n_test, "n_alpha": n_alpha, "epochs": epochs,
                               "seed": seed, "solver_seconds": gen_s, "training_seconds": train_s}, assumptions)
    res = {"outputs": outputs, "checks": checks, "decision": {"verdict": decision, "reasons": reasons,
                                                              "policy": pol.name},
           "history": {k: [float(v) for v in hist[k]] for k in hist}, "files": files, "assumptions": assumptions}
    return res, model, show, show_a


def _library_version() -> dict[str, str]:
    import subprocess

    out = {}
    try:
        import pinneapple
        out["pinneapple"] = str(getattr(pinneapple, "__version__", ""))
    except Exception:  # noqa: BLE001 - a version is informative, never required
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _write_artifacts(out_dir, model, cases, X, Y, idx, train_ids, test_ids, checks, decision, outputs, bounds, run,
                     assumptions) -> list[str]:
    """The trained model with its model card, and the solver dataset with its data card (#1462)."""
    import json

    model.save(os.path.join(out_dir, "surrogate_model.pt"))
    ver = _library_version()
    card = {
        "model": "pinneapple.aero.aircraft_surrogate.FieldEnsemble", "format": MODEL_FORMAT, "file": "surrogate_model.pt",
        "predicts": "pressure jump dCp on each lattice panel of the wing and horizontal tail (fixed topology: "
                    f"{Y.shape[1]} panels); CL, CDi and Cm follow from it by the lattice's own post-processing",
        "inputs": [{"name": k, "unit": UNITS[k]} for k in DESIGN],
        "validated_design_space": {k: list(v) for k, v in bounds.items()},
        "architecture": dict(model.config, kind="bagged MLP ensemble (GELU), standardised outputs"),
        "applicability": "a request is refused outside the validated design space or when its geometry is more than "
                         "2x farther from the training designs than their usual spacing (FieldEnsemble.ood)",
        "decision": decision, "checks": checks,
        "measured": {k: outputs[k] for k in ("field_error_held_out", "field_error_training",
                                              "lookup_error_held_out_median", "CL_max_abs_error_held_out",
                                              "CDi_median_rel_error_held_out", "surrogate_ms_per_inference",
                                              "solver_seconds_per_design", "speedup_vs_solver")},
        "training": run, "data": "dataset.npz (see data_card.json)", "library": ver, "limits": assumptions,
        "load": "from pinneapple_design.aero.aircraft_surrogate import load_surrogate; "
                "s = load_surrogate('surrogate_model.pt'); s.predict({...})",
    }
    with open(os.path.join(out_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=1, default=float)
    geo = [k for k in DESIGN if k != "alpha"]
    np.savez_compressed(os.path.join(out_dir, "dataset.npz"), X=X, Y=Y, case=idx,
                        geometry=np.array([[c.row[k] for k in geo] for c in cases]),
                        panel_centres=np.stack([c.L.C for c in cases]),
                        train_cases=np.array(train_ids, int), test_cases=np.array(test_ids, int))
    data_card = {
        "file": "dataset.npz",
        "arrays": {"X": f"(samples, {len(DESIGN)}) inputs {list(DESIGN)}", "Y": f"(samples, {Y.shape[1]}) dCp per panel",
                   "case": "(samples,) geometry index of each sample", "geometry": f"(cases, {len(geo)}) {geo}",
                   "panel_centres": "(cases, panels, 3) panel control points [m]",
                   "train_cases": "geometries used for training", "test_cases": "held-out geometries"},
        "samples": int(len(X)), "cases": int(len(cases)), "angles_per_case": run["n_alpha"],
        "design": "Latin hypercube over the geometry bounds, angles stratified per geometry",
        "design_space": {k: list(v) for k, v in bounds.items()},
        "solver": {"method": "vortex lattice (pinneapple_design.aero.vlm.solve)", "chordwise_panels": NC,
                   "spanwise_wing": NS_WING, "spanwise_tail": NS_TAIL, "seconds_total": run["solver_seconds"]},
        "seed": run["seed"], "library": ver, "limits": assumptions[:1],
    }
    with open(os.path.join(out_dir, "data_card.json"), "w") as fh:
        json.dump(data_card, fh, indent=1, default=float)
    return ["surrogate_model.pt", "model_card.json", "dataset.npz", "data_card.json"]


class AircraftSurrogate:
    """A trained aircraft surrogate loaded from disk (:func:`load_surrogate`), with its applicability gate."""

    def __init__(self, model: FieldEnsemble, card: dict[str, Any] | None = None):
        self.model, self.card = model, card or {}

    def predict(self, design: dict[str, float], allow_outside: bool = False) -> dict[str, Any]:
        """The dCp field with its ensemble spread and CL, CDi, Cm for ``design`` (the six :data:`DESIGN` values).

        Outside the validated domain the answer is a refusal with the reasons (``accepted`` False, no numbers),
        unless ``allow_outside`` is set; the flags are returned either way."""
        missing = [k for k in DESIGN if k not in design]
        if missing:
            raise KeyError(f"design is missing {missing}; inputs are {list(DESIGN)}")
        x = np.array([[float(design[k]) for k in DESIGN]])
        flag = self.model.ood(x[0])
        if flag["ood"] and not allow_outside:
            return {"accepted": False, "reasons": flag["reasons"], "applicability": flag}
        m, s = self.model.predict(x)
        af = _airframe({k: design[k] for k in DESIGN if k != "alpha"})
        x_ref = af.mac_le_x + 0.25 * af.mac
        L = build_lattice(af, NC, NS_WING, NS_TAIL)
        loads = loads_from_circulation(L, _gamma(L, m[0]), af.wing_area, af.mac, x_ref)
        return {"accepted": not flag["ood"], "dcp": m[0], "dcp_spread": s[0], **loads, "applicability": flag}


def load_surrogate(path: str) -> AircraftSurrogate:
    """Load ``surrogate_model.pt`` (or a folder holding it) and its ``model_card.json`` when present."""
    import json

    p = os.path.join(path, "surrogate_model.pt") if os.path.isdir(path) else path
    card_p = os.path.join(os.path.dirname(os.path.abspath(p)), os.path.basename(p).replace("surrogate_model.pt",
                                                                                            "model_card.json"))
    card = json.load(open(card_p)) if os.path.isfile(card_p) else None
    return AircraftSurrogate(FieldEnsemble.load(p), card)
