"""Compact-range design: quiet-zone performance against feed gain and reflector size (and optionally frequency and
focal length), a surrogate with uncertainty, design maps, the Pareto front and the smallest reflector meeting a spec.

* :class:`Spec` -- the quiet zone asked for: diameter, amplitude and phase tolerances (+-), optional cross-polar limit;
* :func:`qz_report` -- one PO solve per design for both feed polarisations (V and H), the disk metrics of
  :func:`~.qz.qz_metrics` for each, the same metrics along the two principal cuts (horizontal ``y = y_QZ`` and
  vertical ``x = 0``), and the worst case over the polarisations against the spec;
* :func:`design_grid` / :func:`doe` -- the designs of a (gain x side [x f] [x F]) grid, solved in parallel processes
  into a :class:`~..antenna.runner.ResultTable`;
* :func:`facet_convergence`, :func:`power_balance`, :func:`agreement_887` -- the checks: the metrics against the
  facet size, the power the collimated beam carries through the quiet-zone plane against the feed power the
  reflector intercepts, and the #887 reference design solved again by this module's code path;
* :func:`fit_qz_surrogate` -- a Gaussian process per metric on the inputs that vary, with k-fold held-out RMSE and the
  coverage of its 95 % intervals (is the stated uncertainty honest);
* :func:`design_maps`, :func:`pareto_front`, :func:`pareto_figure`, :func:`cuts_figure` -- figures in the
  :mod:`pinneapple_tools.visualization.style` light theme;
* :func:`smallest_reflector` -- the smallest side for which some feed gain meets the spec by the surrogate's lower
  95 % bound, verified by PO; a design that fails PO is added to the data and the search repeats;
* :func:`run_design_study` -- all of it into a folder; :func:`from_config` builds it from a JSON-like dict (the
  example's private entry point runs a user's geometry file through it, outside the repository).

Physical-optics limits (stated with every result): PO leaves out the fringe edge currents (PTD) and multiple
reflections, uses a ``cos^q`` feed with no back radiation, blockage or chamber; the edge diffraction it does give is
that of the truncated PO current. No MoM solver is wired in this module, so the optimiser's verification is PO only.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any

import numpy as np

from .compact_range import CompactRange, evaluate, scaled
from .geometry import OffsetReflector, mesh_reflector
from .po import cosq_feed, po_currents, radiate_fields, wavenumber
from .qz import disk_grid, qz_metrics

__all__ = ["Spec", "PO_LIMITS", "REF_887", "qz_report", "design_grid", "doe", "facet_convergence", "power_balance",
           "agreement_887", "fit_qz_surrogate", "predict", "design_maps", "pareto_front", "pareto_figure",
           "cuts_figure", "smallest_reflector", "run_design_study", "generic_2ghz", "from_config", "outside"]

PO_LIMITS = ("physical optics: no fringe edge currents (PTD) and no multiple reflections; ideal cos^q feed (no back "
             "radiation, no feed or strut blockage, no chamber); the edge diffraction is that of the truncated PO "
             "current; verification by PO only (no MoM solver wired in)")

# the #887 reference design (CompactRange() defaults) as measured by #887: docs/guides/compact_range.md and
# examples/use_cases/antennas/results/compact_range/summary.json ("reference", lambda/8 mesh, lambda/6 QZ grid)
REF_887 = {"amp_pp_dB": 3.354054420125901, "phase_pp_deg": 20.250662813994495, "taper_dB": 2.261314324764561,
           "ripple_pp_dB": 2.5141311501880343, "phase_ripple_pp_deg": 12.831620248330399,
           "usable_diameter": 0.13324109244444446, "xpol_dB": -22.910026343018632}

DISK = ("amp_pp_dB", "phase_pp_deg", "taper_dB", "ripple_pp_dB", "phase_ripple_pp_deg", "usable_diameter", "xpol_dB")
CUT = ("taper_dB", "ripple_pp_dB", "phase_ripple_pp_deg")
POLS = ("V", "H")


@dataclass
class Spec:
    qz_diameter: float                      # required usable quiet-zone diameter [m]
    amp_tol_db: float = 0.5                 # amplitude within +- this [dB]
    phase_tol_deg: float = 5.0              # phase within +- this [deg]
    xpol_max_db: float | None = None        # cross-polar limit over the quiet zone [dB], None: reported, not required

    def text(self) -> str:
        x = f", cross-pol <= {self.xpol_max_db:g} dB" if self.xpol_max_db is not None else ""
        return f"{self.qz_diameter:g} m within +-{self.amp_tol_db:g} dB / +-{self.phase_tol_deg:g} deg{x}"


# ---------------------------------------------------------------------------------------------------- one design
def _cut_metrics(s: np.ndarray, E: np.ndarray, diameter: float) -> dict[str, float]:
    """1-D taper, ripple and phase ripple along a cut (``s`` positions from the centre)."""
    c = int(np.argmin(np.abs(s)))
    A = 20 * np.log10(np.abs(E) / np.abs(E[c]))
    P = np.rad2deg(np.unwrap(np.angle(E / E[c])))
    P -= P[c]
    m = np.abs(s) <= diameter / 2 * (1 + 1e-9)
    V = np.stack([np.ones(m.sum()), s[m], s[m] ** 2], 1)
    fa = V @ np.linalg.lstsq(V, A[m], rcond=None)[0]
    fp = V @ np.linalg.lstsq(V, P[m], rcond=None)[0]
    return {"taper_dB": float(np.ptp(fa)), "ripple_pp_dB": float(np.ptp(A[m] - fa)),
            "phase_ripple_pp_deg": float(np.ptp(P[m] - fp))}


def _fields(cr: CompactRange, pols: Sequence[str], points_xy=None, magnetic: bool = False):
    lam = cr.wavelength
    k = wavenumber(cr.freq_hz)
    mesh = mesh_reflector(cr.reflector, lam / cr.mesh_per_wavelength)
    feed = cr.reflector.focus + np.asarray(cr.feed_offset, float)
    axis = cr.reflector.centre - feed
    Js = []
    for p in pols:
        _, H = cosq_feed(mesh.centres, k, position=feed, axis=axis, q=cr.feed_q,
                         pol=(0.0, 1.0, 0.0) if p.upper() == "V" else (1.0, 0.0, 0.0))
        Js.append(po_currents(mesh.normals, mesh.centres, H, source=feed))
    if points_xy is None:
        points_xy = disk_grid(cr.eval_factor * cr.qz_diameter, lam / cr.qz_step_per_wavelength)
    c = cr.qz_centre
    obs = np.c_[points_xy[:, 0] + c[0], points_xy[:, 1] + c[1], np.full(len(points_xy), c[2])]
    out = radiate_fields(mesh.centres, mesh.areas, np.array(Js), obs, k, magnetic=magnetic)
    return mesh, points_xy, out


def qz_report(cr: CompactRange, spec: Spec | None = None, *, pols: Sequence[str] = POLS) -> dict[str, Any]:
    """Quiet-zone metrics of ``cr`` over a disk of ``spec.qz_diameter`` (default ``cr.qz_diameter``) for each feed
    polarisation and each principal cut, the worst case over the polarisations and whether it meets ``spec``.

    Keys: ``{pol}_{metric}`` (disk), ``{pol}_{h|v}cut_{metric}``, the worst-case ``{metric}`` (max of the "bad" ones,
    min of ``usable_diameter``), ``feasible``, the inputs, mesh size and seconds."""
    t0 = time.perf_counter()
    spec = spec or Spec(cr.qz_diameter)
    cr = replace(cr, qz_diameter=spec.qz_diameter)
    mesh, xy, E = _fields(cr, pols)
    return _summarise(cr, spec, pols, mesh, xy, E, t0)


def _summarise(cr, spec, pols, mesh, xy, E, t0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    on_h, on_v = np.abs(xy[:, 1]) < 1e-9, np.abs(xy[:, 0]) < 1e-9
    for i, p in enumerate(pols):
        co, xp = (1, 0) if p.upper() == "V" else (0, 1)
        m = qz_metrics(xy, E[i][:, co], spec.qz_diameter, E_x=E[i][:, xp], amp_tol=spec.amp_tol_db,
                       phase_tol=spec.phase_tol_deg)
        out.update({f"{p}_{k}": m[k] for k in DISK})
        out[f"{p}_usable_limited_by_region"] = m["usable_limited_by_region"]
        for cut, sel, ax in (("h", on_h, 0), ("v", on_v, 1)):
            cm = _cut_metrics(xy[sel, ax], E[i][sel, co], spec.qz_diameter)
            out.update({f"{p}_{cut}cut_{k}": v for k, v in cm.items()})
    for k in DISK:
        vals = [out[f"{p}_{k}"] for p in pols]
        out[k] = float(min(vals) if k == "usable_diameter" else max(vals))
    out["feasible"] = _meets(out, spec)
    r = cr.reflector
    out.update({"feed_gain_dBi": cr.feed_gain_dbi, "side": 2 * r.half_x, "focal": r.focal,
                "side_wl": 2 * r.half_x / cr.wavelength, "f_GHz": cr.freq_hz / 1e9, "roll": r.roll,
                "n_triangles": int(len(mesh.areas)), "n_qz_points": int(len(xy)),
                "seconds": round(time.perf_counter() - t0, 3)})
    return out


# ---------------------------------------------------------------------------------------------------- the grid
def design_grid(base: CompactRange, gains: Sequence[float], sides: Sequence[float],
                freqs_hz: Sequence[float | None] = (None,), focals: Sequence[float | None] = (None,)):
    """``[(inputs, CompactRange)]`` for every combination. ``side`` is the projected square side (the clearance
    below the section and the quiet zone on its centre line are kept, :func:`~.compact_range.scaled`); a new focal
    length moves the quiet-zone plane with the focal plane (``qz_z - F`` kept)."""
    out = []
    for f, F, s, g in product(freqs_hz, focals, sides, gains):
        cr = scaled(base, side=float(s), feed_gain_dbi=float(g))
        if f is not None:
            cr = replace(cr, freq_hz=float(f))
        if F is not None:
            cr = replace(cr, reflector=replace(cr.reflector, focal=float(F)),
                         qz_z=base.qz_z - base.reflector.focal + float(F))
        inp = {"feed_gain_dBi": float(g), "side": float(s), "f_GHz": cr.freq_hz / 1e9,
               "focal": cr.reflector.focal}
        out.append((inp, cr))
    return out


def _tag(inp: Mapping[str, float]) -> str:
    return f"g{inp['feed_gain_dBi']:g}_s{inp['side']:g}_f{inp['f_GHz']:g}_F{inp['focal']:g}"


def _solve(args):
    inp, cr, spec = args
    m = qz_report(cr, spec)
    return {"tag": _tag(inp), **inp, **{k: v for k, v in m.items() if k not in inp}}


def _order(row: dict[str, Any]) -> dict[str, Any]:
    """``tag``, the inputs, then ``f_GHz`` (ResultTable.param_names stops there) and the metrics."""
    head = ["tag", "feed_gain_dBi", "side", "focal", "f_GHz"]
    return {**{k: row[k] for k in head if k in row}, **{k: v for k, v in row.items() if k not in head}}


def _load_jsonl(path: str | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path and os.path.isfile(path):
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:          # a line cut by a kill: solved again
                        continue
                    out[r["tag"]] = r
    return out


def _append_jsonl(path: str | None, row: Mapping[str, Any]) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cut = os.path.isfile(path) and os.path.getsize(path) > 0 and open(path, "rb").read()[-1:] != b"\n"
        with open(path, "a") as fh:
            fh.write(("\n" if cut else "") + json.dumps(row, default=_json) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def doe(base: CompactRange, spec: Spec, gains: Sequence[float], sides: Sequence[float],
        freqs_hz: Sequence[float | None] = (None,), focals: Sequence[float | None] = (None,), *, workers: int = 1,
        checkpoint: str | None = None, log: Callable[[str], None] | None = None):
    """:func:`qz_report` of every design of :func:`design_grid` (``workers`` processes) as a ResultTable, in grid
    order. ``checkpoint``: a JSON-lines file each solved design is appended to as it finishes; designs already in
    it are not solved again, so a killed run resumes where it stopped."""
    from ..antenna.runner import ResultTable
    jobs = [(inp, cr, spec) for inp, cr in design_grid(base, gains, sides, freqs_hz, focals)]
    done = _load_jsonl(checkpoint)
    todo = [j for j in jobs if _tag(j[0]) not in done]
    if log and done:
        log(f"checkpoint {checkpoint}: {len(jobs) - len(todo)} of {len(jobs)} designs already solved")

    def keep(r):
        r = _order(r)
        done[r["tag"]] = r
        _append_jsonl(checkpoint, r)
        if log:
            log(_line(r))
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(_solve, todo):
                keep(r)
    else:
        for j in todo:
            keep(_solve(j))
    return ResultTable([done[_tag(j[0])] for j in jobs])


def _line(r) -> str:
    return (f"{r['tag']}: usable {r['usable_diameter']:.2f} m, ripple {r['ripple_pp_dB']:.2f} dB, taper "
            f"{r['taper_dB']:.2f} dB, phase ripple {r['phase_ripple_pp_deg']:.1f} deg, xpol {r['xpol_dB']:.1f} dB "
            f"({r['seconds']:.1f} s)")


# ---------------------------------------------------------------------------------------------------- checks
def facet_convergence(cr: CompactRange, spec: Spec | None = None, densities: Sequence[float] = (6, 8, 12),
                      metrics: Sequence[str] = DISK) -> dict[str, Any]:
    """The worst-case metrics at triangles of lambda/``m`` for each ``m``, and each one's difference from the
    finest."""
    runs = [{"per_wavelength": float(m), **qz_report(replace(cr, mesh_per_wavelength=float(m)), spec)}
            for m in densities]
    fine = runs[-1]
    diff = [{"per_wavelength": r["per_wavelength"], **{k: float(r[k] - fine[k]) for k in metrics}} for r in runs]
    return {"runs": [{k: r[k] for k in ("per_wavelength", "n_triangles", "seconds", *metrics)} for r in runs],
            "diff_to_finest": diff}


def power_balance(cr: CompactRange, *, pol: str = "V", plane_factors: Sequence[float] = (1.0, 1.25, 1.5),
                  step_per_wavelength: float = 2.0, plane_offset_wl: float = 4.0) -> dict[str, Any]:
    """Energy check: the feed power intercepted by the lit reflector (flux of the incident Poynting vector through
    the triangles; the feed radiates 1 W) against the power the PO field carries through a plane ``z = const``
    ``plane_offset_wl`` wavelengths in front of the reflector (exact near field, E and H), over disks of
    ``plane_factors`` x (the larger projected size + 2 roll) around the section centre line, where the beam still
    has the reflector's width. PO on a large smooth reflector reflects what it intercepts, so the ratio tends to 1
    as the disk grows; what is missing is radiated outside the disk (edge diffraction), plus the error of a
    lambda/``step_per_wavelength`` grid. ``ratio`` is the largest disk's."""
    k = wavenumber(cr.freq_hz)
    lam = cr.wavelength
    r = cr.reflector
    mesh = mesh_reflector(r, lam / cr.mesh_per_wavelength)
    feed = r.focus + np.asarray(cr.feed_offset, float)
    E_i, H_i = cosq_feed(mesh.centres, k, position=feed, axis=r.centre - feed, q=cr.feed_q,
                         pol=(0.0, 1.0, 0.0) if pol.upper() == "V" else (1.0, 0.0, 0.0))
    S = 0.5 * np.real(np.cross(E_i, np.conj(H_i)))
    lit = np.einsum("ij,ij->i", mesh.normals, feed - mesh.centres) > 0
    p_int = float(np.sum(-np.einsum("ij,ij->i", S, mesh.normals)[lit] * mesh.areas[lit]))
    J = po_currents(mesh.normals, mesh.centres, H_i, source=feed)
    size = 2 * max(r.half_x, r.half_y) + 2 * r.roll
    h = lam / step_per_wavelength
    xy = disk_grid(max(plane_factors) * size, h)
    c = np.array([0.0, r.centre_y, mesh.vertices[:, 2].max() + plane_offset_wl * lam])
    obs = np.c_[xy[:, 0] + c[0], xy[:, 1] + c[1], np.full(len(xy), c[2])]
    E, H = radiate_fields(mesh.centres, mesh.areas, J[None], obs, k, magnetic=True)
    Sz = 0.5 * np.real(np.cross(E[0], np.conj(H[0])))[:, 2]
    rad = np.hypot(xy[:, 0], xy[:, 1])
    by = [{"plane_factor": float(f), "diameter_m": float(f * size),
           "through_plane_W": float(Sz[rad <= f * size / 2 * (1 + 1e-9)].sum() * h * h)} for f in plane_factors]
    for b in by:
        b["ratio"] = b["through_plane_W"] / p_int
    return {"feed_W": 1.0, "intercepted_W": p_int, "by_diameter": by, "ratio": by[-1]["ratio"],
            "plane_z_m": float(c[2]), "step_m": float(h), "n_points": int(len(xy))}


def agreement_887(*, mesh_per_wavelength: float | None = None) -> dict[str, Any]:
    """The #887 reference design (``CompactRange()``) solved by :func:`qz_report` (V polarisation, the #887 one) and
    by #887's own :func:`~.compact_range.evaluate`, against the values #887 measured (:data:`REF_887`)."""
    cr = CompactRange() if mesh_per_wavelength is None else CompactRange(mesh_per_wavelength=mesh_per_wavelength)
    new = qz_report(cr, Spec(cr.qz_diameter), pols=("V",))
    old = evaluate(cr)
    rows = []
    for k_, ref in REF_887.items():
        rows.append({"metric": k_, "887": ref, "evaluate": float(old[k_]), "qz_report": float(new[f"V_{k_}"]),
                     "abs_diff_vs_887": float(abs(new[f"V_{k_}"] - ref)),
                     "abs_diff_report_vs_evaluate": float(abs(new[f"V_{k_}"] - old[k_]))})
    return {"rows": rows, "max_abs_diff_vs_887": max(r["abs_diff_vs_887"] for r in rows),
            "max_abs_diff_report_vs_evaluate": max(r["abs_diff_report_vs_evaluate"] for r in rows),
            "mesh_per_wavelength": cr.mesh_per_wavelength}


# ---------------------------------------------------------------------------------------------------- surrogate
def _inputs(table) -> list[str]:
    return [c for c in ("feed_gain_dBi", "side", "f_GHz", "focal") if np.ptp(table.col(c)) > 0]


def fit_qz_surrogate(table, metrics: Sequence[str] = ("usable_diameter", "ripple_pp_dB", "taper_dB",
                                                      "phase_ripple_pp_deg", "xpol_dB"), *, folds: int = 5,
                     seed: int = 0) -> dict[str, Any]:
    """A GP per metric on the inputs that vary (``feed_gain_dBi``, ``side``, and ``f_GHz`` / ``focal`` when swept),
    fitted on all rows; and ``folds``-fold held-out predictions (each design predicted by a GP that never saw it):
    RMSE and the fraction inside mean +- 1.96 std (``coverage95``; 0.95 when the uncertainty is calibrated).
    Returns ``{"inputs", "metrics": {m: {"model", "rmse", "coverage95", "n"}}}``."""
    from ..antenna import surrogates as sg
    names = _inputs(table)
    feats = sg.columns(*names)
    tags = np.array(table.tags)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(tags)) % max(2, min(folds, len(tags)))
    out: dict[str, Any] = {"inputs": names, "metrics": {}}
    for m in metrics:
        def make():
            return sg.GPModel(prior="const", features=feats)
        err, std = [], []
        for f in np.unique(fold):
            v = sg.holdout(table, m, list(tags[fold == f]), models={"gp": make})
            err += list(v.errors("gp"))
            std += [r["std_gp"] for r in v.rows]
        err, std = np.asarray(err), np.asarray(std)
        out["metrics"][m] = {"model": make().fit(table, m), "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "coverage95": float(np.mean(np.abs(err) <= 1.96 * std)), "n": int(len(err))}
    return out


def predict(sur: Mapping[str, Any], metric: str, points: Sequence[Mapping[str, float]]):
    """Mean and std of ``metric`` at ``points`` (dicts of the surrogate's inputs; ``f_GHz`` defaults to the
    data's)."""
    model = sur["metrics"][metric]["model"]
    f = np.array([p.get("f_GHz", 0.0) for p in points], float)
    return model.predict(list(points), f)


# ---------------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import LIGHT_RC, get_cmap
    return plt, matplotlib.rc_context(LIGHT_RC), get_cmap("rainbow")


_LABEL = {"usable_diameter": "usable QZ diameter [m]", "ripple_pp_dB": "amplitude ripple p-p [dB]",
          "taper_dB": "amplitude taper [dB]", "phase_ripple_pp_deg": "phase ripple p-p [deg]",
          "xpol_dB": "cross-polar level [dB]", "phase_pp_deg": "phase p-p [deg]", "amp_pp_dB": "amplitude p-p [dB]"}


def design_maps(table, sur: Mapping[str, Any], spec: Spec, path: str, *, fixed: Mapping[str, float] | None = None,
                title: str = "") -> str:
    """Contours over (feed gain, side) of every surrogate metric (worst case over V and H), the predictive std of
    the usable diameter, the simulated designs, the spec's diameter as a white line and the region where the
    surrogate's lower 95 % bound meets the spec hatched. ``fixed``: values of the other inputs (default: the
    first)."""
    plt, rc, cmap = _style()
    fixed = dict(fixed or {})
    for c in sur["inputs"]:
        if c not in ("feed_gain_dBi", "side"):
            fixed.setdefault(c, float(np.unique(table.col(c))[0]))
    sel = table.where(lambda r: all(abs(float(r[c]) - v) < 1e-9 for c, v in fixed.items()))
    g, s = table.col("feed_gain_dBi"), table.col("side")
    G, S = np.meshgrid(np.linspace(g.min(), g.max(), 80), np.linspace(s.min(), s.max(), 80))
    f0 = float(table.col("f_GHz")[0])
    P = [{"feed_gain_dBi": a, "side": b, "f_GHz": f0, **fixed} for a, b in zip(G.ravel(), S.ravel(), strict=True)]
    U, Us = (x.reshape(G.shape) for x in predict(sur, "usable_diameter", P))
    ok = U - 1.96 * Us >= spec.qz_diameter
    mets = list(sur["metrics"])
    with rc:
        n = len(mets) + 1
        cols = 3
        rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(5.2 * cols, 4.3 * rows_), squeeze=False)
        panels = [(m, predict(sur, m, P)[0].reshape(G.shape), _LABEL.get(m, m),
                   sur["metrics"][m]) for m in mets] + [("std", Us, "std of usable QZ diameter [m]",
                                                         sur["metrics"]["usable_diameter"])]
        for ax, (m, Z, lab, info) in zip(axes.ravel(), panels, strict=False):
            cf = ax.contourf(G, S, Z, levels=16, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=lab)
            if U.min() < spec.qz_diameter < U.max():
                ax.contour(G, S, U, levels=[spec.qz_diameter], colors="white", linewidths=2)
            if ok.any():
                ax.contourf(G, S, ok.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
            ax.scatter(sel.col("feed_gain_dBi"), sel.col("side"), c="k", s=10, label="PO designs")
            ax.set_xlabel("feed gain [dBi]")
            ax.set_ylabel("reflector side [m]")
            t = f"held-out RMSE {info['rmse']:.3g}, 95% cover {info['coverage95']:.2f}" if m != "std" else \
                "GP predictive std"
            ax.set_title(f"{_LABEL.get(m, m) if m != 'std' else 'uncertainty'}\n{t}", fontsize=9)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, loc="upper left")
        extra = ", ".join(f"{k} = {v:g}" for k, v in fixed.items())
        fig.suptitle(title or f"design maps (worst of V/H); spec {spec.text()}; hatched: lower 95% bound meets it"
                     + (f"; {extra}" if extra else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return path


def pareto_front(table, spec: Spec | None = None) -> list[dict[str, Any]]:
    """The PO designs no other design beats on both counts (smaller or equal side and larger or equal usable QZ
    diameter, one strictly), by increasing side; each with ``meets_spec``."""
    rows = sorted(table.rows, key=lambda r: (float(r["side"]), -float(r["usable_diameter"])))
    front, best = [], -np.inf
    for r in rows:
        u = float(r["usable_diameter"])
        if u > best + 1e-12:
            front.append({k: r[k] for k in ("tag", "feed_gain_dBi", "side", "f_GHz", "focal", "usable_diameter",
                                            "ripple_pp_dB", "xpol_dB") if k in r})
            best = u
    if spec is not None:
        for f in front:
            f["meets_spec"] = bool(f["usable_diameter"] >= spec.qz_diameter)
    return front


def pareto_figure(table, sur: Mapping[str, Any] | None, spec: Spec, path: str, *, gains_dense=None,
                  chosen: Mapping[str, Any] | None = None) -> str:
    """Usable QZ diameter against reflector side: every PO design (colour = feed gain), the PO Pareto front, the
    surrogate's best over feed gain with its 95 % band, the spec line, the feasible region and the chosen design."""
    plt, rc, cmap = _style()
    front = pareto_front(table, spec)
    with rc:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        sc = ax.scatter(table.col("side"), table.col("usable_diameter"), c=table.col("feed_gain_dBi"), cmap=cmap,
                        s=28, edgecolors="k", linewidths=0.3, label="PO designs")
        fig.colorbar(sc, ax=ax, label="feed gain [dBi]")
        ax.step([f["side"] for f in front], [f["usable_diameter"] for f in front], where="post", c="k", lw=1.6,
                label="Pareto front (PO)")
        if sur is not None:
            sides = np.linspace(table.col("side").min(), table.col("side").max(), 120)
            gd = np.linspace(table.col("feed_gain_dBi").min(), table.col("feed_gain_dBi").max(), 41) \
                if gains_dense is None else np.asarray(gains_dense, float)
            f0 = float(table.col("f_GHz")[0])
            fixed = {c: float(np.unique(table.col(c))[0]) for c in sur["inputs"] if c not in ("feed_gain_dBi", "side")}
            P = [{"feed_gain_dBi": g, "side": s, "f_GHz": f0, **fixed} for s in sides for g in gd]
            mu, sd = (x.reshape(len(sides), len(gd)) for x in predict(sur, "usable_diameter", P))
            j = np.argmax(mu, axis=1)
            i = np.arange(len(sides))
            ax.plot(sides, mu[i, j], c="tab:blue", lw=1.4, label="surrogate, best feed gain")
            ax.fill_between(sides, mu[i, j] - 1.96 * sd[i, j], mu[i, j] + 1.96 * sd[i, j], color="tab:blue",
                            alpha=0.18, label="95% band")
        ax.axhline(spec.qz_diameter, c="tab:red", ls="--", lw=1.2, label=f"spec {spec.qz_diameter:g} m")
        ax.axhspan(spec.qz_diameter, max(spec.qz_diameter, table.col("usable_diameter").max()) * 1.1,
                   color="tab:green", alpha=0.08, label="meets the diameter")
        if chosen:
            ax.scatter([chosen["side"]], [chosen["usable_diameter"]], marker="*", s=300, c="white", edgecolors="k",
                       zorder=6, label=f"smallest verified ({chosen['feed_gain_dBi']:g} dBi, {chosen['side']:g} m)")
        ax.set_xlabel("reflector side [m]")
        ax.set_ylabel("usable QZ diameter [m] (worst of V/H)")
        ax.set_title(f"QZ size vs reflector size; spec {spec.text()}", fontsize=9)
        ax.grid(alpha=0.4)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return path


def cuts_figure(cr: CompactRange, spec: Spec, path: str, *, title: str = "") -> dict[str, Any]:
    """Amplitude and phase (re centre) along the horizontal and vertical cuts for both polarisations, with the
    spec's diameter and tolerance band; returns :func:`qz_report`."""
    plt, rc, _ = _style()
    t0 = time.perf_counter()
    crq = replace(cr, qz_diameter=spec.qz_diameter)
    mesh, xy, E = _fields(crq, POLS)
    rep = _summarise(crq, spec, POLS, mesh, xy, E, t0)
    with rc:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for i, p in enumerate(POLS):
            co = 1 if p == "V" else 0
            for cut, sel, ax_ in (("horizontal", np.abs(xy[:, 1]) < 1e-9, 0), ("vertical", np.abs(xy[:, 0]) < 1e-9, 1)):
                s = xy[sel, ax_]
                o = np.argsort(s)
                e = E[i][sel, co][o]
                c = int(np.argmin(np.abs(s[o])))
                A = 20 * np.log10(np.abs(e) / np.abs(e[c]))
                Ph = np.rad2deg(np.unwrap(np.angle(e / e[c])))
                Ph -= Ph[c]
                ls = "-" if cut == "horizontal" else "--"
                axes[0].plot(s[o], A, ls, label=f"{p} pol, {cut} cut")
                axes[1].plot(s[o], Ph, ls, label=f"{p} pol, {cut} cut")
        R = spec.qz_diameter / 2
        for ax, tol, lab in ((axes[0], spec.amp_tol_db, "amplitude [dB re centre]"),
                             (axes[1], spec.phase_tol_deg, "phase [deg re centre]")):
            ax.axvspan(-R, R, color="0.85", alpha=0.6)
            ax.axhspan(-tol, tol, color="tab:green", alpha=0.12)
            ax.set_xlabel("position from the QZ centre [m]")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.4)
            ax.legend(fontsize=7)
        axes[0].set_ylim(max(-6.0, axes[0].get_ylim()[0]), min(3.0, axes[0].get_ylim()[1]))
        fig.suptitle(title or f"QZ cuts, {cr.feed_gain_dbi:g} dBi feed, {2 * cr.reflector.half_x:g} m reflector, "
                              f"{cr.freq_hz / 1e9:g} GHz: usable {rep['usable_diameter']:.2f} m (worst of V/H), "
                              f"xpol {rep['xpol_dB']:.1f} dB", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return rep


# ---------------------------------------------------------------------------------------------------- optimiser
def smallest_reflector(base: CompactRange, spec: Spec, table, *, gain_bounds: tuple[float, float],
                       side_bounds: tuple[float, float], max_iter: int = 4, z95: float = 1.96, n_gain: int = 41,
                       n_side: int = 81, checkpoint: str | None = None,
                       log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The smallest side in ``side_bounds`` for which a feed gain in ``gain_bounds`` meets ``spec`` by the
    surrogate's lower 95 % bound (``mean - z95 std`` of the usable diameter, and ``mean + z95 std`` of the
    cross-polar level when the spec has one), verified by PO (:func:`qz_report`). A candidate PO rejects is added to
    ``table`` and the GP refitted; when no point meets the bound, the most promising one by the mean is solved
    instead (that is how the data grows where it matters). Stops at the first PO-verified design or after
    ``max_iter`` PO solves. Frequency and focal length stay at ``base``'s. ``checkpoint``: a JSON-lines file of
    the PO verifications (a resumed run reuses them; the search is deterministic, so it asks for the same ones)."""
    from ..antenna.runner import ResultTable
    rows = list(table.rows)
    hist = []
    gains = np.linspace(*gain_bounds, n_gain)
    sides = np.linspace(*side_bounds, n_side)
    f0, F0 = base.freq_hz / 1e9, base.reflector.focal
    sub = lambda t: t.where(lambda r: abs(float(r["f_GHz"]) - f0) < 1e-9 and abs(float(r["focal"]) - F0) < 1e-9)  # noqa: E731
    for it in range(max_iter):
        t = sub(ResultTable(rows))
        mets = ("usable_diameter",) + (("xpol_dB",) if spec.xpol_max_db is not None else ())
        sur = fit_qz_surrogate(t, mets)
        P = [{"feed_gain_dBi": g, "side": s, "f_GHz": f0, "focal": F0} for s in sides for g in gains]
        mu, sd = (x.reshape(len(sides), len(gains)) for x in predict(sur, "usable_diameter", P))
        lo = mu - z95 * sd
        good = lo >= spec.qz_diameter
        if spec.xpol_max_db is not None:
            xm, xs = (x.reshape(mu.shape) for x in predict(sur, "xpol_dB", P))
            good &= xm + z95 * xs <= spec.xpol_max_db
        tried = np.zeros(mu.shape, bool)
        for r in t.rows:
            ii = np.flatnonzero(np.isclose(sides, float(r["side"])))
            jj = np.flatnonzero(np.isclose(gains, float(r["feed_gain_dBi"])))
            tried[np.ix_(ii, jj)] = True
        good &= ~tried
        if good.any():
            i = int(np.flatnonzero(good.any(1))[0])
            j = int(np.argmax(np.where(good[i], lo[i], -np.inf)))
            how = "lower 95% bound meets the spec"
        elif (mean_ok := (mu >= spec.qz_diameter) & ~tried).any():
            i = int(np.flatnonzero(mean_ok.any(1))[0])
            j = int(np.argmax(np.where(mean_ok[i], mu[i], -np.inf)))
            how = "mean meets the spec (bound does not): exploring"
        else:
            i, j = np.unravel_index(int(np.argmax(np.where(tried, -np.inf, mu + z95 * sd))), mu.shape)
            how = "no point meets the spec by the mean: solving the largest upper bound"
        cand = (gains[j], sides[i], i, j)
        g, s, i, j = cand
        inp = {"feed_gain_dBi": float(g), "side": float(s), "f_GHz": f0, "focal": F0}
        cached = _load_jsonl(checkpoint).get(_tag(inp))
        if cached is None:
            rep = qz_report(scaled(base, side=float(s), feed_gain_dbi=float(g)), spec)
            row = _order({"tag": _tag(inp), **inp, **{k: v for k, v in rep.items() if k not in inp}})
            _append_jsonl(checkpoint, row)
        else:
            row = rep = cached
        rows.append(row)
        h = {"iteration": it + 1, "feed_gain_dBi": float(g), "side": float(s), "how": how,
             "predicted_usable": float(mu[i, j]), "predicted_std": float(sd[i, j]),
             "po_usable": float(rep["usable_diameter"]), "po_xpol_dB": float(rep["xpol_dB"]),
             "po_feasible": _meets(rep, spec)}
        hist.append(h)
        if log:
            log(f"optimiser {it + 1}: {g:.2f} dBi, {s:.3f} m ({how}): predicted {mu[i, j]:.2f} +- "
                f"{1.96 * sd[i, j]:.2f} m, PO {rep['usable_diameter']:.2f} m -> "
                f"{'meets' if h['po_feasible'] else 'fails'} the spec")
        if _meets(rep, spec):
            break
    po_ok = [r for r in sub(ResultTable(rows)).rows if _meets(r, spec)]
    best = min(po_ok, key=lambda r: float(r["side"])) if po_ok else None
    found = bool(hist) and hist[-1]["po_feasible"]
    return {"found": found, "design": {**inp, **{k: rep[k] for k in DISK}} if found else None,
            "report": rep if found else None, "history": hist, "table": ResultTable(rows),
            "smallest_po_feasible_in_data": {k: best[k] for k in ("tag", "feed_gain_dBi", "side", *DISK)} if best
            else None, "spec": asdict(spec), "verified_by": "PO", "limits": PO_LIMITS}


def _meets(row: Mapping[str, Any], spec: Spec) -> bool:
    """``row`` (a :func:`qz_report` or a sweep row) against ``spec``."""
    ok = float(row["usable_diameter"]) >= spec.qz_diameter * (1 - 1e-9)
    if spec.xpol_max_db is not None:
        ok &= float(row["xpol_dB"]) <= spec.xpol_max_db
    return bool(ok)


# ---------------------------------------------------------------------------------------------------- study
def run_design_study(base: CompactRange, spec: Spec, gains: Sequence[float], sides: Sequence[float], out: str, *,
                     freqs_hz: Sequence[float | None] = (None,), focals: Sequence[float | None] = (None,),
                     gain_bounds: tuple[float, float] | None = None, side_bounds: tuple[float, float] | None = None,
                     checks: bool = True, densities: Sequence[float] = (6, 8, 12), max_iter: int = 4,
                     workers: int = 1, table=None, extra: Mapping[str, Any] | None = None,
                     compare_887: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The sweep (or ``table``), the checks (facet convergence and power balance on the middle design), the
    surrogate, the design maps, the Pareto front, the optimiser, the QZ cuts of the chosen design, into ``out``:
    ``design_sweep.csv``, ``design_summary.json`` (with ``extra``), ``design_maps.png``, ``pareto.png``,
    ``qz_cuts.png``. Every PO solve and check is checkpointed in ``out/checkpoints``: run it again after a kill
    and it resumes (delete the folder to start over). ``compare_887``: add :func:`agreement_887`."""
    from ..antenna.runner import write_table
    os.makedirs(out, exist_ok=True)
    ck = os.path.join(out, "checkpoints")
    os.makedirs(ck, exist_ok=True)

    def stage(name, fn):
        path = os.path.join(ck, f"{name}.json")
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh)
        v = fn()
        with open(path + ".tmp", "w") as fh:
            json.dump(v, fh, default=_json)
        os.replace(path + ".tmp", path)
        return json.loads(json.dumps(v, default=_json))
    t0 = time.perf_counter()
    extra = dict(extra or {})
    if compare_887:
        extra["agreement_887"] = stage("agreement_887", agreement_887)
    if table is None:
        table = doe(base, spec, gains, sides, freqs_hz, focals, workers=workers,
                    checkpoint=os.path.join(ck, "sweep.jsonl"), log=log)
    write_table(table, os.path.join(out, "design_sweep.csv"))
    summary: dict[str, Any] = {"spec": asdict(spec), "base": base.to_dict(), "limits": PO_LIMITS,
                               "grid": {"gains_dBi": list(map(float, gains)), "sides_m": list(map(float, sides)),
                                        "freqs_hz": [f for f in freqs_hz if f is not None],
                                        "focals_m": [F for F in focals if F is not None]},
                               "n_designs": len(table), "seconds_median": float(np.median(table.col("seconds"))),
                               "seconds_sweep": float(np.sum(table.col("seconds")))}
    mid = scaled(base, side=float(sorted(sides)[len(sides) // 2]), feed_gain_dbi=float(sorted(gains)[len(gains) // 2]))
    if checks:
        summary["facet_convergence"] = stage("facet_convergence", lambda: facet_convergence(mid, spec, densities))

        def grid_check():
            a = next(r for r in summary["facet_convergence"]["runs"]
                     if abs(r["per_wavelength"] - mid.mesh_per_wavelength) < 1e-9) \
                if mid.mesh_per_wavelength in densities else qz_report(mid, spec)
            b = qz_report(replace(mid, qz_step_per_wavelength=1.5 * mid.qz_step_per_wavelength), spec)
            return {"steps_per_wavelength": [mid.qz_step_per_wavelength, 1.5 * mid.qz_step_per_wavelength],
                    **{k: [float(a[k]), float(b[k])] for k in DISK}}
        summary["qz_grid_check"] = stage("qz_grid_check", grid_check)
        summary["power_balance"] = stage("power_balance", lambda: power_balance(mid))
        log(f"facet convergence: {summary['facet_convergence']['diff_to_finest']}")
        log(f"power balance: {summary['power_balance']['ratio']:.4f} of the intercepted power crosses the plane")
    base_t = table.where(lambda r: abs(float(r["f_GHz"]) - base.freq_hz / 1e9) < 1e-9
                         and abs(float(r["focal"]) - base.reflector.focal) < 1e-9)
    sur = fit_qz_surrogate(table)
    summary["surrogate"] = {"inputs": sur["inputs"], **{m: {k: v for k, v in d.items() if k != "model"}
                                                        for m, d in sur["metrics"].items()}}
    sur_b = fit_qz_surrogate(base_t) if len(base_t) != len(table) else sur
    design_maps(base_t, sur_b, spec, os.path.join(out, "design_maps.png"))
    summary["pareto_front"] = pareto_front(base_t, spec)
    opt = smallest_reflector(base, spec, base_t, gain_bounds=gain_bounds or (min(gains), max(gains)),
                             side_bounds=side_bounds or (min(sides), max(sides)), max_iter=max_iter,
                             checkpoint=os.path.join(ck, "optimiser.jsonl"), log=log)
    write_table(opt["table"], os.path.join(out, "design_sweep_with_optimiser.csv"))
    summary["optimiser"] = {k: v for k, v in opt.items() if k not in ("table", "report")}
    chosen = opt["design"] or opt.get("smallest_po_feasible_in_data")
    sur_o = fit_qz_surrogate(opt["table"], ("usable_diameter",))
    pareto_figure(opt["table"], sur_o, spec, os.path.join(out, "pareto.png"), chosen=chosen)
    pick = chosen or max(base_t.rows, key=lambda r: float(r["usable_diameter"]))
    summary["cuts_design"] = {"feed_gain_dBi": float(pick["feed_gain_dBi"]), "side": float(pick["side"])}
    summary["cuts"] = cuts_figure(scaled(base, side=float(pick["side"]), feed_gain_dbi=float(pick["feed_gain_dBi"])),
                                  spec, os.path.join(out, "qz_cuts.png"))
    summary["checkpoints"] = os.path.relpath(ck, out)
    summary["seconds_total"] = round(time.perf_counter() - t0, 1)
    summary.update(extra)
    with open(os.path.join(out, "design_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json)
    return {"table": table, "surrogate": sur, "optimiser": opt, "summary": summary}


def _json(v):
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


# ---------------------------------------------------------------------------------------------------- configs
def generic_2ghz(quick: bool = False) -> dict[str, Any]:
    """The configuration of the public example: a generic 2 GHz range (lambda = 0.15 m), not any installation. A
    square offset section (superellipse n = 6) 0.4 m above the axis, F = 4 m (F/D about 1 at 4 m), a 0.45 m
    (3 lambda) rolled edge, the quiet zone 3 m in front of the focal plane; spec: a 1 m quiet zone within +-0.5 dB /
    +-5 deg with cross-polar <= -25 dB. ``quick``: 3 x 3 designs, coarse mesh, for a smoke run."""
    cfg = {
        "name": "generic 2 GHz compact range",
        "freq_hz": 2e9,
        "reflector": {"focal": 4.0, "half_x": 1.6, "half_y": 1.6, "centre_y": 2.0, "n": 6.0, "roll": 0.45},
        "feed_gain_dbi": 16.0, "qz_z": 7.0, "qz_diameter": 1.0,
        "mesh_per_wavelength": 8.0, "qz_step_per_wavelength": 4.0, "eval_factor": 1.6,
        "spec": {"qz_diameter": 1.0, "amp_tol_db": 0.5, "phase_tol_deg": 5.0, "xpol_max_db": -25.0},
        "sweep": {"gains_dBi": [10.0, 13.0, 16.0, 19.0, 22.0], "sides_m": [2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8],
                  "focals_m": [3.0, 4.0, 5.0]},
        "optimise": {"gain_bounds": [10.0, 22.0], "side_bounds": [2.4, 4.8], "max_iter": 4},
        "densities": [6, 8, 12], "workers": 4,
    }
    if quick:
        cfg.update({"mesh_per_wavelength": 6.0, "qz_step_per_wavelength": 3.0, "densities": [5, 6],
                    "sweep": {"gains_dBi": [12.0, 16.0, 20.0], "sides_m": [2.8, 3.6, 4.4]},
                    "optimise": {"gain_bounds": [12.0, 20.0], "side_bounds": [2.8, 4.4], "max_iter": 2}})
    return cfg


def from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``{"base": CompactRange, "spec": Spec, "grid": {...}, "optimise": {...}}`` from a dict with the
    :class:`~.compact_range.CompactRange` fields (``reflector`` as the :class:`~.geometry.OffsetReflector` fields),
    ``spec`` (the :class:`Spec` fields), ``sweep`` (``gains_dBi``, ``sides_m``, optional ``freqs_hz``,
    ``focals_m``) and optional ``optimise`` (``gain_bounds``, ``side_bounds``, ``max_iter``) and ``checks``."""
    known = set(CompactRange.__dataclass_fields__)
    bad = set(cfg) - known - {"spec", "sweep", "optimise", "checks", "name", "workers", "densities"}
    if bad:
        raise ValueError(f"unknown keys in the configuration: {sorted(bad)}")
    kw = {k: v for k, v in cfg.items() if k in known and k != "reflector"}
    if "feed_offset" in kw:
        kw["feed_offset"] = tuple(kw["feed_offset"])
    base = CompactRange(reflector=OffsetReflector(**cfg["reflector"]), **kw)
    spec = Spec(**cfg.get("spec", {"qz_diameter": base.qz_diameter}))
    sw = cfg.get("sweep", {})
    grid = {"gains": [float(g) for g in sw.get("gains_dBi", [base.feed_gain_dbi])],
            "sides": [float(s) for s in sw.get("sides_m", [2 * base.reflector.half_x])],
            "freqs_hz": [float(f) for f in sw.get("freqs_hz", [])] or [None],
            "focals": [float(F) for F in sw.get("focals_m", [])] or [None]}
    op = cfg.get("optimise", {})
    return {"base": base, "spec": spec, "grid": grid,
            "optimise": {"gain_bounds": tuple(op["gain_bounds"]) if "gain_bounds" in op else None,
                         "side_bounds": tuple(op["side_bounds"]) if "side_bounds" in op else None,
                         "max_iter": int(op.get("max_iter", 4))},
            "checks": bool(cfg.get("checks", True)), "workers": int(cfg.get("workers", 1)),
            "densities": tuple(cfg.get("densities", (6, 8, 12)))}


def outside(path: str, root: str) -> bool:
    """True when ``path`` (resolved, symlinks followed) is not ``root`` and not inside it."""
    p, r = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([p, r]) != r

