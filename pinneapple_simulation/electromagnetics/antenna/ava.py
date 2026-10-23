"""An open parametric antipodal Vivaldi antenna (AVA) for the EMerge FEM solver, written from scratch.

Geometry (millimetres; the board lies in the x-y plane, the antenna radiates end-fire along +y, z is normal to the
board; the substrate spans ``0 <= y <= Lb``, ``|x| <= W/2``, ``0 <= z <= h``):

* top layer (z = h): a 50-ohm microstrip of width ``wf`` (Pozar, *Microwave Engineering*, 4th ed., eq. 3.197) from
  the board edge y = 0 to ``y = xs``, then the top flare towards +x: inner edge
  ``x(y) = c1 exp(R (y - xs)) + c2`` from ``-wf/2`` at ``y = xs`` to ``Wa/2`` at the mouth ``y = Lb``, outer edge a
  quarter ellipse from ``+wf/2`` to the board side ``W/2`` (semi-axis ``(Lb - xs)/2`` along y), then straight;
* bottom layer (z = 0): the ground of width ``Wg`` from y = 0 to ``y = Lo`` (the microstrip ground), a linear taper
  to ``wf`` at ``y = xs`` (the microstrip-to-parallel-strip transition, the balun), then the mirror image
  (x -> -x) of the top flare;
* optional ``nslot`` rectangular corrugations of depth ``Ls``, width ``ws`` and pitch ``ps`` cut into the outer
  straight edge of both arms, counted from the mouth;
* optional elliptical director (semi-axes ``dir_a`` along y and ``dir_b`` along x) on the top layer, centred at
  ``(0, dir_x)`` in the flare opening;
* substrate ``er`` 3.55, ``tand`` 0.003, ``h`` 1.524 mm (a hydrocarbon-ceramic laminate class, e.g. 60-mil
  RO4003C-type values; change them freely), PEC conductors of zero thickness;
* air box with a margin of ``air * lambda`` on every side and a second-order absorbing boundary on its faces; the far
  field is the Stratton-Chu integral over those faces;
* a lumped 50-ohm port across the substrate at y = 0, under the microstrip.

Per frequency :func:`simulate` returns the complex S11, the directivity, IEEE gain and realized gain at boresight
(+y) and at the pattern peak, the radiation efficiency, and saves the 3-D far field (theta, phi, E_theta, E_phi) to
``ff_<f>GHz.npz``. Definitions (IEEE Std 145): ``U = |E|^2 r^2 / (2 eta0)``; ``D = 4 pi U / P_rad`` with ``P_rad``
the integral of ``U`` over the sphere; ``G = 4 pi U / P_acc`` with ``P_acc = P_inc (1 - |S11|^2)``;
``G_realized = 4 pi U / P_inc``; ``e_rad = P_rad / P_acc = G / D``.

:func:`layout` is pure NumPy (no EMerge) and raises :class:`InvalidGeometry` for a design that cannot be built
(the DOE skips those). EMerge (GPL-2.0, ``pip install emerge``) is imported only by :func:`simulate`; it is called,
never copied. ``python -m pinneapple_simulation.electromagnetics.antenna.ava --freq 3 --out run/`` runs one design
from the command line (the script the :mod:`.runner` drives, one frequency per process).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

__all__ = ["AVAParams", "InvalidGeometry", "Layout", "layout", "microstrip_width", "simulate", "use_superlu",
           "PARAMS", "main"]

C0 = 299_792_458.0
ETA0 = 376.730313412


class InvalidGeometry(ValueError):
    """The parameters do not describe a buildable antenna (overlapping metal, slots through the arm, ...)."""


@dataclass
class AVAParams:
    """Design parameters, millimetres (``R`` in 1/mm, ``res`` and ``air`` in wavelengths)."""
    Lb: float = 210.0        # board length (end-fire direction)
    W: float = 120.0         # board width
    Wa: float = 80.0         # mouth (aperture) opening between the two inner edges at y = Lb
    R: float = 0.03          # exponential flare rate [1/mm]
    xs: float = 45.0         # flare start (end of the microstrip / transition) along y
    Wg: float = 40.0         # ground width under the microstrip
    Lo: float = 20.0         # length of the full-width ground
    nslot: int = 0           # corrugation slots per arm (0 = none)
    Ls: float = 12.0         # slot depth (along x)
    ws: float = 3.0          # slot width (along y)
    ps: float = 8.0          # slot pitch
    dir_a: float = 0.0       # director semi-axis along y (0 = no director)
    dir_b: float = 0.0       # director semi-axis along x
    dir_x: float = 0.0       # director centre along y
    res: float = 0.25        # mesh size as a fraction of the wavelength
    er: float = 3.55
    tand: float = 0.003
    h: float = 1.524
    wf: float = 0.0          # microstrip width; 0 = the 50-ohm width from er and h
    air: float = 0.25        # air margin around the board [wavelengths]
    n_flare: int = 40        # points per flare curve

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AVAParams:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown AVA parameters {sorted(unknown)}")
        return cls(**{k: (int(round(float(v))) if k in ("nslot", "n_flare") else float(v)) for k, v in d.items()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def feed_width(self) -> float:
        return self.wf if self.wf > 0 else microstrip_width(50.0, self.er, self.h)


PARAMS = ("Lb", "W", "Wa", "R", "xs", "Wg", "Lo", "nslot", "Ls", "ws", "ps", "dir_a", "dir_b", "dir_x", "res")


def microstrip_width(z0: float, er: float, h: float) -> float:
    """Width of a microstrip of impedance ``z0`` on a substrate ``er``, ``h`` (Pozar eq. 3.197, same unit as h)."""
    a = z0 / 60.0 * math.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    wd = 8 * math.exp(a) / (math.exp(2 * a) - 2)
    if wd > 2:
        b = 377 * math.pi / (2 * z0 * math.sqrt(er))
        wd = 2 / math.pi * (b - 1 - math.log(2 * b - 1) + (er - 1) / (2 * er) * (math.log(b - 1) + 0.39 - 0.61 / er))
    return wd * h


@dataclass
class Layout:
    """Conductor polygons (mm, closed, counter-clockwise not guaranteed) and the derived dimensions."""
    params: AVAParams
    top: np.ndarray            # (n, 2)
    bottom: np.ndarray         # (n, 2)
    director: np.ndarray | None
    wf: float
    min_metal: float           # narrowest metal between the inner flare edge and the outer edge (or slot bottom)

    @property
    def area_mm2(self) -> float:
        return float(self.params.Lb * self.params.W)


def _inner(p: AVAParams, wf: float, y: np.ndarray) -> np.ndarray:
    span = p.Lb - p.xs
    c1 = (p.Wa / 2 + wf / 2) / math.expm1(p.R * span)
    return c1 * np.exp(p.R * (y - p.xs)) - wf / 2 - c1


def _slots(p: AVAParams) -> list[tuple[float, float]]:
    """(y_low, y_high) of each slot, from the mouth down."""
    out = []
    for k in range(int(p.nslot)):
        hi = p.Lb - p.ps / 2 - k * p.ps + p.ws / 2
        out.append((hi - p.ws, hi))
    return out


def layout(p: AVAParams) -> Layout:
    """The conductor polygons of design ``p``; :class:`InvalidGeometry` if it cannot be built."""
    wf = p.feed_width
    tol = 0.5                                                        # smallest metal / gap feature [mm]
    checks = [
        (p.Lb > 0 and p.W > 0 and p.h > 0, "Lb, W and h must be positive"),
        (0 < p.Lo < p.xs < p.Lb - 10, "need 0 < Lo < xs < Lb - 10 mm"),
        (wf + tol < p.Wg <= p.W, "need wf < Wg <= W"),
        (wf < p.Wa < p.W - 2 * tol, "need wf < Wa < W - 1 mm"),
        (p.R > 0, "R must be positive"),
        (0.05 <= p.res <= 0.4, "res must be in [0.05, 0.4] wavelengths"),
        (p.nslot >= 0, "nslot must be >= 0"),
    ]
    if p.nslot > 0:
        checks += [(p.ws > tol and p.Ls > tol and p.ps > p.ws + tol, "slots need ws, Ls > 0.5 mm and ps > ws")]
    if p.dir_a > 0 or p.dir_b > 0:
        checks += [(p.dir_a > 0 and p.dir_b > 0, "a director needs dir_a > 0 and dir_b > 0"),
                   (p.xs < p.dir_x - p.dir_a and p.dir_x + p.dir_a <= p.Lb, "the director must lie in the flare")]
    for ok, why in checks:
        if not ok:
            raise InvalidGeometry(why)

    n = int(p.n_flare)
    y = np.linspace(p.xs, p.Lb, n)
    xin = _inner(p, wf, y)
    B = (p.Lb - p.xs) / 2
    t = np.linspace(0, np.pi / 2, n)
    yo = p.xs + B * (1 - np.cos(t))
    xo = wf / 2 + (p.W / 2 - wf / 2) * np.sin(t)
    # outer edge: ellipse then the straight board side up to the mouth, with the slots cut in
    outer = [(x, yy) for x, yy in zip(xo, yo, strict=True)]
    slots = _slots(p)
    if slots and slots[-1][0] < yo[-1] + tol:
        raise InvalidGeometry("the slots reach into the curved part of the arm (lower nslot or ps)")
    for lo, hi in reversed(slots):
        outer += [(p.W / 2, lo), (p.W / 2 - p.Ls, lo), (p.W / 2 - p.Ls, hi), (p.W / 2, hi)]
    outer.append((p.W / 2, p.Lb))

    # narrowest metal: outer edge (or slot bottom) minus inner edge, sampled finely
    ys = np.linspace(p.xs + 1e-6, p.Lb, 2000)
    xo_s = np.interp(ys, yo, xo, right=p.W / 2)
    for lo, hi in slots:
        xo_s = np.where((ys >= lo) & (ys <= hi), p.W / 2 - p.Ls, xo_s)
    metal = xo_s - _inner(p, wf, ys)
    min_metal = float(metal[ys > p.xs + 2].min())
    if min_metal < tol:
        raise InvalidGeometry(f"the arm is {min_metal:.2f} mm wide somewhere (inner flare meets the outer edge "
                              "or a slot)")

    # top: microstrip (y = 0 .. xs), outer edge up to the mouth, inner edge back down
    top = [(-wf / 2, 0.0), (wf / 2, 0.0)] + outer + [(x, yy) for x, yy in zip(xin[::-1], y[::-1], strict=True)]
    top = np.array(_dedupe(top))
    # bottom: ground, taper, mirrored arm
    arm = np.array(_dedupe(outer + [(x, yy) for x, yy in zip(xin[::-1], y[::-1], strict=True)]))
    arm_m = arm * [-1, 1]
    bottom = [(-p.Wg / 2, 0.0), (p.Wg / 2, 0.0), (p.Wg / 2, p.Lo)] + [tuple(v) for v in arm_m[::-1]] \
        + [(-p.Wg / 2, p.Lo)]
    # arm_m reversed starts at the mirrored inner edge (x = +wf/2 at y = xs) and ends at -wf/2 at y = xs
    bottom = np.array(_dedupe(bottom))

    director = None
    if p.dir_a > 0:
        yd = np.linspace(p.dir_x - p.dir_a, p.dir_x + p.dir_a, 200)
        half = p.dir_b * np.sqrt(np.clip(1 - ((yd - p.dir_x) / p.dir_a) ** 2, 0, 1))
        if np.any(half > _inner(p, wf, yd) - tol):
            raise InvalidGeometry("the director touches the flare (lower dir_b or move dir_x towards the mouth)")
        th = np.linspace(0, 2 * np.pi, 33)[:-1]
        director = np.c_[p.dir_b * np.cos(th), p.dir_x + p.dir_a * np.sin(th)]
    return Layout(p, top, bottom, director, wf, min_metal)


def _dedupe(pts: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for q in pts:
        if not out or abs(q[0] - out[-1][0]) > eps or abs(q[1] - out[-1][1]) > eps:
            out.append((float(q[0]), float(q[1])))
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
        out.pop()
    return out


# ------------------------------------------------------------------------------------------------- EMerge model
def use_superlu() -> None:
    """Force EMerge's SuperLU direct solver.

    Measured here: EMerge's PARDISO backend can die with SIGBUS when MKL is not usable; SuperLU always runs."""
    import emerge._emerge.solver as _s
    _s._PARDISO_AVAILABLE = False


def _farfield(ds, faces, n_theta: int = 37, n_phi: int = 73):
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(-np.pi, np.pi, n_phi)
    ff = ds.farfield_3d(faces, th, ph)
    T = ff.theta
    Et, Ep = ff.Etheta, ff.Ephi
    U = (np.abs(Et) ** 2 + np.abs(Ep) ** 2) / (2 * ETA0)              # r = 1 m
    # P_rad = integral U dOmega on the (phi, theta) grid (trapezoid in both angles)
    prad = float(np.trapezoid(np.trapezoid(U * np.sin(T), th, axis=1), ph))
    # boresight +y: theta = 90 deg, phi = 90 deg
    Eb = ds.farfield(np.array([np.pi / 2]), np.array([np.pi / 2]), faces)[0][:, 0]
    Ub = float(np.sum(np.abs(Eb) ** 2) / (2 * ETA0))
    return th, ph, Et, Ep, U, prad, Ub, float(ff.Ptot)


def simulate(params: AVAParams | dict, freqs_ghz, out_dir: str | None = None, *, superlu: bool = True,
             log: str = "WARNING") -> list[dict[str, Any]]:
    """Build the design in EMerge, solve each frequency (one mesh per frequency, size ``res * lambda``) and return
    one row per frequency (the columns of :mod:`.runner`'s table)."""
    p = params if isinstance(params, AVAParams) else AVAParams.from_dict(params)
    lay = layout(p)
    os.environ.setdefault("EMERGE_STD_LOGLEVEL", log)
    import emerge as em
    if superlu:
        use_superlu()
    rows = []
    for f_ghz in np.atleast_1d(np.asarray(freqs_ghz, dtype=float)):
        t0 = time.perf_counter()
        rows.append(_solve_one(em, p, lay, float(f_ghz), out_dir))
        rows[-1]["wall_s"] = round(time.perf_counter() - t0, 2)
    return rows


def _solve_one(em, p: AVAParams, lay: Layout, f_ghz: float, out_dir: str | None) -> dict[str, Any]:
    mm = 1e-3
    f = f_ghz * 1e9
    lam = C0 / f
    m_air = p.air * lam
    h = p.h * mm
    name = f"ava_{f_ghz:g}GHz"
    with em.Simulation(name, save_file=False) as sim:
        sub_mat = em.Material(er=p.er, tand=p.tand, name="substrate")
        air = em.geo.Box(p.W * mm + 2 * m_air, p.Lb * mm + 2 * m_air, h + 2 * m_air,
                         position=(-p.W * mm / 2 - m_air, -m_air, -m_air))
        sub = em.geo.Box(p.W * mm, p.Lb * mm, h, position=(-p.W * mm / 2, 0, 0)).set_material(sub_mat)
        top = em.geo.XYPolygon(list(lay.top[:, 0] * mm), list(lay.top[:, 1] * mm)).geo(em.GCS.displace(0, 0, h))
        bot = em.geo.XYPolygon(list(lay.bottom[:, 0] * mm), list(lay.bottom[:, 1] * mm)).geo(em.GCS)
        metals = [top, bot]
        if lay.director is not None:
            metals.append(em.geo.XYPolygon(list(lay.director[:, 0] * mm), list(lay.director[:, 1] * mm))
                          .geo(em.GCS.displace(0, 0, h)))
        wf = lay.wf * mm
        port = em.geo.Plate((-wf / 2, 0, 0), (wf, 0, 0), (0, 0, h))
        sim.commit_geometry(air, sub, *metals, port)
        sim.mw.set_resolution(p.res)
        sim.mw.set_frequency(f)
        sim.generate_mesh()
        for mt in metals:
            sim.mw.bc.PEC(mt)
        sim.mw.bc.LumpedPort(port, 1, width=wf, height=h, direction=em.ZAX, Z0=50)
        abc = sim.mw.bc.AbsorbingBoundary(air.boundary())
        data = sim.mw.run_sweep()
        n_tets = int(sim.mesh.n_tets) if hasattr(sim.mesh, "n_tets") else -1
        grid = data.scalar.grid
        s11 = complex(np.asarray(grid.S(1, 1)).ravel()[0])          # grid.model_S fails with one frequency
        ds = data.field[0]
        th, ph, Et, Ep, U, prad, Ub, ptot = _farfield(ds, abc.selection if hasattr(abc, "selection") else
                                                       air.boundary())
    pinc = 1.0                                                        # LumpedPort(power=1): 1 W incident
    pacc = pinc * (1 - abs(s11) ** 2)
    umax = float(U.max())
    D_b, D_p = 4 * np.pi * Ub / prad, 4 * np.pi * umax / prad
    G_b, G_p = 4 * np.pi * Ub / pacc, 4 * np.pi * umax / pacc
    Gr_b, Gr_p = 4 * np.pi * Ub / pinc, 4 * np.pi * umax / pinc
    dB = lambda x: float(10 * np.log10(max(x, 1e-30)))  # noqa: E731
    row = {"f_GHz": f_ghz, "S11_re": s11.real, "S11_im": s11.imag, "S11_dB": float(20 * np.log10(abs(s11))),
           "D_bore_dBi": dB(D_b), "G_bore_dBi": dB(G_b), "Greal_bore_dBi": dB(Gr_b), "D_peak_dBi": dB(D_p),
           "G_peak_dBi": dB(G_p), "Greal_peak_dBi": dB(Gr_p), "rad_eff": float(prad / pacc),
           "P_rad_pattern": prad, "P_rad_surface": ptot, "n_tets": n_tets, "res": p.res}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"ff_{f_ghz:g}GHz.npz"), theta=th, phi=ph, E_theta=Et, E_phi=Ep,
                            f_GHz=f_ghz)
    return row


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Simulate one open AVA design in EMerge, one table row per frequency.")
    ap.add_argument("--params", default="{}", help="JSON object of AVAParams (mm), or @file.json")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V", help="override parameters")
    ap.add_argument("--freq", type=float, nargs="+", required=True, help="frequencies [GHz]")
    ap.add_argument("--out", default=".", help="output folder (result.json and far-field npz)")
    ap.add_argument("--tag", default="design")
    ap.add_argument("--pardiso", action="store_true", help="use PARDISO instead of SuperLU")
    a = ap.parse_args(argv)
    raw = a.params
    d = json.load(open(raw[1:])) if raw.startswith("@") else json.loads(raw)
    d.update({k: v for k, v in (s.split("=", 1) for s in a.set)})
    os.makedirs(a.out, exist_ok=True)
    try:
        p = AVAParams.from_dict(d)
        rows = simulate(p, a.freq, a.out, superlu=not a.pardiso)
        res = {"tag": a.tag, "status": "ok", "params": p.to_dict(), "rows": rows}
    except InvalidGeometry as e:
        res = {"tag": a.tag, "status": "invalid", "error": str(e), "params": d, "rows": []}
    with open(os.path.join(a.out, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"tag": a.tag, "status": res["status"], "n": len(res["rows"])}))
    return 0 if res["status"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
