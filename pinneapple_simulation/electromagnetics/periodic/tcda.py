"""A generic tightly coupled dipole array (TCDA) unit cell for EMerge Floquet runs, written from scratch.

The cell (millimetres; infinite array in the x-y plane, E-plane = x-z): a square cell of side ``a``; the ground
plane at z = 0 (the bottom face of the cell, PEC); a dipole strip of width ``w`` along x at height ``h``, fed across
a gap ``s`` at the centre by a lumped port of ``z_port`` ohms; its arms stop ``g / 2`` from the cell edge, so
neighbouring dipoles couple capacitively across a gap ``g`` (the "tight coupling"); an optional dielectric
superstrate (``er_s``, thickness ``t_s``) at ``d_s`` above the dipole; air up to half a wavelength (at the lowest
frequency) above everything, closed by a Floquet port. Periodic boundary conditions with the phase of the scan
angle (theta, phi) close the sides.

Per scan angle and frequency the run returns the active reflection coefficient at the lumped port
(``S11`` with every element excited with the scan phase). One mesh serves every scan angle of a cell (the scan only
changes the boundary phases and the Floquet port), so a cell at several angles costs one mesh and one solve per
angle. EMerge (GPL-2.0, ``pip install emerge``) is imported only to solve; this module is called, never copies it.
SuperLU is used (PARDISO can die with SIGBUS without MKL).

``python tcda.py --params '{"h": 9}' --freq 3 4 5 6 --scan 0 30 45 --out run/`` writes ``result.json`` (the
:mod:`pinneapple_simulation.electromagnetics.antenna.runner` format); it has no package imports, so the runner
starts it as a plain script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

__all__ = ["TCDACell", "InvalidCell", "simulate", "main"]

C0 = 299_792_458.0


class InvalidCell(ValueError):
    """Parameters that do not describe a buildable cell."""


@dataclass
class TCDACell:
    a: float = 22.0          # cell side [mm]
    h: float = 10.0          # dipole height above the ground [mm]
    w: float = 2.0           # strip width [mm]
    g: float = 1.0           # gap between neighbouring dipoles (coupling) [mm]
    s: float = 1.0           # feed gap [mm]
    t_s: float = 3.0         # superstrate thickness [mm] (0: none)
    d_s: float = 4.0         # superstrate distance above the dipole [mm]
    er_s: float = 3.0        # superstrate relative permittivity
    z_port: float = 100.0    # feed impedance [ohm]
    res: float = 0.1         # mesh size as a fraction of the wavelength
    air: float = 0.35        # air above the stack [wavelengths at the lowest frequency]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCDACell:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown TCDA parameters {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, f_max_ghz: float | None = None, theta_max_deg: float = 0.0) -> None:
        ok = [(self.a > 0 and self.h > 0 and self.w > 0, "a, h and w must be positive"),
              (0 < self.g < self.a / 4 and 0 < self.s < self.a / 4, "0 < g, s < a / 4"),
              (self.w < self.a / 2, "w < a / 2"),
              (self.t_s >= 0 and self.d_s >= 0 and self.er_s >= 1, "t_s, d_s >= 0 and er_s >= 1"),
              (0.02 <= self.res <= 0.3, "res in [0.02, 0.3]")]
        if f_max_ghz:
            lam = C0 / (f_max_ghz * 1e9) * 1e3
            lim = lam / (1 + np.sin(np.deg2rad(theta_max_deg)))
            ok.append((self.a < lim, f"a = {self.a:g} mm >= lambda / (1 + sin theta) = {lim:.1f} mm: a grating lobe "
                                     "enters at the highest frequency and scan"))
        for good, why in ok:
            if not good:
                raise InvalidCell(why)


def _use_superlu() -> None:
    import emerge._emerge.solver as _s
    _s._PARDISO_AVAILABLE = False


def simulate(cell: TCDACell | dict, freqs_ghz, scans_deg=((0.0, 0.0),), *, log: str = "WARNING") -> list[dict]:
    """Active reflection of ``cell`` at every frequency (GHz) and scan ``(theta, phi)`` (degrees); one row each."""
    c = cell if isinstance(cell, TCDACell) else TCDACell.from_dict(cell)
    fs = sorted(float(f) for f in np.atleast_1d(freqs_ghz))
    c.check(max(fs), max(abs(float(t)) for t, _ in scans_deg))
    os.environ.setdefault("EMERGE_STD_LOGLEVEL", log)
    import emerge as em
    _use_superlu()
    mm = 1e-3
    lam_max = C0 / (min(fs) * 1e9)
    top_stack = c.h + (c.d_s + c.t_s if c.t_s > 0 else 0.0)
    ztop = top_stack * mm + c.air * lam_max
    a = c.a * mm
    arm = (c.a / 2 - c.g / 2 - c.s / 2) * mm
    rows = []
    t0 = time.perf_counter()
    with em.Simulation("tcda_cell", save_file=False) as sim:
        cellg = em.RectCell(a, a)
        sim.set_periodic_cell(cellg)
        objs = [cellg.volume(0, ztop)]
        if c.t_s > 0:
            objs.append(em.geo.Box(a, a, c.t_s * mm, position=(-a / 2, -a / 2, (c.h + c.d_s) * mm))
                        .set_material(em.Material(er=c.er_s, tand=0.0, name="superstrate")))
        right = em.geo.XYPlate(arm, c.w * mm, position=(c.s / 2 * mm, -c.w / 2 * mm, c.h * mm))
        left = em.geo.XYPlate(arm, c.w * mm, position=(-(c.a / 2 - c.g / 2) * mm, -c.w / 2 * mm, c.h * mm))
        port = em.geo.XYPlate(c.s * mm, c.w * mm, position=(-c.s / 2 * mm, -c.w / 2 * mm, c.h * mm))
        top = cellg.port_face(ztop)
        sim.commit_geometry(*objs, right, left, port, top)
        sim.mw.set_resolution(c.res)
        sim.mw.set_frequency([f * 1e9 for f in fs])
        sim.generate_mesh()
        sim.mw.bc.PEC(right)
        sim.mw.bc.PEC(left)
        sim.mw.bc.LumpedPort(port, 1, width=c.w * mm, height=c.s * mm, direction=em.XAX, Z0=c.z_port)
        sim.mw.bc.floquet_port(top, 2)
        sim.mw._initialize_bc_data()                 # creates the periodic conditions the scan phase goes into
        n_tets = int(sim.mesh.n_tets)
        t_mesh = time.perf_counter() - t0
        for th, ph in scans_deg:
            t1 = time.perf_counter()
            cellg.set_scanangle(float(th), float(ph))
            data = sim.mw.run_sweep()
            sets = list(data.scalar.iter())[-len(fs):]
            for f, ds in zip(fs, sets, strict=True):
                s11 = complex(np.asarray(ds.S(1, 1)).ravel()[0])
                rows.append({"f_GHz": f, "theta_deg": float(th), "phi_deg": float(ph), "S11_re": s11.real,
                             "S11_im": s11.imag, "S11_dB": float(20 * np.log10(max(abs(s11), 1e-12))),
                             "solve_s": round(time.perf_counter() - t1, 2), "n_tets": n_tets,
                             "mesh_s": round(t_mesh, 2)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Active reflection of one TCDA unit cell (EMerge Floquet runs).")
    ap.add_argument("--params", default="{}", help="JSON object of TCDACell (mm), or @file.json")
    ap.add_argument("--freq", type=float, nargs="+", required=True, help="frequencies [GHz]")
    ap.add_argument("--scan", type=float, nargs="+", default=[0.0], help="scan theta [deg] (E-plane, phi = 0)")
    ap.add_argument("--phi", type=float, default=0.0, help="scan plane phi [deg] (0: E-plane, 90: H-plane)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="cell")
    a = ap.parse_args(argv)
    d = json.load(open(a.params[1:])) if a.params.startswith("@") else json.loads(a.params)
    os.makedirs(a.out, exist_ok=True)
    try:
        cell = TCDACell.from_dict(d)
        rows = simulate(cell, a.freq, [(t, a.phi) for t in a.scan])
        res = {"tag": a.tag, "status": "ok", "params": cell.to_dict(), "rows": rows}
    except InvalidCell as e:
        res = {"tag": a.tag, "status": "invalid", "error": str(e), "params": d, "rows": []}
    with open(os.path.join(a.out, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"tag": a.tag, "status": res["status"], "n": len(res["rows"])}))
    return 0 if res["status"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
