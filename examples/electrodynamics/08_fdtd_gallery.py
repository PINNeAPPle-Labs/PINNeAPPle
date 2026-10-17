"""FDTD gallery: a plane wave scattered by a dielectric cylinder (animation, field map with Poynting arrows, comparison
with the exact series), and the radiation pattern of a Hertzian dipole from the near-to-far-field transform.
Every figure carries its check: the cylinder's field against the series of cylinder functions, the snapshots
through the Maxwell physics checks (trust report), the dipole pattern and power against the closed forms.

    python examples/electrodynamics/08_fdtd_gallery.py            # 30 cells per wavelength, about 1 min
    python examples/electrodynamics/08_fdtd_gallery.py --mp4      # also an MP4 (needs ffmpeg)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pinneapple_physics import maxwell_checks as mc
from pinneapple_simulation.electromagnetics import fdtd as F
from pinneapple_simulation.electromagnetics import waves
from pinneapple_simulation.electromagnetics.constants import C0, ETA0
from pinneapple_simulation.electromagnetics.media import Medium
from pinneapple_tools.visualization import electromagnetics as emviz

OUT = Path(__file__).parent / "results"
f0 = 1e9
lam = C0 / f0
summary: dict = {}

# ------------------------------------------------------------------ 1. plane wave on a dielectric cylinder (TMz)
ncl, a, eps_c = 30, 0.3 * lam, 3.0
dx = lam / ncl
N = int(3 * lam / dx) + 50
c = N // 2
grid = F.YeeGrid((N, N, 1), (dx, dx, dx))
sim = F.FDTD(grid, {"x": "cpml", "y": "cpml"}, npml=20)
fill = F.fill_fraction(grid, lambda x, y, z: np.hypot(x - c * dx, y - c * dx) <= a, 16)
sim.set_material(fill, Medium(eps_c))
sim.add_plane_wave(F.PlaneWaveTFSF("x", "Ez", F.modulated_gaussian(f0, 0.6 * f0), ((30, N - 30), (30, N - 30), (0, 1))))
inc = sim.incident_monitor([f0])
dft = sim.add_dft(["Ez", "Hx", "Hy"], [f0])
frames, times, snaps = [], [], []
sim.run_until(40 / f0)

E0 = inc.phasor("Ey")[0, c, 0, 0]
Ez = dft.phasor("Ez")[0, :, :, 0] / E0
ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
ii = np.round(c + 0.5 * lam * np.cos(ang) / dx).astype(int)
jj = np.round(c + 0.5 * lam * np.sin(ang) / dx).astype(int)
exact = waves.cylinder_scattering_tm((ii - c) * dx, (jj - c) * dx, 2 * np.pi / lam, a, eps_c)
summary["cylinder"] = {"cells_per_wavelength": ncl, "eps": eps_c, "radius_wavelengths": 0.3,
                       "ring_error_vs_series": float(np.max(np.abs(Ez[ii, jj] - exact)) / np.max(np.abs(exact)))}

# time-averaged Poynting vector S = (1/2) Re(E x H*) of the TMz phasors: Sx = -Re(Ez Hy*)/2, Sy = Re(Ez Hx*)/2
Hx = dft.phasor("Hx")[0, :, :, 0] / E0
Hy = dft.phasor("Hy")[0, :, :, 0] / E0
Sx, Sy = -0.5 * np.real(Ez * np.conj(Hy)), 0.5 * np.real(Ez * np.conj(Hx))
tf = slice(31, N - 31)                                         # the total-field box (outside it: scattered field)
ext_tf = (31 * dx, (N - 32) * dx, 31 * dx, (N - 32) * dx)
emviz.field_map(np.abs(Ez)[tf, tf], ext_tf, OUT / "fdtd_cylinder_field_map.png", outline=fill[tf, tf, 0],
                vectors=(Sx[tf, tf], Sy[tf, tf]), symmetric=False, cmap="magma", label="|Ez| / |E_inc|",
                title="|Ez| at 1 GHz and the time-averaged Poynting vector")

# the snapshots through the Maxwell physics checks (one period, cropped to the total-field box, cylinder excluded)
check = F.FDTD(grid, {"x": "cpml", "y": "cpml"}, npml=20)
check.set_material(fill, Medium(eps_c))
check.add_plane_wave(F.PlaneWaveTFSF("x", "Ez", F.sinusoid(f0, 2.0), ((30, N - 30), (30, N - 30), (0, 1))))
cut = slice(20, N - 20)                                        # drop the PML from the movie
ext_cut = (20 * dx, (N - 21) * dx, 20 * dx, (N - 21) * dx)
check.run(int(6 / f0 / check.dt), every=3, callback=lambda s: (frames.append(s.field_at_nodes("Ez")[cut, cut, 0].copy()),
                                                              times.append(s.n * s.dt)))
emviz.field_movie(frames, ext_cut, OUT / "fdtd_cylinder_scattering.gif", times=times, outline=fill[cut, cut, 0],
                  title="Ez, 1 GHz wave on an eps = 3 cylinder", label="Ez [V/m]", fps=16, dpi=60)
if "--mp4" in sys.argv:
    emviz.field_movie(frames, ext_cut, OUT / "fdtd_cylinder_scattering.mp4", times=times, outline=fill[cut, cut, 0],
                      title="Ez, 1 GHz wave on an eps = 3 cylinder", label="Ez [V/m]", fps=24)
ts = []
for _ in range(int(1 / f0 / check.dt)):
    check.step()
    t, snap = check.snapshot()
    ts.append(t)
    snaps.append({k: v[35:60, 35:N - 35, 0] for k, v in snap.items() if k in ("Ez", "Hx", "Hy")})
fields = {k: np.stack([s[k] for s in snaps], -1) for k in ("Ez", "Hx", "Hy")}
report, checks = mc.maxwell_trust_report({"x": np.arange(35, 60) * dx, "y": np.arange(35, N - 35) * dx,
                                          "t": np.array(ts)}, fields)
summary["maxwell_checks"] = {"decision": report.decisao,
                             **{c.key: (c.value if c.passed is not None else None) for c in checks}}

# ------------------------------------------------------------------ 2. Hertzian dipole: far-field pattern
ncl3 = 15
dx3 = lam / ncl3
N3 = int(1.2 * lam / dx3) + 28
c3 = N3 // 2
dip = F.FDTD(F.YeeGrid((N3, N3, N3), (dx3, dx3, dx3)), {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=10)
w = F.modulated_gaussian(f0, 0.5 * f0)
dip.add_current("Ez", (c3, c3, c3), w, 1.0)
ntff = F.NearToFarField(dip, [f0], ((14, N3 - 14),) * 3)
dip.run_until(14 / f0)
tj = (np.arange(dip.n) + 0.5) * dip.dt
Il = np.sum(np.array([w(x) for x in tj]) * np.exp(-2j * np.pi * f0 * tj)) * dip.dt * dx3 ** 3
th = np.linspace(0, 2 * np.pi, 361)
Et, _ = ntff.far_field(np.abs(np.where(th > np.pi, 2 * np.pi - th, th)), np.where(th > np.pi, np.pi, 0.0))
analytic = ETA0 * (2 * np.pi / lam) * abs(Il) * np.abs(np.sin(th)) / (4 * np.pi)
emviz.radiation_pattern(th, Et[0], OUT / "fdtd_dipole_pattern.png", reference=analytic,
                        labels=("FDTD + near-to-far field", "sin(theta)"), title="Hertzian dipole, E-plane")
P = float(ntff.radiated_power()[0])
summary["dipole"] = {"cells_per_wavelength": ncl3,
                     "pattern_error": float(np.max(np.abs(np.abs(Et[0]) - analytic)) / analytic.max()),
                     "radiated_power_W": P, "radiated_power_closed_form_W": waves.hertzian_dipole_power(Il, 2 * np.pi * f0)}

OUT.mkdir(exist_ok=True)
(OUT / "fdtd_gallery.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
