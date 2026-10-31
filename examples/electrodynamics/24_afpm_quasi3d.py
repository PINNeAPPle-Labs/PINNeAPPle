"""Axial-flux PM machine: full 3-D against quasi-3D on the same discretisation (edge elements).

Machine (``AFPMMachine`` defaults): 12 slots, 8 poles, r = 50 to 90 mm, 1 mm gap, 4 mm magnets (Br = 1.2 T, 80 % of
the pole pitch), open slots (half the slot pitch), linear iron (mu_r = 1000); single-sided (rotor + slotted stator
with a yoke) and double-rotor (yokeless stator between two rotors). Grid: 384 steps round the axis, 4 mm radially,
2.5 mm axially, 15 mm of air inside, outside, above and below.

Quasi-3D: the active annulus cut into 1 or 5 radial slices with flux-parallel faces, each solved alone on the
same grid and summed. Against 3-D: one slice (flux-parallel at r_in and r_out) leaves out the fringing at the inner
and outer radius; five slices also leave out the radial flux inside the annulus. Compared: cogging over one period
(15 deg, every 1.875 deg) and the no-load flux linkage of a coil around a tooth with a magnet centred on it.
A coarse grid (192 steps, 8 mm, 4 mm) gives the discretisation sensitivity.

    python examples/electrodynamics/24_afpm_quasi3d.py       # about 1.5 h, 9 GB; resumes from its checkpoint
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, AFPMMachine

OUT = Path(__file__).parent / "results"
TOPOLOGIES = ("single", "double_rotor")
GRIDS = {"fine": {"n_theta": 384, "dr": 0.004, "dz": 0.0025}, "coarse": {"n_theta": 192, "dr": 0.008, "dz": 0.004}}
SLICES = (1, 5)


def _run(model: AFPM3D, thetas, theta_link: float) -> dict:
    t0 = time.perf_counter()
    cog = model.cogging(thetas)
    s = model.solve(theta_link)
    out = {"cogging": cog.tolist(), "pkpk": float(cog.max() - cog.min()), "linkage_Wb": model.tooth_linkage(s),
           "tets": int(len(model.mesh.tets)), "seconds": time.perf_counter() - t0}
    if model.r_range is None:
        out["linkage_o5mm_Wb"] = model.tooth_linkage(s, overhang=0.005)
    model.release()
    return out


def main() -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    ck = OUT / "24_afpm_quasi3d.rows.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    for topo in TOPOLOGIES:
        m = AFPMMachine(topology=topo)
        for grid, g in GRIDS.items():
            model = AFPM3D(m, air_r=0.015, air_z=0.015, **g)
            period = m.cogging_period
            thetas = np.arange(0, round(period / model.step), 2) * model.step       # even steps over a period
            slot_pitch = 2 * math.pi / m.slots
            theta_link = slot_pitch / 2                                              # magnet 0 centred on tooth 0
            cases = [(f"{topo}.{grid}.3d", None)] + [(f"{topo}.{grid}.q{n}", n) for n in SLICES]
            for key, n in cases:
                if key in done:
                    continue
                if n is None:
                    r = _run(model, thetas, theta_link)
                else:
                    parts = [_run(q, thetas, theta_link) for q in model.quasi3d_models(n)]
                    cog = np.sum([p["cogging"] for p in parts], axis=0)
                    r = {"cogging": cog.tolist(), "pkpk": float(cog.max() - cog.min()),
                         "linkage_Wb": float(sum(p["linkage_Wb"] for p in parts)),
                         "tets": sum(p["tets"] for p in parts), "seconds": sum(p["seconds"] for p in parts)}
                done[key] = {"key": key, "thetas_deg": np.degrees(thetas).tolist(), "theta_link_deg":
                             math.degrees(theta_link), "slot_pitch_deg": math.degrees(slot_pitch), **r}
                with open(ck, "a") as fh:
                    fh.write(json.dumps(done[key]) + "\n")
                print(f"{key}: cogging pk-pk {r['pkpk']:.4f} N m, tooth linkage {r['linkage_Wb'] * 1e3:.4f} mWb, "
                      f"{r['tets']} tetrahedra, {r['seconds']:.0f} s", flush=True)
            del model
    rows = {}
    for topo in TOPOLOGIES:
        for grid in GRIDS:
            ref = done[f"{topo}.{grid}.3d"]
            for n in SLICES:
                q = done[f"{topo}.{grid}.q{n}"]
                rows[f"{topo}.{grid}.q{n}"] = {"pkpk_ratio": q["pkpk"] / ref["pkpk"],
                                               "linkage_ratio": q["linkage_Wb"] / ref["linkage_Wb"]}
    out = {"cases": done, "ratios_quasi3d_over_3d": rows, "seconds": time.perf_counter() - t0,
           "date": time.strftime("%Y-%m-%d")}
    (OUT / "24_afpm_quasi3d.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {"3d": "#2a78d6", "q1": "#eb6834", "q5": "#1baf7a"}
    labels = {"3d": "3-D", "q1": "quasi-3D, 1 slice", "q5": "quasi-3D, 5 slices"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), facecolor="#fcfcfb")
    for a, topo in zip(ax, TOPOLOGIES, strict=True):
        for k, c in colours.items():
            r = out["cases"][f"{topo}.fine.{k}"]
            th = r["thetas_deg"]
            a.plot(th, r["cogging"], "-o", color=c, lw=2, ms=5, label=f"{labels[k]} (pk-pk {r['pkpk']:.2f} N m)")
        a.set(xlabel="rotor angle (deg)", ylabel="cogging torque (N m)",
              title=f"{'Single-sided' if topo == 'single' else 'Double-rotor'} AFPM, one cogging period")
        a.legend(frameon=False, fontsize=8)
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
        a.axhline(0, color="#52514e", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "24_afpm_quasi3d.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
