"""TEAM workshop problem 7 (asymmetrical conductor with a hole) with the 3-D edge-element eddy-current solver,
against the published measurements, under mesh refinement.

The plate size ``h_plate`` sets the refinement (the skin depth is 12 mm at 50 Hz and 6 mm at 200 Hz in the
19 mm plate); the coil (20 mm), the probe layer between plate and coil (10 mm) and the air (80 mm) keep their
sizes. Each level is solved at 50 and 200 Hz with the tree-gauged A* formulation factorised by PARDISO (pypardiso)
and refined to a relative residual of 1e-10. Measured: ``Bz`` along A1-B1 and A2-B2 and the eddy-current density
``Jy`` on both faces of the plate, at wt = 0 and 90 deg (``fem3d.team7.MEASURED``).

    python examples/electrodynamics/21_team7.py       # needs gmsh; about 10 min
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.fem3d import team7

OUT = Path(__file__).parent / "results"
H_PLATE = (0.012, 0.009, 0.007, 0.006, 0.005)


def main() -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    ck = OUT / "21_team7.levels.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            r = json.loads(line)
            done[r["h_plate"]] = r
    levels = []
    for h in H_PLATE:
        if h in done:
            levels.append(done[h])
            print(f"h_plate {h}: from the checkpoint", flush=True)
            continue
        msh = team7.mesh_team7(OUT / f"_team7_{h}.msh", h_plate=h, h_coil=0.02, h_air=0.08, h_layer=0.01)
        r = team7.run(msh, freqs=(50, 200), solver="pardiso", rtol=1e-10)
        msh.unlink()
        r["h_plate"] = h
        r["f"] = {str(k): v for k, v in r["f"].items()}
        levels.append(r)
        with open(ck, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        f50 = r["f"]["50"]["compare"]
        print(f"h_plate {h}: {r['unknowns']} unknowns, {r['seconds']:.0f} s, loss 50 Hz "
              f"{r['f']['50']['loss_W']:.3f} W, A1B1 wt=0 rms/peak {f50['A1B1 wt=0']['rms_rel_peak']:.3f}",
              flush=True)
    out = {"levels": levels, "h_plate": list(H_PLATE), "seconds": time.perf_counter() - t0,
           "date": time.strftime("%Y-%m-%d")}
    (OUT / "21_team7.json").write_text(json.dumps(out, indent=1))
    figure(out)
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"
    fin = out["levels"][-1]["f"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), facecolor="#fcfcfb")
    for j, (line, f) in enumerate((("A1B1", "50"), ("A2B2", "200"))):
        a = ax[0, j]
        for wt, c in ((0, blue), (90, orange)):
            r = fin[f]["compare"][f"{line} wt={wt}"]
            a.plot(team7.X_B, r["measured"], "o", color=c, ms=5, label=f"measured, wt = {wt}")
            a.plot(team7.X_B, r["computed"], "-", color=c, lw=2, label=f"computed, wt = {wt}")
        a.set(xlabel="x (mm)", ylabel="Bz (gauss)", title=f"Bz along {line}, {f} Hz")
        a.legend(frameon=False, fontsize=8)
    a = ax[1, 0]
    for wt, c in ((0, blue), (90, orange)):
        r = fin["200"]["compare"][f"A3B3 wt={wt}"]
        a.plot(team7.X_J, r["computed"], "-", color=c, lw=2, label=f"computed z = 19 mm, wt = {wt}")
        a.plot(team7.X_J, r["measured"], "s", color=c, mfc="none", ms=5, label=f"table A3-B3 (z = 19), wt = {wt}")
        a.plot(team7.X_J, team7.MEASURED[("A4B4", 200, wt)], "o", color=c, ms=4,
               label=f"table A4-B4 (z = 0), wt = {wt}")
    a.set(xlabel="x (mm)", ylabel="Jy (1e6 A/m^2)", title="Jy on the face nearer the coil, 200 Hz")
    a.legend(frameon=False, fontsize=7)
    a = ax[1, 1]
    n = [lv["unknowns"] for lv in out["levels"]]
    for f, c in (("50", blue), ("200", orange)):
        a.plot(n, [lv["f"][f]["loss_W"] for lv in out["levels"]], "-o", color=c, lw=2, label=f"loss, {f} Hz")
    a.set(xscale="log", xlabel="unknowns", ylabel="Joule loss (W)", title="Mesh convergence")
    a.legend(frameon=False, fontsize=8)
    for a in ax.flat:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
        a.axhline(0, color=muted, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "21_team7.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
