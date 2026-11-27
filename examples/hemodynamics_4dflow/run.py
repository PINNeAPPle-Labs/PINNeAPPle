"""Pressure and wall shear stress from 4D-flow MRI, measured on a phantom against Womersley's closed form (#362).

    python examples/hemodynamics_4dflow/run.py --out examples/hemodynamics_4dflow [--quick]

4D-flow MRI measures velocity. A clinician wants the pressure gradient and the wall shear stress, and neither is a
velocity: both are derivatives, one along the vessel and one through the wall, and the wall is the one place the
scanner does not sample. The claim that a physics-informed network can supply them is tested here against a case with
a closed form -- not a patient, not a downloaded benchmark, a synthetic straight artery whose exact solution is
Womersley's.

**What is measured.** The acquisition is synthesised from the closed form: voxel centres across the radius (so the
nearest sample is half a voxel from the wall and the wall is never sampled), 24 cardiac phases, 3 planes with
independent noise, and Gaussian noise at 5 % of the peak velocity. The network is the axisymmetric Navier-Stokes
system in Womersley variables, with no-slip, axis symmetry and periodicity built into the ansatz
(:class:`pinneapple_physics.hemodynamics.WomersleyPINN`). The errors are relative L2 over the cardiac cycle, against
the closed form: the wall shear stress waveform and the pressure difference between two planes.

**What else is measured.** The two-point difference every 4D-flow pipeline starts from, for comparison; the same
errors as a function of the voxel size (4, 8, 16, 32 across the radius); the same errors as a function of the noise
level; and the spread of an ensemble of independently seeded fits, which is the uncertainty this module reports.

**What this is not.** No patient data, no real acquisition (no slice profile, no partial volume, no aliasing), no
curved or compliant vessel. A straight rigid tube is where a closed form exists; every number below is a statement
about that phantom and about a network trained on 80 s of CPU time per fit.

Writes ``<out>/results/hemodynamics_4dflow.json`` after every case and four figures in ``<out>/figures``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_physics.hemodynamics import (  # noqa: E402
    BLOOD,
    DEFAULT_VESSELS,
    PHYSIOLOGICAL_FLOW,
    TWO_PI,
    PINNConfig,
    Womersley,
    ensemble,
    measure,
    predict,
    sample_4d_flow,
    two_pi_grid,
)

#: The five phantom cases: the four vessels of the alpha sweep and one physiological five-harmonic waveform at the
#: aortic radius, because a single frequency hides everything a real waveform does to a wall shear stress estimate.
CASES = [(v.name, v.radius, v.flow) for v in DEFAULT_VESSELS] + [("physiological", 1.5e-2, PHYSIOLOGICAL_FLOW)]
SEGMENT_LENGTHS_IN_RADII = 10.0              # the studied segment, in radii: 3 cm on the aorta


def _wave(radius: float, flow) -> Womersley:
    """The closed form of one phantom case."""
    return Womersley.from_flow_rate(radius, BLOOD.nu, TWO_PI, flow, rho=BLOOD.rho)


def _case(name: str) -> tuple[float, tuple]:
    radius, flow = next((r, f) for n, r, f in CASES if n == name)
    return radius, flow


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(_HERE), help="folder for results/ and figures/")
    ap.add_argument("--quick", action="store_true", help="one member per case and one case per sweep: a smoke test")
    ap.add_argument("--members", type=int, default=None, help="ensemble members (3 full, 1 quick)")
    ap.add_argument("--sweep-members", type=int, default=None,
                    help="ensemble members for the two sweeps (2 full, 1 quick): they repeat the same question")
    ap.add_argument("--steps", type=int, default=None, help="Adam steps per member (2000 full, 300 quick)")
    ap.add_argument("--radial", type=int, default=8, help="voxels across the radius of the nominal acquisition")
    ap.add_argument("--noise", type=float, default=0.05, help="velocity noise, as a fraction of the peak")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    members = a.members if a.members is not None else (1 if a.quick else 3)
    sweep_members = a.sweep_members if a.sweep_members is not None else (1 if a.quick else 2)
    steps = a.steps if a.steps is not None else (300 if a.quick else 2000)
    lbfgs = 0 if a.quick else 100
    config = PINNConfig(n_collocation=1000 if a.quick else 2000, adam_steps=steps, lbfgs_steps=lbfgs)

    t_start = time.time()
    result: dict = {"settings": {"members": members, "sweep_members": sweep_members,
                                 "adam_steps": steps, "lbfgs_steps": lbfgs,
                                 "n_collocation": config.n_collocation, "n_radial": a.radial, "noise": a.noise,
                                 "quick": a.quick},
                    "cases": [], "resolution_sweep": [], "noise_sweep": [], "notes": [
                        "synthetic phantom: a rigid straight tube with a Womersley closed-form solution, not patient "
                        "data and not a downloaded benchmark",
                        "every error is relative L2 over the cardiac cycle against the closed form",
                        "the ensemble spread prices initialisation and optimisation, not the model form"]}

    def dump() -> None:
        result["seconds"] = round(time.time() - t_start, 1)
        (out / "results" / "hemodynamics_4dflow.json").write_text(
            json.dumps(result, indent=1, default=float) + "\n")

    # ---------------------------------------------------------------- the five phantom cases
    detail: dict = {}
    for name, radius, flow in CASES:
        wave = _wave(radius, flow)
        four_d = sample_4d_flow(wave, n_radial=a.radial, n_phases=24, n_planes=3, noise=a.noise, seed=1)
        ens = ensemble(wave, four_d, members=members, config=config)
        got = measure(wave, four_d, ens, segment_length=SEGMENT_LENGTHS_IN_RADII * radius)
        got["name"] = name
        got["note"] = next((v.note for v in DEFAULT_VESSELS if v.name == name),
                           "five harmonics: a mean, three harmonics and a systolic peak")
        got["state"] = wave.state()
        result["cases"].append(got)
        if name == "aorta":
            detail = {"wave": wave, "four_d": four_d, "ensemble": ens}
        dump()
        print(f"{name:14s} alpha {wave.alpha:5.2f}  WSS {got['pinn']['wss_relative_error']:.4f} "
              f"(two-point {got['data_only']['wss_relative_error_two_point']:.3f})  "
              f"pressure {got['pinn']['pressure_gradient_relative_error']:.4f}  "
              f"ensemble spread {got['uncertainty']['wss_std_over_peak']:.3f} of peak", flush=True)

    # ---------------------------------------------------------------- the voxel-size sweep
    # n_radial = 8 is the nominal case above; these are the resolutions either side of it
    sweep_vessels = [(DEFAULT_VESSELS[0].name, DEFAULT_VESSELS[0].radius, DEFAULT_VESSELS[0].flow)]
    if not a.quick:
        sweep_vessels.append((DEFAULT_VESSELS[3].name, DEFAULT_VESSELS[3].radius, DEFAULT_VESSELS[3].flow))
    for name, radius, flow in sweep_vessels:
        wave = _wave(radius, flow)
        for n_radial in ((16,) if a.quick else (4, 16, 32)):
            four_d = sample_4d_flow(wave, n_radial=n_radial, n_phases=24, n_planes=3, noise=a.noise, seed=1)
            ens = ensemble(wave, four_d, members=sweep_members, config=config)
            got = measure(wave, four_d, ens, segment_length=SEGMENT_LENGTHS_IN_RADII * radius)
            got["name"] = name
            result["resolution_sweep"].append(got)
            dump()
            print(f"  {name:14s} {n_radial:3d} voxels  gap {got['scales']['wall_gap_m'] * 1e3:5.2f} mm  "
                  f"WSS {got['pinn']['wss_relative_error']:.4f} "
                  f"(two-point {got['data_only']['wss_relative_error_two_point']:.3f})", flush=True)

    # ---------------------------------------------------------------- the noise sweep
    wave = _wave(DEFAULT_VESSELS[3].radius, DEFAULT_VESSELS[3].flow)
    for noise in ((0.10,) if a.quick else (0.01, 0.10, 0.20)):
        four_d = sample_4d_flow(wave, n_radial=a.radial, n_phases=24, n_planes=3, noise=noise, seed=1)
        ens = ensemble(wave, four_d, members=sweep_members, config=config)
        got = measure(wave, four_d, ens, segment_length=SEGMENT_LENGTHS_IN_RADII * wave.radius)
        got["name"] = DEFAULT_VESSELS[3].name
        got["settings"] = {"noise": noise, "n_radial": a.radial}
        result["noise_sweep"].append(got)
        dump()
        print(f"  aorta  noise {noise:4.2f}  WSS {got['pinn']['wss_relative_error']:.4f} "
              f"(two-point {got['data_only']['wss_relative_error_two_point']:.3f})", flush=True)

    _figures(out, result, detail)
    result["seconds"] = round(time.time() - t_start, 1)
    dump()
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "notes")}, indent=1)[:2000])
    return result


# -------------------------------------------------------------------------------------------------- figures
def _figures(out: Path, result: dict, detail: dict | None = None) -> None:
    """Four figures, all drawn from the record: they can be redrawn without refitting anything.

    ``detail`` is the one case kept in memory -- the aorta's acquisition and its fitted ensemble -- for the profile
    figure, which needs the network evaluated on a fine radial grid.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from pinneapple_tools.visualization.style import use_theme
        use_theme("report")                       # one theme for every figure of the library (#918)
    except ImportError:
        return
    cases = result["cases"]
    if not cases:
        return
    blue, orange, green, dark = "#0081be", "#a23a00", "#00814f", "#262626"

    # 1. the measured profile and the fit, and the last 15 % of the radius where the gradient lives
    if detail:
        wave, four_d, ens = detail["wave"], detail["four_d"], detail["ensemble"]
        ys, vs = four_d.profiles()
        y_line = np.linspace(0.0, 1.0, 240)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for k, phase in enumerate((0, 8, 16)):
            colour = (blue, orange, green)[k]
            tau = two_pi_grid(24)[phase]
            truth = wave.velocity(y_line, tau)
            _, fitted, _ = predict(ens["members"][0][0], y_line, np.full(y_line.size, tau))
            fitted = fitted * float(ens["members"][0][0].peak_velocity)
            for ax in axes:
                ax.plot(y_line, truth, "-", color=colour, lw=1.8, label=f"Womersley, t = {phase / 24:.2f} T")
                ax.plot(y_line, fitted, "--", color=colour, lw=1.0)
                ax.plot(ys, vs[:, phase], "o", color=colour, ms=5, mfc="white")
        axes[0].set(xlabel="r / R", ylabel="axial velocity [m/s]", title="the phantom, and what the fit does with 8 voxels")
        axes[1].set(xlabel="r / R", ylabel="axial velocity [m/s]", xlim=(0.85, 1.0),
                    title="the last 15 % of the radius: the wall is never sampled")
        axes[0].legend(fontsize=7, title="circle = measurement\ndashed = network", title_fontsize=7)
        axes[1].axvline(1.0, color=dark, lw=0.9, ls=":")
        top = float(np.max(vs))
        axes[1].annotate(f"first sample {1e3 * four_d.wall_gap:.2f} mm from the wall",
                         xy=(1.0 - four_d.wall_gap / wave.radius, 0.62 * top),
                         xytext=(0.858, 0.92 * top), fontsize=7,
                         arrowprops={"arrowstyle": "->", "lw": 0.7})
        axes[1].annotate(f"Stokes layer {1e3 * wave.stokes_layer:.2f} mm",
                         xy=(1.0 - wave.stokes_layer / wave.radius, 0.10 * top),
                         xytext=(0.858, 0.42 * top), fontsize=7,
                         arrowprops={"arrowstyle": "->", "lw": 0.7})
        fig.tight_layout()
        fig.savefig(out / "figures" / "profiles.png", dpi=120)
        plt.close(fig)

    # 2. the wall shear stress waveform: closed form, network with its ensemble spread, and the data-only estimate
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, case in zip(axes.ravel(), cases, strict=False):
        w = case["waveforms"]
        cycle = np.asarray(w["tau"]) / TWO_PI
        ax.fill_between(cycle, np.subtract(w["wss_pinn"], 2 * np.asarray(w["wss_std"])),
                        np.add(w["wss_pinn"], 2 * np.asarray(w["wss_std"])), color=blue, alpha=0.22, lw=0)
        ax.plot(cycle, w["wss_truth"], "-", color=dark, lw=1.8, label="Womersley")
        ax.plot(cycle, w["wss_pinn"], "-", color=blue, lw=1.2, label="network (mean, +/- 2 sd)")
        ax.plot(cycle, w["wss_two_point"], ":", color=orange, lw=1.3, label="two-point difference")
        ax.set_title(f"{case['name']}, alpha = {case['scales']['alpha']:.1f}", fontsize=9)
        ax.set_ylabel("WSS [Pa]")
        ax.text(0.03, 0.05, f"network {100 * case['pinn']['wss_relative_error']:.1f} %\n"
                            f"two-point {100 * case['data_only']['wss_relative_error_two_point']:.0f} %",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.9})
    axes.ravel()[-1].axis("off")
    for ax in axes[-1][:2]:
        ax.set_xlabel("t / T")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Wall shear stress over one cardiac cycle, against Womersley's closed form")
    fig.tight_layout()
    fig.savefig(out / "figures" / "wss_waveforms.png", dpi=120)
    plt.close(fig)

    # 3. error against the Womersley number
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    alphas = [c["scales"]["alpha"] for c in cases]
    ax.semilogy(alphas, [max(c["pinn"]["wss_relative_error"], 1e-4) for c in cases], "o-", color=blue,
                label="network, wall shear stress")
    ax.semilogy(alphas, [max(c["pinn"]["pressure_gradient_relative_error"], 1e-4) for c in cases], "^-",
                color=green, label="network, pressure gradient")
    ax.semilogy(alphas, [max(c["data_only"]["wss_relative_error_two_point"], 1e-4) for c in cases], "s--",
                color=orange, label="two-point difference, wall shear stress")
    for ax_x, case in zip(alphas, cases, strict=False):
        ax.annotate(f"{case['name']}\n{case['scales']['gap_over_stokes_layer']:.1f} gaps",
                    (ax_x, max(case["pinn"]["wss_relative_error"], 1e-4)), textcoords="offset points",
                    xytext=(5, -16), fontsize=6.5, color="#555555")
    ax.set(xlabel="Womersley number alpha = R sqrt(omega / nu)", ylabel="relative L2 over the cycle",
           ylim=(5e-3, 3.0), title="what the network recovers, and what the difference of two voxels recovers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "errors_vs_alpha.png", dpi=120)
    plt.close(fig)

    # 4. error against the gap the estimate has to extrapolate across
    rows = result["resolution_sweep"] + [c for c in cases if c["acquisition"]["n_radial"] < 0]
    if rows:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        names = sorted({s["name"] for s in rows})
        for k, name in enumerate(names):
            group = sorted((s for s in rows if s["name"] == name), key=lambda s: s["acquisition"]["n_radial"])
            xs = [1e3 * s["acquisition"]["wall_gap_m"] for s in group]
            point = ("o-", "^-")[k % 2]
            dash = ("s--", "v--")[k % 2]
            ax.loglog(xs, [max(s["pinn"]["wss_relative_error"], 1e-4) for s in group], point,
                      color=(blue, green)[k % 2], label=f"network, {name}")
            ax.loglog(xs, [max(s["data_only"]["wss_relative_error_two_point"], 1e-4) for s in group], dash,
                      color=(orange, "#8a6d00")[k % 2], label=f"two-point, {name}")
        layer = cases[0]["scales"]["stokes_layer_m"] * 1e3
        ax.axvline(layer, color=dark, lw=0.9, ls=":")
        ax.annotate(f"R / alpha = {layer:.2f} mm\nthe boundary layer", xy=(layer, 0.02), xytext=(0.28 * layer, 0.008),
                    fontsize=7, arrowprops={"arrowstyle": "->", "lw": 0.7})
        ax.set(xlabel="distance from the first sample to the wall [mm]", ylabel="relative L2 over the cycle",
               title="wall shear stress against the gap the estimate extrapolates across")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "resolution.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
