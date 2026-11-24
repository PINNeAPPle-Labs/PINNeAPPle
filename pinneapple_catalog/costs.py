"""Measured cost of every method in the catalog (#52, roadmap D1).

``pinneapple_analysis.cost`` answers "how much does this operation cost" for one callable. This module applies it to
the catalog: one probe per method, run here, on this machine, and stored in ``method_costs.json`` next to the
catalog itself. Every number in that file came out of a run, never out of a formula:

- **time**: median of repeated runs at each problem size, with the interquartile range as the spread;
- **memory**: peak increase over the starting footprint, and ``mem_source`` says how it was sampled (``rss`` with
  psutil, ``"tracemalloc"`` without it -- which sees Python and NumPy only);
- **FLOPs**: only where a PyTorch path makes them countable, and ``flops_source`` says whether the count is exact
  (``flop_counter``) or a lower bound (``profiler``); ``null`` is a real answer, not a gap to fill with a guess;
- **scaling exponent**: the log-log slope of cost against problem size, with a bootstrap confidence interval, and
  the verdict against the complexity the probe declares.

The **machine** is recorded with the numbers (CPU, cores, platform, Python and torch versions), because a wall time
without its machine cannot be compared with anything. Two runs on two machines are two different measurements;
:func:`compare_with_baseline` refuses to compare across machines for exactly that reason.

Coverage is a first-class value, not an afterthought: :func:`coverage` returns one row per method -- ``measured``, or
``unmeasured`` **with the reason** (an external solver that is not installed, a GPU kernel, a method whose entry
point needs a case the probe cannot build). ``scripts/measure_method_costs.py --check`` fails when a catalog method
is missing from the file, so a new solver cannot enter the catalog without either a probe or a stated reason.

What a probe measures, exactly: the ``probe`` field names the class or function that ran, and the size axis says what
``n`` means for that method (grid points per side, particles, samples, layers). A family (SPH / ISPH / DFSPH, the
signal decompositions) records which member of the family was timed, so the number is never read as the whole family.

    python scripts/measure_method_costs.py --only S4 S19      # measure a couple of methods, print the table
    python scripts/measure_method_costs.py                    # measure everything with a probe, rewrite the file
    python scripts/measure_method_costs.py --check            # gates: every method measured or explained
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pinneapple_analysis.cost import ScalingStudy, scaling_study
from pinneapple_core.errors import BadValue, UnknownName

from . import methods as _methods

HERE = Path(__file__).resolve().parent
COSTS_PATH = HERE / "method_costs.json"

__all__ = ["Probe", "PROBES", "UNMEASURED", "all_costs", "compare_with_baseline", "coverage", "load_costs",
           "machine", "measure_all", "method_cost"]


# --------------------------------------------------------------------------- the machine
def machine() -> dict[str, Any]:
    """What has to travel with a wall time for it to mean anything: the CPU, the cores, the platform, the versions."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception:  # torch absent: the numpy-only part of the library still measures
        info["torch"] = None
        info["cuda"] = False
    try:
        import psutil
        info["memory_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        info["memory_bytes"] = None
    return info


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- the probes
@dataclass(frozen=True)
class Probe:
    """One method's measuring stick.

    ``build(n)`` returns the callable that does the work at problem size ``n`` (its setup is not timed, as in
    :func:`pinneapple_analysis.cost.scaling_study`); ``declared`` is the complexity the code claims, so the fitted
    exponent has something to be checked against; ``what`` and ``size_axis`` say exactly what was run.
    """
    method_id: str
    what: str
    size_axis: str
    sizes: Sequence[int]
    build: Callable[[int], Callable[[], Any]]
    declared: str | None = None
    repeats: int = 5


def _probe(method_id, what, size_axis, sizes, declared=None, repeats=5):
    def deco(build):
        return Probe(method_id, what, size_axis, tuple(sizes), build, declared, repeats)
    return deco


# --------------------------------------------------------------------------- solvers
@_probe("S4", "pinneapple_simulation.numerical_solvers.spectral.poisson_periodic_fft",
        "n (grid points per side, periodic square)", [64, 128, 256, 512], declared="n^2")
def _probe_s4(n):
    import torch

    from pinneapple_simulation.numerical_solvers.spectral import poisson_periodic_fft
    f = torch.randn(n, n, dtype=torch.float64)
    return lambda: poisson_periodic_fft(f)


@_probe("S19", "pinneapple_simulation.numerical_solvers.fft.FFTSolver.forward (the FFT member of the family)",
        "n (signal samples)", [2 ** 16, 2 ** 18, 2 ** 20, 2 ** 21], declared="n log n")
def _probe_s19(n):
    import torch

    from pinneapple_simulation.numerical_solvers.fft import FFTSolver
    x = torch.randn(n)
    solver = FFTSolver()
    return lambda: solver(x)


@_probe("S13", "pinneapple_simulation.numerical_solvers.beam_bvp_fdm.BeamBVPFDMSolver.forward",
        "n (1-D grid points along the beam)", [200, 400, 800, 1600], declared="n")
def _probe_s13(n):
    from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import BeamBVPFDMSolver
    solver = BeamBVPFDMSolver(nx=n)
    return lambda: solver(L_m=2.0, E_Pa=210e9, I_m4=1e-6, q_N_per_m=1e3)


@_probe("S15", "pinneapple_simulation.numerical_solvers.bekker_wong.BekkerWongSolver.forward",
        "n (slip ratios evaluated in one call)", [256, 512, 1024, 2048], declared="n")
def _probe_s15(n):
    import torch

    from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver
    solver = BekkerWongSolver()
    slip = torch.linspace(0.05, 0.35, n)
    sinkage = torch.full_like(slip, 0.02)
    x = torch.stack([slip, sinkage], dim=1)
    return lambda: solver(x)


@_probe("S34", "pinneapple_simulation.electromagnetics.mie.mie",
        "x (size parameter; the series needs ~x + 4x^(1/3) terms)", [50, 200, 800, 3200], declared="n")
def _probe_s34(n):
    from pinneapple_simulation.electromagnetics.mie import mie
    return lambda: mie(1.5 + 0.01j, float(n))


@_probe("S35", "pinneapple_simulation.electromagnetics.waves.multilayer",
        "n (layers in the stack)", [16, 64, 256, 1024], declared="n")
def _probe_s35(n):
    from pinneapple_simulation.electromagnetics.waves import multilayer
    n_layers = [1.5] * n
    d_layers = [0.1] * n
    return lambda: multilayer(n_layers, d_layers, 0.55)


@_probe("S33", "pinneapple_simulation.electromagnetics.statics.infinite_wire_B",
        "n (field points)", [16384, 65536, 262144, 1048576], declared="n")
def _probe_s33(n):
    import numpy as np

    from pinneapple_simulation.electromagnetics.statics import infinite_wire_B
    pts = np.random.default_rng(0).random((n, 3))
    return lambda: infinite_wire_B(pts, 1.0)


@_probe("S28", "pinneapple_simulation.numerical_solvers.stable_fluids.StableFluids2D (one full step)",
        "n (cells per side)", [64, 128, 256, 512], declared="n^2")
def _probe_s28(n):
    import numpy as np

    from pinneapple_simulation.numerical_solvers.stable_fluids import StableFluids2D
    sim = StableFluids2D(n, n, viscosity=1e-4)
    vel = np.zeros((2, n, n))
    vel[0] = np.sin(sim.X) * np.cos(sim.Y)
    vel[1] = -np.cos(sim.X) * np.sin(sim.Y)
    state = sim.initial_state(vel)
    return lambda: sim.step(state, dt=0.01)


@_probe("S26", "pinneapple_simulation.numerical_solvers.shallow_water_fv.ShallowWater2D (one step)",
        "n (cells per side)", [64, 128, 256, 512], declared="n^2")
def _probe_s26(n):
    import numpy as np

    from pinneapple_simulation.numerical_solvers.shallow_water_fv import ShallowWater2D
    sim = ShallowWater2D(n, n, dx=1.0 / n)
    x, y = sim.centers()
    sim.h = 1.0 + 0.1 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
    return lambda: sim.step()


@_probe("S27", "pinneapple_simulation.numerical_solvers.etdrk4.ETDRK4 (one step of the stiff semilinear problem)",
        "n (points on the periodic grid)", [512, 1024, 2048, 4096], declared="n log n")
def _probe_s27(n):
    import numpy as np

    from pinneapple_simulation.numerical_solvers.etdrk4 import ETDRK4, fourier_wavenumbers
    k = fourier_wavenumbers(n, 2 * np.pi)
    sim = ETDRK4(-0.1 * k ** 2, lambda u, t: -u ** 3, 1e-2)
    u = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    return lambda: sim.step(u, 0.0)


@_probe("S9", "pinneapple_simulation.numerical_solvers.discrete_time_pinn.DiscreteTimePINNSolver "
        "(one implicit step, trained to a fixed iteration cap)",
        "n (iterations of the implicit step, q = 32 fixed)", [5, 10, 20, 40], declared="n", repeats=3)
def _probe_s9(n):
    import numpy as np
    import torch

    from pinneapple_simulation.numerical_solvers.discrete_time_pinn import DiscreteTimePINNSolver
    x = torch.tensor(np.linspace(-1.0, 1.0, 32), dtype=torch.float32)
    u = torch.sin(torch.pi * x)
    solver = DiscreteTimePINNSolver(pde="burgers_1d", q=32, max_iter=int(n), seed=0)
    return lambda: solver(x, u, 0.01)


UNMEASURED: dict[str, str] = {
    "S2": "the FEM solver needs a ProblemSpec preset; measured through S1's probe in the benchmark (see "
          "pinneapple_analysis.benchmark), not on its own here",
    "S3": "same as S2: the finite-volume solver runs from a ProblemSpec preset",
    "S1": "needs a ProblemSpec preset (geometry and boundary conditions per case); the benchmark harness measures "
          "it on real cases instead",
    "S5": "the LBM needs a spec plus the number of steps to be meaningful; a fixed-step probe would measure the "
          "step count, not the method",
    "S6": "SPH / ISPH / DFSPH are particle methods whose cost is set by the particle count and the time step; "
          "a probe here would need a settled configuration to be comparable",
    "S7": "the Kansa RBF solver runs from a ProblemSpec plus interior/boundary point sets",
    "S8": "X-TFC trains a network per call: its cost is a training budget, measured by the trainer probes (T*)",
    "S10": "rfk45 stiff kinetics: the module exposes the steppers per problem, no single entry point to time",
    "S11": "TVD advection runs from a ProblemSpec preset",
    "S12": "immersed boundary wraps an FDM flow solve (spec-driven, see S1)",
    "S14": "the eddy-current FDM needs a coil/material field set; the probe would measure the field builder",
    "S16": "(nbody_rebound) needs the REBOUND package, which is not installed here",
    "S17": "JAX-CFD needs jax, which is not installed here",
    "S18": "particle dynamics: MPM / rigid bodies / MD each need their own configuration",
    "S20": "co-simulation: the cost is the coupling graph the user builds, measured by the case itself",
    "S21": "state estimation: cost is set by the state dimension and the number of steps of the run",
    "S22": "1-D process components: the network is user-built; no single operation to time",
    "S23": "explicit equation system: cost is the size of the system the user writes",
    "S24": "external solvers (OpenFOAM, FEniCS, ANSYS, MATLAB, Modelica, MuJoCo) are not installed here",
    "S25": "telemetry conditioning: the cost is per-sample and dominated by the caller's loop",
    "S29": "2-D magnetostatics FEM needs a mesh and materials; measured through its validation case",
    "S30": "time-harmonic eddy currents: same as S29",
    "S31": "surface-PM machine model: cost is set by the harmonic set of the run",
    "S32": "2-D electrostatics FEM: same as S29",
    "S36": "Jiles-Atherton hysteresis: the cost is the applied-field path the user supplies",
    "S37": "dispersive media: coefficient arithmetic, measured by the FDTD run that uses it",
    "S38": "Maxwell checks: cost is the field the user hands in",
    "S39": "topology optimisation: the cost is iterations x cells, measured by each design study",
    "S40": "Yee FDTD: cost is grid x steps; a probe would need a canonical case kept with the campaign",
    "S41": "the near-to-far transform runs on an FDTD box (see S40)",
    "S42": "first passage: cost is the number of paths the user asks for",
    "S44": "Fisher information: needs a model and a design matrix from the caller",
    "S45": "early warning: per-window cost, dominated by the series length the user passes",
    "S46": "Koopman: needs snapshot data from the caller",
    "S47": "NVIDIA Newton needs the Warp stack and a GPU",
    "S48": "the PICT local solver needs the case's grid and boundary set",
    "S49": "PICT needs CUDA",
    "S50": "NVIDIA Warp kernels need the warp package and a GPU",
}

# The probes above run anywhere, on a laptop, without a case the user has to build first.
PROBES: dict[str, Probe] = {p.method_id: p for p in (
    _probe_s4, _probe_s19, _probe_s13, _probe_s15, _probe_s34, _probe_s35, _probe_s33, _probe_s28, _probe_s26,
    _probe_s27, _probe_s9,
) if p.method_id in {m.id for m in _methods.METHODS}}


# --------------------------------------------------------------------------- measuring
def _study(probe: Probe, repeats: int | None = None) -> ScalingStudy:
    return scaling_study(probe.build, list(probe.sizes), name=probe.what,
                         repeats=probe.repeats if repeats is None else repeats,
                         declared=probe.declared, memory=True)


def _record(probe: Probe, study: ScalingStudy) -> dict[str, Any]:
    per_size = []
    for n, rec in zip(study.sizes, study.records, strict=True):
        per_size.append({
            "n": n, "time_s": rec.time_s, "time_min_s": rec.time_min_s, "time_iqr_s": rec.time_iqr_s,
            "mem_bytes": rec.peak_mem_bytes, "flops": rec.flops, "n_runs": rec.n_runs,
        })
    fits = {q: {"exponent": f.exponent, "ci": list(f.ci), "r2": f.r2, "coefficient": f.coefficient}
            for q, f in study.fits.items()}
    time_fit = study.fits["time"]
    return {
        "status": "measured",
        "probe": probe.what,
        "size_axis": probe.size_axis,
        "sizes": list(study.sizes),
        "declared": probe.declared,
        "verdict": time_fit.verdict,
        "exponent": time_fit.exponent,
        "exponent_ci": list(time_fit.ci),
        "r2": time_fit.r2,
        "fits": fits,
        "per_size": per_size,
        "mem_source": next((r.mem_source for r in study.records if r.mem_source), ""),
        "flops_source": next((r.flops_source for r in study.records if r.flops_source), ""),
        "median_time_s": float(sorted(r.time_s for r in study.records)[len(study.records) // 2]),
        "peak_mem_bytes": int(max(r.peak_mem_bytes or 0 for r in study.records)),
        "flops": next((int(r.flops) for r in study.records if r.flops), None),
    }


def measure_all(only: Sequence[str] | None = None, *, repeats: int | None = None,
                path: Path = COSTS_PATH) -> dict[str, Any]:
    """Measure every method that has a probe and write ``path``: the measured records plus the reasons for the rest.

    ``only`` restricts the run to some ids, leaving the other measured records as they are on disk (so one solver can
    be remeasured without touching the others).
    """
    current = load_costs(path)
    methods = current.get("methods", {})
    chosen = [m for m in _methods.METHODS if m.category in ("solver", "training")]
    todo = [p for mid, p in sorted(PROBES.items()) if (only is None or mid in only)]
    for probe in todo:
        study = _study(probe, repeats=repeats)
        methods[probe.method_id] = _record(probe, study)
        print(f"  {probe.method_id:<6} {study.fits['time'].exponent:5.2f}  "
              f"{min(r.time_s for r in study.records) * 1e3:8.2f} ms .. "
              f"{max(r.time_s for r in study.records) * 1e3:9.2f} ms  {probe.what}")
    for m in chosen:
        if m.id in methods and methods[m.id].get("status") == "measured":
            continue
        if m.id in PROBES:
            continue
        methods[m.id] = {"status": "unmeasured", "reason": UNMEASURED.get(
            m.id, "no probe written for this method yet")}
    doc = {
        "what": "measured cost per catalog method (issue #52, roadmap D1); every number came out of a run",
        "how": "python scripts/measure_method_costs.py [--only ID ...]",
        "machine": machine(),
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "methods": methods,
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- reading
def load_costs(path: Path = COSTS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"methods": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def method_cost(method_id: str, path: Path = COSTS_PATH) -> dict[str, Any]:
    """One method's record (``measure_all`` does not invent an entry for an id the catalog does not have)."""
    _methods.get_method(method_id)
    try:
        return load_costs(path)["methods"][method_id]
    except KeyError:
        raise UnknownName(f"method {method_id!r} has no cost record yet",
                          fix="run python scripts/measure_method_costs.py (or --only "
                              f"{method_id} to measure just this one)") from None


def all_costs(path: Path = COSTS_PATH) -> dict[str, dict[str, Any]]:
    return dict(load_costs(path).get("methods", {}))


def coverage(path: Path = COSTS_PATH) -> list[dict[str, Any]]:
    """One row per catalog method (solvers and training methods): measured, or unmeasured and why."""
    costs = all_costs(path)
    rows = []
    for m in _methods.METHODS:
        if m.category not in ("solver", "training"):
            continue
        rec = costs.get(m.id)
        if not rec:
            rows.append({"id": m.id, "name": m.name, "status": "missing", "reason": "not in method_costs.json"})
        elif rec.get("status") == "measured":
            rows.append({"id": m.id, "name": m.name, "status": "measured", "probe": rec.get("probe"),
                         "exponent": rec.get("exponent"), "median_time_s": rec.get("median_time_s")})
        else:
            rows.append({"id": m.id, "name": m.name, "status": "unmeasured", "reason": rec.get("reason", "")})
    return rows


def compare_with_baseline(current: dict[str, Any], baseline: dict[str, Any], *, flops_factor: float = 1.05,
                          time_factor: float = 1.5, mem_factor: float = 1.5) -> list[dict[str, Any]]:
    """Regressions between two ``method_costs.json`` documents on the same machine (issue #53's gate).

    FLOPs gate tightly (``flops_factor`` 1.05: they are deterministic, so a 6 % rise is a real 6 % rise); time and
    memory gate loosely (1.5: the same code on the same machine still moves by tens of percent between runs). A
    different machine is refused rather than compared: the same code measured on two CPUs is two measurements, not a
    regression.
    """
    if current.get("machine", {}).get("platform") != baseline.get("machine", {}).get("platform"):
        raise BadValue(
            "the two cost tables were measured on different machines",
            fix="remeasure on the machine that produced the baseline "
                "(scripts/measure_method_costs.py), or take that machine's file as the baseline")
    rows: list[dict[str, Any]] = []
    for mid, cur in current.get("methods", {}).items():
        base = baseline.get("methods", {}).get(mid)
        if not base or cur.get("status") != "measured" or base.get("status") != "measured":
            continue
        for quantity, b, c, factor in (("flops", base.get("flops"), cur.get("flops"), flops_factor),
                                       ("time_s", base.get("median_time_s"), cur.get("median_time_s"), time_factor),
                                       ("peak_mem_bytes", base.get("peak_mem_bytes"), cur.get("peak_mem_bytes"),
                                        mem_factor)):
            if b and c and c > b * factor:
                rows.append({"method_id": mid, "probe": cur.get("probe", ""), "quantity": quantity,
                             "baseline": float(b), "current": float(c), "ratio": float(c) / float(b),
                             "factor": factor})
    return rows
