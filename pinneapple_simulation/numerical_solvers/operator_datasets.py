"""Operator-learning benchmark datasets: 1-D Burgers, 2-D Darcy, 2-D Navier-Stokes (vorticity).

The three problems and input distributions of Li et al. (2021), "Fourier Neural Operator for Parametric Partial
Differential Equations" (ICLR), solved with the reference solvers of :mod:`.operator_benchmarks`. This module holds the
dataset recipes of the zoo datasets ``burgers_1d_li2021``, ``darcy_2d_li2021`` and
``navier_stokes_2d_li2021_nu1e-3`` (``pp.datasets``):

* :data:`BURGERS`, :data:`DARCY`, :data:`NS`: the problem parameters, resolutions, time steps, seeds and split sizes;
* ``burgers_inputs`` / ``darcy_inputs`` / ``ns_inputs`` and ``solve_burgers`` / ``solve_darcy`` / ``solve_ns``;
* :func:`convergence`: the resolution / time-step study each dataset card quotes;
* :func:`generate`: writes ``<split>[-NN].npz`` files and ``meta.json`` (``scripts/zoo/operator_data.py`` is the CLI).

Every input is drawn in the parent process from a fixed seed, so the data do not depend on the number of workers.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import operator_benchmarks as ob

__all__ = ["BURGERS", "DARCY", "NS", "SHARD_ROWS", "fourier_pad", "burgers_inputs", "darcy_inputs", "ns_inputs",
           "block_average", "darcy_set", "solve_burgers", "solve_darcy", "solve_ns", "rel_l2", "write_npz", "convergence", "generate"]

# --------------------------------------------------------------------------- problem definitions
# Burgers: u0 ~ N(0, 625 (-lap + 25 I)^-2) on the periodic [0, 1), nu = 0.1, map u0 -> u(., 1) (Li et al. 2021, 5.1)
BURGERS = dict(n=1024, nu=0.1, t_end=1.0, alpha=2.0, tau=5.0, sigma=25.0, dt=1e-4, n_save=10, seed=20211,
               n_train=1000, n_test=200)
# Darcy: a = psi(g), g ~ N(0, (-lap + 9 I)^-2), psi = 12 on g >= 0 else 3; f = 1; u = 0 on the boundary (5.2)
DARCY = dict(n=128, refine=4, alpha=2.0, tau=3.0, high=12.0, low=3.0, f=1.0, seed=20212, n_train=1000, n_test=200)
# Navier-Stokes: w0 ~ N(0, 7^(3/2) (-lap + 49 I)^-2.5), Li forcing, nu = 1e-3, t in [0, 50] (5.3)
NS = dict(n_solve=128, n_store=64, nu=1e-3, t_end=50.0, n_save=50, dt=5e-2, alpha=2.5, tau=7.0, seed=20213,
          n_train=1000, n_test=200)


def burgers_inputs(n_samples: int, n: int, seed: int) -> np.ndarray:
    p = BURGERS
    u = ob.periodic_grf(n_samples, (n,), alpha=p["alpha"], tau=p["tau"], sigma=p["sigma"], seed=seed)
    return fourier_pad(u, n, axes=(1,))                         # drop the Nyquist mode: exactly band-limited


def fourier_pad(u: np.ndarray, n_new: int, axes) -> np.ndarray:
    """Band-limited (trigonometric) interpolation of periodic grid data onto ``n_new`` points along each of ``axes``;
    exact for fields whose Fourier content is below the coarse Nyquist mode (the Nyquist mode is dropped)."""
    for a in axes:
        n = u.shape[a]
        k = np.fft.fftfreq(n, 1.0 / n).astype(int)
        keep = np.flatnonzero(np.abs(k) < n // 2)
        uh = np.take(np.fft.fft(u, axis=a), keep, axis=a)
        shape = list(u.shape)
        shape[a] = n_new
        out = np.zeros(shape, dtype=complex)
        idx = [slice(None)] * u.ndim
        idx[a] = k[keep] % n_new
        out[tuple(idx)] = uh
        u = np.real(np.fft.ifft(out, axis=a)) * (n_new / n)
    return u


def solve_burgers(u0: np.ndarray, dt: float | None = None) -> np.ndarray:
    p = BURGERS
    _, u = ob.burgers_1d(u0, p["nu"], p["t_end"], dt=dt or p["dt"], n_save=p["n_save"])
    return u                                                    # (batch, n_save + 1, n)


def darcy_inputs(n_samples: int, n: int, seed: int, *, tau: float | None = None, high: float | None = None,
                 low: float | None = None) -> np.ndarray:
    """Coefficients of the Darcy benchmark; ``tau`` (inverse correlation length), ``high`` and ``low`` override the
    dataset's values to draw shifted inputs (other length scale, other contrast) from the same family."""
    p = DARCY
    g = ob.periodic_grf(n_samples, (n, n), alpha=p["alpha"], tau=p["tau"] if tau is None else tau, seed=seed)
    return ob.threshold_coefficient(g, high=p["high"] if high is None else high, low=p["low"] if low is None else low)


def block_average(x: np.ndarray, factor: int) -> np.ndarray:
    """Cell averages on a grid ``factor`` times coarser over the last two axes (``(..., n, n)``, ``n % factor == 0``):
    the coarse-grid version of a cell-averaged field (``u`` of the dataset) or of a coefficient."""
    x = np.asarray(x)
    n0, n1 = x.shape[-2:]
    if n0 % factor or n1 % factor:
        raise ValueError(f"grid {n0} x {n1} is not divisible by {factor}")
    return x.reshape(*x.shape[:-2], n0 // factor, factor, n1 // factor, factor).mean(axis=(-3, -1))


def solve_darcy(a: np.ndarray, refine: int | None = None) -> np.ndarray:
    """Solve on the grid refined ``refine`` times (the piecewise-constant coefficient is represented exactly) and
    return the cell averages of u on the coefficient's own grid."""
    r = refine or DARCY["refine"]
    out = []
    for ai in (a if a.ndim == 3 else a[None]):
        n = ai.shape[0]
        u = ob.darcy_2d(np.kron(ai, np.ones((r, r))), DARCY["f"])
        out.append(u.reshape(n, r, n, r).mean(axis=(1, 3)))
    return np.stack(out)


def ns_inputs(n_samples: int, n: int, seed: int) -> np.ndarray:
    p = NS
    w = ob.periodic_grf(n_samples, (n, n), alpha=p["alpha"], tau=p["tau"], seed=seed)
    return fourier_pad(w, n, axes=(1, 2))                       # drop the Nyquist modes: exactly band-limited


def solve_ns(w0: np.ndarray, n_solve: int | None = None, dt: float | None = None) -> np.ndarray:
    """Solve on ``n_solve``^2 (inputs given on a coarser grid are Fourier-padded) and return the point values on the
    stored ``n_store``^2 grid at t = 0, 1, ..., t_end."""
    p = NS
    n_solve = n_solve or p["n_solve"]
    if w0.shape[-1] != n_solve:
        w0 = fourier_pad(w0, n_solve, axes=(1, 2))
    _, w = ob.navier_stokes_2d_vorticity(w0, p["nu"], p["t_end"], forcing=ob.li_forcing(n_solve),
                                         dt=dt or p["dt"], n_save=p["n_save"])
    return ob.spectral_downsample(w, n_solve // p["n_store"], axes=(2, 3))


def rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 error of ``a`` against reference ``b`` (all axes but the first)."""
    ax = tuple(range(1, a.ndim))
    return np.sqrt(((a - b) ** 2).sum(ax) / (b ** 2).sum(ax))


def _pmap(fn, chunks, workers):
    if workers <= 1:
        return [fn(c) for c in chunks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, chunks))


# --------------------------------------------------------------------------- generation
SHARD_ROWS = {"burgers_1d_li2021": 0, "darcy_2d_li2021": 0, "navier_stokes_2d_li2021_nu1e-3": 100}


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Deflated npz with fixed zip timestamps (the SHA-256 depends only on the data)."""
    import io
    import zipfile
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in sorted(zin.infolist(), key=lambda i: i.filename):
            zi = zipfile.ZipInfo(item.filename, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, zin.read(item.filename))


def _write_splits(out: Path, name: str, cols: dict[str, np.ndarray], n_train: int, meta: dict) -> None:
    """Rows [0, n_train) are ``train``, the rest ``test`` (inputs are i.i.d., so a contiguous split is a random
    split); ``<split>.npz``, or ``<split>-NN.npz`` shards of ``SHARD_ROWS[name]`` rows."""
    d = out / name
    n = len(next(iter(cols.values())))
    files = []
    for split, rows in (("train", range(0, n_train)), ("test", range(n_train, n))):
        shard = SHARD_ROWS[name] or len(rows)
        parts = [rows[i:i + shard] for i in range(0, len(rows), shard)]
        for j, r in enumerate(parts):
            fn = f"{split}.npz" if len(parts) == 1 else f"{split}-{j:02d}.npz"
            write_npz(d / fn, {f"{split}__{c}": v[r.start:r.stop] for c, v in cols.items()})
            files.append(fn)
    meta = {**meta, "name": name, "files": files, "rows": {"train": n_train, "test": n - n_train}}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name}: {len(files)} files, {sum((d / f).stat().st_size for f in files) / 1e6:.1f} MB", flush=True)


def _chunks(x: np.ndarray, size: int) -> list[np.ndarray]:
    return [x[i:i + size] for i in range(0, len(x), size)]


def darcy_set(n_samples: int, seed: int, *, n: int | None = None, tau: float | None = None,
              high: float | None = None, low: float | None = None, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(a, u)`` pairs of the Darcy benchmark drawn and solved exactly as the zoo dataset ``darcy_2d_li2021``
    (same solver, refinement and source), optionally from a shifted input distribution: ``tau`` (correlation
    length ~ 1 / tau; the dataset uses 3), ``high`` / ``low`` (the two coefficient values; 12 / 3)."""
    n = n or DARCY["n"]
    a = darcy_inputs(n_samples, n, seed, tau=tau, high=high, low=low)
    u = np.concatenate(_pmap(solve_darcy, _chunks(a, max(1, -(-n_samples // max(workers, 1)))), workers))
    return a, u


def generate(out: Path, workers: int = 1, only: str | None = None) -> None:
    """Generate the three datasets (or ``only`` one) into ``out/<name>/`` (resumable for Navier-Stokes)."""
    t0 = time.time()
    if only in (None, "burgers"):
        p = BURGERS
        n = p["n_train"] + p["n_test"]
        u0 = burgers_inputs(n, p["n"], p["seed"])
        u = np.concatenate(_pmap(solve_burgers, _chunks(u0, 25), workers))
        assert np.all(np.isfinite(u)) and np.allclose(u[:, 0], u0)
        _write_splits(out, "burgers_1d_li2021", {"index": np.arange(n, dtype=np.int32),
                                                 "u0": u0.astype(np.float32), "u": u.astype(np.float32)},
                      p["n_train"], {"params": p})
        print(f"  burgers done {time.time() - t0:.0f}s", flush=True)
    if only in (None, "darcy"):
        p = DARCY
        n = p["n_train"] + p["n_test"]
        a = darcy_inputs(n, p["n"], p["seed"])
        u = np.concatenate(_pmap(solve_darcy, _chunks(a, 10), workers))
        assert np.all(np.isfinite(u)) and u.min() > 0
        _write_splits(out, "darcy_2d_li2021", {"index": np.arange(n, dtype=np.int32), "a": a.astype(np.float32),
                                               "u": u.astype(np.float32)}, p["n_train"], {"params": p})
        print(f"  darcy done {time.time() - t0:.0f}s", flush=True)
    if only in (None, "ns"):
        p = NS
        n = p["n_train"] + p["n_test"]
        w0 = ns_inputs(n, p["n_solve"], p["seed"])
        part = out / "_ns_parts"
        part.mkdir(parents=True, exist_ok=True)
        todo = [(i, c) for i, c in enumerate(_chunks(w0, 4)) if not (part / f"{i:04d}.npy").exists()]
        print(f"  ns: {len(todo)} chunks of 4 to solve (resumable: {part})", flush=True)
        with ProcessPoolExecutor(max(1, workers)) as ex:
            solved = ex.map(solve_ns, [c for _, c in todo])
            for k, (i, w) in enumerate(zip([i for i, _ in todo], solved, strict=True)):
                np.save(part / f"{i:04d}.npy", w.astype(np.float32))
                if k % 10 == 0:
                    print(f"  ns chunk {i} ({k + 1}/{len(todo)}) {time.time() - t0:.0f}s", flush=True)
        w = np.concatenate([np.load(part / f"{i:04d}.npy") for i in range((n + 3) // 4)])
        assert w.shape == (n, p["n_save"] + 1, p["n_store"], p["n_store"]) and np.all(np.isfinite(w))
        _write_splits(out, "navier_stokes_2d_li2021_nu1e-3", {"index": np.arange(n, dtype=np.int32), "w": w},
                      p["n_train"], {"params": p})
        print(f"  ns done {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- convergence evidence
def _burgers_job(args):
    u0, n, dt = args
    u = solve_burgers(fourier_pad(u0, n, axes=(1,)) if n != u0.shape[-1] else u0, dt=dt)
    return u[..., :: n // u0.shape[-1]]


def _ns_job(args):
    w0, n, dt = args
    return solve_ns(w0, n_solve=n, dt=dt)


def _darcy_job(args):
    a, r = args
    return solve_darcy(a, refine=r)


def convergence(out: Path, workers: int = 1, n_samples: int = 4, only: str | None = None) -> dict:
    """Resolution and time-step study on held-aside inputs (seed + 7): each setting is compared with a finer
    reference at the stored nodes, all saved times; the dataset setting must be converged to well below the error
    level operator-learning methods reach on these problems (FNO: 3e-3 to 1e-2 rel. L2). ``only``: one of
    "burgers", "darcy", "ns"; the report is merged into an existing ``out/convergence.json``."""
    f = out / "convergence.json"
    rep: dict = json.loads(f.read_text()) if f.exists() else {}
    t0 = time.time()
    if only in (None, "burgers"):
        rep["burgers_1d"] = _convergence_burgers(n_samples, workers)
        print("burgers", json.dumps(rep["burgers_1d"]), f"{time.time() - t0:.0f}s", flush=True)
    if only in (None, "darcy"):
        rep["darcy_2d"] = _convergence_darcy(n_samples, workers)
        print("darcy", json.dumps(rep["darcy_2d"]), f"{time.time() - t0:.0f}s", flush=True)
    if only in (None, "ns"):
        rep["navier_stokes_2d"] = _convergence_ns(n_samples, workers)
        print("ns", json.dumps(rep["navier_stokes_2d"]), f"{time.time() - t0:.0f}s", flush=True)
    rep["n_samples"] = n_samples
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def _convergence_burgers(n_samples: int, workers: int) -> dict:
    p = BURGERS
    u0 = burgers_inputs(n_samples, p["n"], p["seed"] + 7)
    cases = {"dataset (n 1024, dt 1e-4)": (p["n"], p["dt"]), "n 1024, dt 5e-5": (p["n"], p["dt"] / 2),
             "n 2048, dt 1e-4": (2 * p["n"], p["dt"]), "reference (n 4096, dt 2.5e-5)": (4 * p["n"], p["dt"] / 4)}
    res = dict(zip(cases, _pmap(_burgers_job, [(u0, n, dt) for n, dt in cases.values()], workers), strict=True))
    ref = res.pop("reference (n 4096, dt 2.5e-5)")
    out = {k: {"rel_l2_max": float(rel_l2(v[:, 1:], ref[:, 1:]).max()), "max_abs": float(np.abs(v - ref).max())}
           for k, v in res.items()}
    out["mean_conservation_max_abs"] = float(np.abs(res["dataset (n 1024, dt 1e-4)"].mean(-1)
                                                    - u0.mean(-1)[:, None]).max())
    return out


def _convergence_darcy(n_samples: int, workers: int) -> dict:
    p = DARCY
    a = darcy_inputs(n_samples, p["n"], p["seed"] + 7)
    rs = (1, 2, 4, 8)
    sols = dict(zip(rs, _pmap(_darcy_job, [(a, r) for r in rs], workers), strict=True))
    d = {f"refine {r} ({p['n'] * r}^2)": {"rel_l2_max": float(rel_l2(sols[r], sols[8]).max()),
                                          "max_abs": float(np.abs(sols[r] - sols[8]).max())} for r in rs[:-1]}
    e12, e24 = np.abs(sols[1] - sols[2]).max(), np.abs(sols[2] - sols[4]).max()
    e48 = np.abs(sols[4] - sols[8]).max()
    d["observed_order"] = [float(np.log2(e12 / e24)), float(np.log2(e24 / e48))]
    d["reference"] = f"refine 8 ({p['n'] * 8}^2 cells)"
    return d


def _convergence_ns(n_samples: int, workers: int) -> dict:
    p = NS
    w0 = ns_inputs(n_samples, p["n_solve"], p["seed"] + 7)
    cases = {"dataset (n 128, dt 5e-2)": (128, p["dt"]), "n 128, dt 2.5e-2": (128, p["dt"] / 2),
             "n 128, dt 1e-2": (128, 1e-2), "n 64, dt 5e-2": (64, p["dt"]),
             "reference (n 256, dt 5e-3)": (256, 5e-3)}
    jobs = [(w0[i:i + 1], n, dt) for n, dt in cases.values() for i in range(n_samples)]
    flat = _pmap(_ns_job, [(j[0] if j[1] >= 128 else ob.spectral_downsample(j[0], 2, (1, 2)), j[1], j[2])
                           for j in jobs], workers)
    res = {k: np.concatenate(flat[i * n_samples:(i + 1) * n_samples]) for i, k in enumerate(cases)}
    ref = res.pop("reference (n 256, dt 5e-3)")
    out = {k: {"rel_l2_max": float(rel_l2(v[:, 1:], ref[:, 1:]).max()),
               "rel_l2_final_max": float(rel_l2(v[:, -1], ref[:, -1]).max())} for k, v in res.items()}
    out["note"] = "the n 64 run starts from the n 128 input sampled on 64^2 (its Fourier modes |k| >= 32 dropped)"
    return out
