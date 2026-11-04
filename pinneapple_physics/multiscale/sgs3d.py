"""A-priori benchmark for the 3-D subgrid closures, on a real DNS.

A subgrid closure lives or dies on one question, asked before any simulation is run: given a resolved field that
came from a *known* flow, how close is the stress the closure predicts to the stress the filter actually removed?

    tau_ij = filter(u_i u_j) - filter(u_i) filter(u_j)

That is the a-priori test, and it needs three things: a resolved flow one can trust (here a pseudo-spectral DNS of
the Taylor-Green vortex), the exact subfilter stress of that flow (the sharp spectral filter, computed without
aliasing), and the closures evaluated on the same resolved field. This module builds all three and reports the
numbers the closures are usually judged by:

* the correlation between modelled and true stress, globally and per component;
* the magnitude ratio (a closure that correlates perfectly but is 3x too strong still fails a posteriori);
* the correlation of the subfilter energy transfer ``tau_ij S_ij`` -- the quantity the dynamics actually feels;
* the "ideal" eddy viscosity, the least-squares viscosity that reproduces the true stress exactly, which says
  whether a closure's error is in its *shape* or merely in its *coefficient*.

The last block trains a small network on the same pairs and scores it with the same metrics, held out on a
*different snapshot* rather than on shuffled points of the same one: learning the stress from data is only
meaningful if the learned model survives a time it was not trained on.

What this benchmark is not: an a-posteriori test. A closure that scores well here can still be unstable in an LES,
because a-priori correlation says nothing about how the error feeds back into the resolved scales. That is
:func:`pinneapple_physics.multiscale.closures.aposteriori_test` for the 2-D spectral case, and an open gap in 3-D.

Written from the equations; the DNS comes from
:class:`pinneapple_simulation.numerical_solvers.ns3d_spectral.SpectralNS3D` (imported on demand, so this module
does not pull the solver in).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "LENGTH", "AprioriCase", "apriori_case", "taylor_green_dns", "taylor_green_snapshots", "spectral_gradient",
    "sharp_filter", "periodic_stencil", "closure_eddy_viscosity", "modelled_stress", "ideal_eddy_viscosity",
    "apriori_metrics", "learn_sgs_stress", "run_benchmark", "default_closures", "gradient_model_stress",
]

LENGTH = 2.0 * np.pi


# ---------------------------------------------------------------------------
# spectral tools on a periodic box
# ---------------------------------------------------------------------------

def _wavenumbers(n: int, length: float) -> np.ndarray:
    return np.fft.fftfreq(n, d=length / n) * 2.0 * np.pi


def _modes(n: int) -> np.ndarray:
    """Integer mode indices ``m`` of an ``n``-point axis (``k = 2 pi m / L``)."""
    return np.fft.fftfreq(n) * n


def spectral_gradient(u: np.ndarray, length: float = LENGTH) -> np.ndarray:
    """Exact velocity gradient ``g[i, j] = du_i/dx_j`` of a periodic ``(3, n, n, n)`` field."""
    n = u.shape[-1]
    k = _wavenumbers(n, length)
    g = np.empty((3, 3, n, n, n))
    for i in range(3):
        uh = np.fft.fftn(u[i])
        for j in range(3):
            shape = [1, 1, 1]
            shape[j] = n
            g[i, j] = np.fft.ifftn(1j * k.reshape(shape) * uh).real
    return g


def _cutoff_mask(n: int, kc: float) -> np.ndarray:
    """Keep the modes ``|m_i| <= kc`` on an ``n^3`` grid (``kc`` in mode indices, not radians)."""
    k = _modes(n)
    inside = (np.abs(k) <= kc + 1e-9)
    return inside[:, None, None] & inside[None, :, None] & inside[None, None, :]


def sharp_filter(f: np.ndarray, kc: float) -> np.ndarray:
    """Sharp spectral filter: keep the Fourier modes ``|m_i| <= kc``, drop the rest."""
    mask = _cutoff_mask(f.shape[-1], kc)
    return np.fft.ifftn(np.fft.fftn(f, axes=(-3, -2, -1)) * mask, axes=(-3, -2, -1)).real


def _pad_to(f: np.ndarray, m: int) -> np.ndarray:
    """Zero-pad a periodic field in Fourier space to an ``m^3`` grid (Orszag's 3/2 rule).

    Two details make this work and are easy to get wrong: the negative wavenumbers have to be moved to the *end*
    of the padded array (leaving them at index ``n-1`` puts them at a positive frequency, which returns the field
    multiplied by a mode that vanishes at every other sample -- it looks almost right), and the spectrum has to be
    scaled by ``(m/n)^3``, because the inverse transform normalises by the grid size.
    """
    n = f.shape[-1]
    fh = np.fft.fftn(f)
    k = n // 2
    out = np.zeros((m, m, m), dtype=complex)
    for src0, dst0 in ((slice(0, k), slice(0, k)), (slice(n - k, n), slice(m - k, m))):
        for src1, dst1 in ((slice(0, k), slice(0, k)), (slice(n - k, n), slice(m - k, m))):
            for src2, dst2 in ((slice(0, k), slice(0, k)), (slice(n - k, n), slice(m - k, m))):
                out[dst0, dst1, dst2] = fh[src0, src1, src2]
    out *= (m / n) ** 3
    return np.fft.ifftn(out).real


def _product_filtered_to_coarse(a: np.ndarray, b: np.ndarray, kc: float, n_coarse: int) -> np.ndarray:
    """``filter(a*b)`` on the coarse grid, free of aliasing.

    The product of two fields band-limited to ``n/3`` has modes up to ``2n/3``, so multiplying on the DNS grid
    would fold them back into exactly the low modes this test is trying to measure. The product is therefore
    formed on a 3n/2 grid (where ``2n/3`` still fits), truncated to ``kc``, and only then sampled on the coarse
    grid -- which is exact, because a field band-limited to ``kc <= n_coarse/3`` is represented there without
    loss.
    """
    n = a.shape[-1]
    m = 3 * n // 2
    prod = _pad_to(a, m) * _pad_to(b, m)
    ph = np.fft.fftn(prod) * _cutoff_mask(m, kc)
    fine = np.fft.ifftn(ph).real
    stride = m // n_coarse
    return fine[::stride, ::stride, ::stride]


def periodic_stencil(n: int, radius: int = 1, kernel: str = "box", bandwidth: float | None = None):
    """Neighbour stencil of a periodic ``n^3`` grid, for the dynamic model's test filter.

    Returned in the ``(idx, w)`` form :func:`pinneapple_physics.pde_environment.turbulence_presets.
    filter_stencil` produces, and in the same row-major ``(i, j, k)`` ordering as ``u.reshape(3, -1)`` -- so it can
    be handed straight to ``DynamicSmagorinskyResiduals(stencil=...)`` instead of a k-nearest-neighbour search
    that would only approximate the same thing.
    """
    r = int(radius)
    if r < 1:
        raise ValueError(f"radius must be >= 1; got {radius}")
    offs = [(di, dj, dk) for di in range(-r, r + 1) for dj in range(-r, r + 1) for dk in range(-r, r + 1)]
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ii, jj, kk = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)

    idx = np.empty((ii.size, len(offs)), dtype=np.int64)
    for c, (di, dj, dk) in enumerate(offs):
        idx[:, c] = (((ii + di) % n) * n * n + ((jj + dj) % n) * n + ((kk + dk) % n))

    dist = np.array([np.sqrt(di * di + dj * dj + dk * dk) for di, dj, dk in offs], dtype=float)
    if kernel == "box":
        w = np.ones_like(dist)
    elif kernel == "gaussian":
        sigma = float(bandwidth) if bandwidth is not None else float(r)
        w = np.exp(-0.5 * (dist / max(sigma, 1e-12)) ** 2)
    else:
        raise ValueError(f"kernel must be 'box' or 'gaussian'; got {kernel!r}")
    w = np.array(np.broadcast_to(w / w.sum(), (ii.size, len(offs))), dtype=np.float64)
    return idx, w


# ---------------------------------------------------------------------------
# the reference flow
# ---------------------------------------------------------------------------

def taylor_green_snapshots(
    n: int = 64,
    re: float = 1600.0,
    times: Sequence[float] = (4.0, 5.0, 6.0, 7.0, 9.0),
    dt: float = 0.01,
) -> dict[float, np.ndarray]:
    """One DNS run, snapshotted at the given times (``{t: u}``), continuing in between.

    A learned closure needs a *training set*, not a training moment: a network fitted to a single snapshot of the
    Taylor-Green vortex reproduces it (in-sample correlation 1.0) and transfers nothing to the next one, which
    measures memorisation rather than closure. Several snapshots across the transition are what makes the
    held-out score mean something -- and they come from the same run, so nothing is re-simulated.
    """
    out: dict[float, np.ndarray] = {}
    u = None
    t_prev = 0.0
    for t in sorted(set(float(t) for t in times)):
        snap = taylor_green_dns(n=n, re=re, t_end=t, dt=dt, t_start=t_prev, u0=u)
        u = snap["u"]
        t_prev = t
        out[float(t)] = u
    return out


def taylor_green_dns(
    n: int = 64,
    re: float = 1600.0,
    t_end: float = 9.0,
    dt: float = 0.01,
    t_start: float = 0.0,
    u0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pseudo-spectral DNS of the Taylor-Green vortex on ``[0, 2*pi)^3``.

    The Taylor-Green vortex is the standard LES test case because it needs no wall and no forcing: one smooth
    initial condition rolls up into sheets, then into a turbulent state whose dissipation peaks near ``t = 9`` at
    ``Re = 1600``. A snapshot there is turbulence a closure has to describe, produced by nothing but the
    equations.

    ``u0`` continues a previous run (pass the ``u`` of an earlier snapshot); ``t_start`` is only used to label the
    returned time. ``dt = 0.01`` is inside the RK4 stability limit at this Reynolds number up to ``n = 64``, the
    reference resolution: at ``Re = 1600`` a 48^3 run is itself under-resolved by ``t = 9``, so its a-priori
    correlations are noticeably poorer and its filter band is mostly numerical noise. ``n = 16`` or ``32``
    resolves only the early, still-laminar phase, and is intended for tests.
    """
    from pinneapple_simulation.numerical_solvers.ns3d_spectral import SpectralNS3D, taylor_green_ic

    solver = SpectralNS3D(n, nu=1.0 / float(re), dt=float(dt))
    if u0 is None:
        u0 = taylor_green_ic(n)
    history = solver.run(u0, t_end=max(float(t_end) - float(t_start), 0.0))
    return {"u": solver.u, "t": float(t_start) + float(solver.t), "history": history, "n": n, "re": float(re)}


# ---------------------------------------------------------------------------
# the a-priori pair: resolved field and the stress the filter removed
# ---------------------------------------------------------------------------

@dataclass
class AprioriCase:
    """One a-priori sample: the resolved field and its exact subfilter stress.

    Attributes
    ----------
    u_bar : (3, N, N, N)
        Sharply filtered velocity, sampled on the coarse grid.
    S_bar, g_bar : (3, 3, N, N, N)
        Strain rate and velocity gradient of ``u_bar``, spectral derivatives (exact).
    tau : (3, 3, N, N, N)
        True subfilter stress ``filter(u_i u_j) - filter(u_i) filter(u_j)``.
    kc, delta, n_coarse :
        Cut-off wavenumber, filter width ``pi / kc``, coarse points per axis.
    """

    u_bar: np.ndarray
    S_bar: np.ndarray
    g_bar: np.ndarray
    tau: np.ndarray
    kc: float
    delta: float
    n_coarse: int
    length: float = LENGTH
    label: str = ""

    @property
    def points(self) -> int:
        return self.n_coarse ** 3

    def transfer(self) -> np.ndarray:
        """Subfilter energy transfer ``tau_ij S_ij`` (positive = forward scatter)."""
        return (self.tau * self.S_bar).sum(axis=(0, 1))


def apriori_case(
    u_fine: np.ndarray,
    n_coarse: int | None = None,
    length: float = LENGTH,
    label: str = "",
) -> AprioriCase:
    """Build the a-priori pair from a DNS velocity field.

    ``n_coarse`` defaults to half the DNS resolution, and the cut-off is then ``kc = n_coarse / 3`` -- the largest
    band the coarse grid carries without dealiasing error, which is what makes sampling it there exact.
    """
    n = u_fine.shape[-1]
    if n_coarse is None:
        n_coarse = n // 2
    if n % n_coarse:
        raise ValueError(f"n_coarse must divide the DNS resolution; got n_coarse={n_coarse}, n={n}")
    kc = n_coarse / 3.0

    stride = n // n_coarse
    u_bar = np.stack([sharp_filter(u_fine[i], kc)[::stride, ::stride, ::stride] for i in range(3)])
    g_bar = spectral_gradient(u_bar, length)
    S_bar = 0.5 * (g_bar + g_bar.transpose(1, 0, 2, 3, 4))

    tau = np.empty((3, 3, n_coarse, n_coarse, n_coarse))
    for i in range(3):
        for j in range(i, 3):
            f_ij = _product_filtered_to_coarse(u_fine[i], u_fine[j], kc, n_coarse)
            tau[i, j] = tau[j, i] = f_ij - u_bar[i] * u_bar[j]

    return AprioriCase(
        u_bar=u_bar, S_bar=S_bar, g_bar=g_bar, tau=tau,
        kc=kc, delta=np.pi / kc, n_coarse=n_coarse, length=length, label=label,
    )


# ---------------------------------------------------------------------------
# closures on that pair
# ---------------------------------------------------------------------------

def closure_eddy_viscosity(closure: Any, case: AprioriCase, stencil: tuple | None = None) -> np.ndarray:
    """Evaluate a real closure's ``sgs_viscosity`` on the resolved field of ``case``.

    The closures live in ``pinneapple_physics.pde_environment.turbulence_presets`` and expect the resolved
    velocity gradient, the strain and the points -- which the a-priori case already has, computed spectrally.
    Nothing is re-implemented here: the same ``sgs_viscosity`` a PINN trains against is the one being scored.
    """
    import torch

    sp = list(closure.spatial_indices)
    n = case.n_coarse
    h = case.length / n
    axis = np.arange(n) * h
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    x = torch.as_tensor(np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1),
                                  np.zeros(n ** 3)], axis=1), dtype=torch.float64)

    g = torch.as_tensor(case.g_bar.transpose(2, 3, 4, 0, 1).reshape(-1, 3, 3), dtype=torch.float64)
    S = 0.5 * (g + g.transpose(1, 2))
    U = torch.as_tensor(case.u_bar.transpose(1, 2, 3, 0).reshape(-1, 3), dtype=torch.float64)
    delta = torch.full((n ** 3, 1), float(case.delta), dtype=torch.float64)

    if stencil is not None and getattr(closure, "stencil", None) is None:
        closure.stencil = (np.asarray(stencil[0]), np.asarray(stencil[1]))

    nu = closure.sgs_viscosity(g, S, U, delta, x)
    return np.asarray(nu.detach().reshape(n, n, n))


def modelled_stress(nu_sgs: np.ndarray, S_bar: np.ndarray) -> np.ndarray:
    """``tau_ij = -2 nu_sgs S_ij`` -- the stress an eddy-viscosity closure predicts."""
    return -2.0 * nu_sgs[None, None] * S_bar


def ideal_eddy_viscosity(case: AprioriCase) -> np.ndarray:
    """Pointwise least-squares viscosity ``-tau_ij S_ij / (2 S_mn S_mn)``.

    The viscosity that would reproduce the true stress exactly. It goes negative where the true stress
    backscatters -- which no eddy-viscosity model with a positive coefficient can do, and is the main thing the
    a-priori numbers below expose.
    """
    ts = (case.tau * case.S_bar).sum(axis=(0, 1))
    ss = (case.S_bar * case.S_bar).sum(axis=(0, 1))
    return -ts / (2.0 * np.maximum(ss, 1e-30))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def apriori_metrics(tau_model: np.ndarray, case: AprioriCase) -> dict[str, Any]:
    """The numbers a closure is judged by, a priori.

    ``corr`` and ``mag_ratio`` are the standard pair (Piomelli 1999): a closure must correlate with the true
    stress *and* get its magnitude right, and either one alone is easy to fake. ``transfer_corr`` correlates
    ``tau_ij S_ij``, the energy the filter actually moved between scales, because that is what the resolved
    dynamics feels. ``nu_ideal`` and ``nu_ratio`` say whether the error is in the coefficient or in the shape.
    """
    comps = {}
    for i in range(3):
        for j in range(i, 3):
            comps[f"tau_{'xyz'[i]}{'xyz'[j]}"] = _corr(tau_model[i, j], case.tau[i, j])

    rms_true = float(np.sqrt((case.tau ** 2).mean()))
    rms_model = float(np.sqrt((tau_model ** 2).mean()))
    nu = ideal_eddy_viscosity(case)
    return {
        "corr": _corr(tau_model, case.tau),
        "corr_components": comps,
        "mag_ratio": rms_model / rms_true if rms_true > 0 else float("nan"),
        "transfer_corr": _corr((tau_model * case.S_bar).sum(axis=(0, 1)), case.transfer()),
        "nu_ideal_mean": float(nu.mean()),
        "nu_ideal_negative_fraction": float((nu < 0).mean()),
        "backscatter_fraction": float((case.transfer() < 0).mean()),
    }


# ---------------------------------------------------------------------------
# learning the same stress from data, on the same pairs
# ---------------------------------------------------------------------------

def _invariant_features(case: AprioriCase) -> np.ndarray:
    """Local features of the resolved field, per point (``(M, 19)``).

    Two groups. The tensor components ``S_ij`` and ``O_ij`` are handed over directly, because the whole point of a
    learned closure is that it does not have to assume ``tau ~ S``: given the strain and the rotation, a network
    can build the *shape* the data wants (the gradient/scale-similarity part, which is what the algebraic
    eddy-viscosity models structurally miss). The five normalised invariants and the two magnitudes ``|S|, |O|``
    add the scale information those components do not carry on their own.
    """
    S = case.S_bar
    O = 0.5 * (case.g_bar - case.g_bar.transpose(1, 0, 2, 3, 4))
    d2 = case.delta ** 2

    magS = np.sqrt(2.0 * (S * S).sum(axis=(0, 1)))
    magO = np.sqrt(2.0 * (O * O).sum(axis=(0, 1)))
    S2 = (S * S).sum(axis=(0, 1))
    O2 = (O * O).sum(axis=(0, 1))
    S3 = np.einsum("ij...,jk...,ki...->...", S, S, S)
    S2O = np.einsum("ij...,jk...,ki...->...", S, S, O)
    SO2 = np.einsum("ij...,jk...,ki...->...", S, O, O)
    eps = 1e-12
    mag2 = np.maximum(magS * magS, eps)
    mag3 = np.maximum(magS ** 3, eps)

    tensors = [S[i, j] for i in range(3) for j in range(i, 3)] + \
              [O[i, j] for i in range(3) for j in range(i, 3)]
    invariants = [
        magS * d2,                                 # |S| Delta^2  (dimensionless)
        magO * d2,
        S2 / mag2,
        O2 / mag2,
        S3 / mag3,
        S2O / mag3,
        SO2 / mag3,
    ]
    # Stack on the *last* axis: ``(n, n, n, 19)`` then flattened row-major is one row per grid point, in the same
    # ordering as ``tau[i, j].reshape(-1)``. Stacking on any other axis gives a permutation of the points that
    # still trains (both sides are scrambled the same way is a coincidence, not a guarantee) and silently destroys
    # the generalisation the comparison is about.
    return np.stack([np.broadcast_to(f, magS.shape) for f in tensors + invariants],
                    axis=-1).reshape(-1, 19)


def _target_stress(case: AprioriCase) -> np.ndarray:
    """``tau_ij / (Delta^2 |S|^2)`` -- the stress the closure has to predict, made dimensionless."""
    scale = np.maximum(case.delta ** 2 * (2.0 * (case.S_bar * case.S_bar).sum(axis=(0, 1))), 1e-30)
    comps = [case.tau[i, j] / scale for i in range(3) for j in range(i, 3)]
    return np.stack(comps, axis=-1).reshape(-1, 6)


def learn_sgs_stress(
    case_train: AprioriCase | Sequence[AprioriCase],
    case_test: AprioriCase,
    *,
    steps: int = 3000,
    hidden: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a small network to the true subfilter stress and score it on a *different* snapshot.

    The model predicts the six independent components of ``tau_ij`` from the local velocity gradient, with no
    eddy-viscosity assumption imposed: whatever shape the data has, it can learn. Everything else is held equal
    to the algebraic closures -- the same snapshot, the same points, the same metrics from
    :func:`apriori_metrics`.

    ``case_train`` should be *several* snapshots, and ``case_test`` a later one. Both choices are the difference
    between a closure and a lookup table: fitted to one snapshot the network reproduces it exactly (in-sample
    correlation 1.0) and scores at chance on the next one, while five snapshots across the transition put it in
    the same range as the algebraic closures. Reporting the single-snapshot number would be reporting
    memorisation.
    """
    import torch

    train_cases = [case_train] if isinstance(case_train, AprioriCase) else list(case_train)
    if not train_cases:
        raise ValueError("learn_sgs_stress needs at least one training snapshot")

    torch.manual_seed(seed)
    X_tr_np = np.concatenate([_invariant_features(c) for c in train_cases])
    y_tr_np = np.concatenate([_target_stress(c) for c in train_cases])
    X_tr = torch.as_tensor(X_tr_np, dtype=torch.float64)
    y_tr = torch.as_tensor(y_tr_np, dtype=torch.float64)
    X_te = torch.as_tensor(_invariant_features(case_test), dtype=torch.float64)

    # Standardise on the training snapshots only -- scaling with test statistics would leak the held-out field.
    x_mean, x_std = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-12)

    net = torch.nn.Sequential(
        torch.nn.Linear(X_tr.shape[1], hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 6),
    ).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = ((net((X_tr - x_mean) / x_std) - y_tr) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = net((X_te - x_mean) / x_std).numpy()

    shape = case_test.tau.shape[2:]
    scale = np.maximum(case_test.delta ** 2
                       * (2.0 * (case_test.S_bar * case_test.S_bar).sum(axis=(0, 1))), 1e-30)
    tau_pred = np.empty(case_test.tau.shape)
    for c, (i, j) in enumerate((a, b) for a in range(3) for b in range(a, 3)):
        comp = (pred[:, c].reshape(shape) * scale)
        tau_pred[i, j] = tau_pred[j, i] = comp

    metrics = apriori_metrics(tau_pred, case_test)
    metrics["train_points"] = int(sum(c.points for c in train_cases))
    metrics["train_snapshots"] = len(train_cases)
    metrics["test_points"] = int(case_test.points)
    metrics["steps"] = int(steps)
    return {"model": net, "tau": tau_pred, "metrics": metrics, "x_mean": x_mean, "x_std": x_std}


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------

def gradient_model_stress(case: AprioriCase) -> np.ndarray:
    """Scale-similarity (Clark) reference stress ``tau_ij = (Delta^2 / 12) d_k u_i d_k u_j``.

    Not a closure this repo runs in an LES -- it is a-priori excellent and notoriously unstable when it is left
    to drive the dynamics on its own. It is in the table as the *control*: it uses exactly the same resolved
    gradient the eddy-viscosity closures use, and because it is not constrained to the shape ``tau ~ S``, the gap
    between its correlation and theirs is the price of the eddy-viscosity assumption, separated from any question
    of coefficient.
    """
    g = case.g_bar
    return (case.delta ** 2 / 12.0) * np.einsum("ik...,jk...->ij...", g, g)


def default_closures(delta: float, n_coarse: int) -> dict[str, Any]:
    """The three closures this repo ships, configured for a grid LES (test filter from the grid, not from k-NN)."""
    from pinneapple_physics.pde_environment.turbulence_presets import (
        DynamicSmagorinskyResiduals, SmagorinskyResiduals, WALEResiduals,
    )

    return {
        "Smagorinsky": SmagorinskyResiduals(nu=0.0, cs=0.17, delta=delta),
        "DynamicSmagorinsky": DynamicSmagorinskyResiduals(
            nu=0.0, delta=delta, test_ratio=2.0, averaging="batch", clip=True,
            stencil=periodic_stencil(n_coarse, radius=2, kernel="gaussian", bandwidth=2.0),
        ),
        "WALE": WALEResiduals(nu=0.0, cw=0.325, delta=delta),
    }


def run_benchmark(
    n: int = 64,
    re: float = 1600.0,
    t_train: float | Sequence[float] = (4.0, 5.0, 6.0, 7.0),
    t_test: float = 9.0,
    dt: float = 0.01,
    n_coarse: int | None = None,
    closures: Sequence[str] | None = None,
    learn: bool = True,
    learn_steps: int = 3000,
    seed: int = 0,
) -> dict[str, Any]:
    """DNS -> exact subfilter stress -> every closure -> the learned one, on held-out snapshots.

    ``t_train`` is a sequence of snapshot times the learned closure is trained on; ``t_test`` is a *later* time it
    never sees. All of them come from one continuous DNS run. Returns a dict with the training and test
    :class:`AprioriCase` objects, a ``table`` mapping closure name to its metrics (computed on the test
    snapshot), the DNS history and the settings that produced them -- enough to reproduce the numbers, which is
    the whole point of a benchmark.
    """
    t_train_times = [float(t_train)] if isinstance(t_train, (int, float)) else [float(t) for t in t_train]
    times = sorted(set(t_train_times + [float(t_test)]))
    snaps = taylor_green_snapshots(n=n, re=re, times=times, dt=dt)

    cases = {t: apriori_case(snaps[t], n_coarse=n_coarse, label=f"t={t:.2f}") for t in times}
    case_test = cases[float(t_test)]
    case_train = [cases[t] for t in t_train_times]

    available = default_closures(case_test.delta, case_test.n_coarse)
    wanted = list(available) if closures is None else list(closures)
    unknown = [c for c in wanted if c not in available]
    if unknown:
        raise ValueError(f"unknown closures {unknown}; available: {sorted(available)}")

    table: dict[str, dict[str, Any]] = {}
    # The scale-similarity control is scored first: it shares the resolved gradient with the eddy-viscosity
    # closures and is not bound to their shape, so it is what separates "wrong coefficient" from "wrong shape".
    table["GradientModel"] = apriori_metrics(gradient_model_stress(case_test), case_test)
    for name in wanted:
        closure = available[name]
        nu = closure_eddy_viscosity(closure, case_test)
        table[name] = apriori_metrics(modelled_stress(nu, case_test.S_bar), case_test)
        # The dynamic model computes its own coefficient; record it, because a clipped (non-positive) C is the
        # difference between "the model is wrong" and "the model switched itself off", and the stress alone
        # cannot tell those apart.
        germano = getattr(closure, "last_germano", None)
        if germano and germano.get("C") is not None:
            table[name]["germano_C"] = float(germano["C"])

    learned = None
    if learn:
        learned = learn_sgs_stress(case_train, case_test, steps=learn_steps, seed=seed)
        table["Learned"] = learned["metrics"]

    return {
        "case_train": case_train,
        "case_test": case_test,
        "table": table,
        "learned": learned,
        "settings": {"n": n, "re": re, "dt": dt, "t_train": t_train_times, "t_test": float(t_test),
                     "n_coarse": case_test.n_coarse, "kc": case_test.kc, "delta": case_test.delta,
                     "learn_steps": learn_steps if learn else 0},
        "times": times,
        "_snapshots": snaps,
    }


def _format_table(table: dict[str, dict[str, Any]]) -> str:
    header = f"{'closure':<22}{'corr':>8}{'mag':>8}{'transfer':>10}{'nu_ideal':>12}"
    lines = [header, "-" * len(header)]
    for name, m in table.items():
        lines.append(f"{name:<22}{m['corr']:>8.3f}{m['mag_ratio']:>8.3f}"
                     f"{m['transfer_corr']:>10.3f}{m['nu_ideal_mean']:>12.3e}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import argparse

    parser = argparse.ArgumentParser(description="A-priori benchmark of the 3-D subgrid closures (Taylor-Green).")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--re", type=float, default=1600.0)
    parser.add_argument("--t-train", type=float, nargs="+", default=[4.0, 5.0, 6.0, 7.0])
    parser.add_argument("--t-test", type=float, default=9.0)
    parser.add_argument("--n-coarse", type=int, default=None)
    parser.add_argument("--learn-steps", type=int, default=3000)
    parser.add_argument("--no-learn", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(n=args.n, re=args.re, t_train=args.t_train, t_test=args.t_test,
                           n_coarse=args.n_coarse, learn=not args.no_learn, learn_steps=args.learn_steps)
    print(f"Taylor-Green vortex, Re={args.re}, DNS {args.n}^3, coarse {report['settings']['n_coarse']}^3, "
          f"kc={report['settings']['kc']:.2f}, delta={report['settings']['delta']:.3f}")
    print(f"learned closure trained on {report['settings']['t_train']}, held-out snapshot: "
          f"{report['case_test'].label}")
    print(_format_table(report["table"]))
