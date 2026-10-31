r"""Koopman operator and spectral analysis: DMD, EDMD, dictionaries of observables, eigenfunctions, residuals.

The Koopman operator of a map :math:`F` acts on observables: :math:`(\mathcal K g)(x) = g(F(x))`. It is linear even
when :math:`F` is not, so a nonlinear system has a spectral decomposition in the space of observables. Data-driven
methods approximate :math:`\mathcal K` on a finite dictionary :math:`\psi = (\psi_1, \dots, \psi_m)` from snapshot
pairs :math:`(x_n, y_n = F(x_n))`:

* **EDMD** (Williams, Kevrekidis & Rowley 2015, J. Nonlinear Sci. 25:1307): with :math:`\Psi_X, \Psi_Y` the
  dictionary evaluated on the pairs, :math:`G = \Psi_X^* \Psi_X / N`, :math:`A = \Psi_X^* \Psi_Y / N`,
  :math:`K = G^{+} A`, so :math:`\psi(F(x)) \approx \psi(x) K`. An eigenpair :math:`K\xi = \lambda\xi` gives the
  approximate eigenfunction :math:`\varphi(x) = \psi(x)\xi`, exact when the dictionary spans an invariant subspace.
* **DMD** (Schmid 2010; exact DMD, Tu et al. 2014, J. Comput. Dyn. 1:391): EDMD with the identity dictionary,
  computed through a rank-:math:`r` SVD of the snapshot matrix so that high-dimensional states (PDE snapshots) cost
  :math:`O(D N r)`. The rank is chosen by the optimal hard threshold of Gavish & Donoho (2014, IEEE TIT 60:5040), by
  an energy fraction, or given. **TLS-DMD** (Hemati et al. 2017, Theor. Comput. Fluid Dyn. 31:349) removes the bias
  that noise in the data puts on the eigenvalues (towards damping).
* **Residuals** (ResDMD, Colbrook & Townsend 2024, Comm. Pure Appl. Math. 77:221): for each candidate eigenpair,
  :math:`\mathrm{res}(\lambda,\xi)^2 = \xi^*(L - \lambda A^* - \bar\lambda A + |\lambda|^2 G)\xi / \xi^* G \xi`
  with :math:`L = \Psi_Y^*\Psi_Y/N`: zero for an exact eigenfunction, large for a spurious eigenvalue produced by the
  truncation. A spectrum is not reported as trustworthy without them.
* **Continuous time**: :math:`\omega = \log\lambda / \Delta t` (the principal branch: frequencies above the Nyquist
  frequency :math:`\pi/\Delta t` alias), or the generator fitted from derivative data (gEDMD, Klus et al. 2020,
  Physica D 406:132416) with the dictionary differentiated along the velocity by central differences.

Objects: :class:`ObservableDictionary` (identity, monomials, RBF, Fourier, custom, neural observables from a PyTorch
module), :class:`DMD` and :class:`EDMD` (estimators) returning a :class:`KoopmanOperator` (the fitted operator, with
:class:`KoopmanEigenvalues`, :class:`KoopmanEigenfunctions`, :class:`DynamicMode` s, residuals, stability),
:class:`KoopmanReconstruction` (reconstruction and multi-step prediction errors), :class:`SpectralAnalyzer`
(frequencies, growth rates, mode ranking, noise sensitivity, order selection, comparison with a nonlinear
state-space regression).

Name: this :class:`KoopmanOperator` is a fitted finite-dimensional approximation (NumPy, no training). It is not
:class:`pinneapple_neural.architectures.rom.koopman.KoopmanOperator`, the learnable linear layer inside the deep
Koopman autoencoder; that class is unchanged and a trained autoencoder's encoder can be used here as a dictionary
(:meth:`ObservableDictionary.neural`).
"""
from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._backends import to_numpy
from ._contract import Assumption, Diagnostic, DynamicsResult, MethodCard, reproducibility
from .trajectories import Trajectory, TrajectoryEnsemble

__all__ = [
    "ObservableDictionary", "DMD", "EDMD", "KoopmanOperator", "KoopmanEigenvalues", "KoopmanEigenfunctions",
    "DynamicMode", "KoopmanReconstruction", "SpectralAnalyzer", "snapshot_pairs", "gavish_donoho_rank",
    "CARD_DMD", "CARD_EDMD",
]


# =========================================================================== data
def snapshot_pairs(data, *, lag: int = 1) -> tuple[np.ndarray, np.ndarray, float | None]:
    """``(X0, X1, dt)`` with rows ``x_n`` and ``x_{n+lag}`` from a :class:`Trajectory` (regular grid), a
    :class:`TrajectoryEnsemble` (pairs within each member, never across members), an ``(n, d)`` array, or an explicit
    ``(X0, X1)`` tuple. Pairs with a missing value are dropped."""
    if isinstance(data, tuple) and len(data) == 2:
        X0, X1 = (np.atleast_2d(to_numpy(a)) for a in data)
        return X0, X1, None
    members = list(data) if isinstance(data, TrajectoryEnsemble) else [data]
    A, B, dts = [], [], []
    for m in members:
        if isinstance(m, Trajectory):
            if not m.is_regular():
                raise ValueError("Koopman estimators need a regular time grid; resample the trajectory first "
                                 "(Trajectory.resample marks interpolated data)")
            X, dt = m.x, m.dt
        else:
            X, dt = np.asarray(to_numpy(m), dtype=float), None
            X = X[:, None] if X.ndim == 1 else X
        a, b = X[:-lag], X[lag:]
        ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
        A.append(a[ok])
        B.append(b[ok])
        dts.append(dt)
    d0 = {round(d, 12) for d in dts if d is not None}
    if len(d0) > 1:
        raise ValueError(f"members have different time steps {sorted(d0)}")
    dt = (dts[0] * lag) if dts[0] is not None else None
    return np.vstack(A), np.vstack(B), dt


# =========================================================================== dictionaries
@dataclass
class ObservableDictionary:
    """A list of observables ``psi_k(X) -> (n,)`` with names; ``__call__(X)`` gives ``Psi`` of shape ``(n, m)``.

    ``state_index``: positions of the dictionary entries equal to the state components (set by the builders that
    include them); reconstruction then reads the state off those entries, otherwise it fits a linear map."""
    funcs: list[Callable[[np.ndarray], np.ndarray]]
    names: list[str]
    state_index: list[int] | None = None
    kind: str = "custom"

    def __post_init__(self):
        if len(self.funcs) != len(self.names):
            raise ValueError("one name per observable")

    def __len__(self) -> int:
        return len(self.names)

    def __call__(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(to_numpy(X), dtype=float))
        cols = [np.asarray(f(X), dtype=float).reshape(X.shape[0]) for f in self.funcs]
        return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))

    def __add__(self, other: ObservableDictionary) -> ObservableDictionary:
        si = self.state_index
        if si is None and other.state_index is not None:
            si = [len(self) + i for i in other.state_index]
        return ObservableDictionary(self.funcs + other.funcs, self.names + other.names, si, f"{self.kind}+{other.kind}")

    @classmethod
    def identity(cls, d: int, names: Sequence[str] | None = None) -> ObservableDictionary:
        names = list(names or [f"x{i}" for i in range(d)])
        return cls([(lambda X, i=i: X[:, i]) for i in range(d)], names, list(range(d)), "identity")

    @classmethod
    def monomials(cls, d: int, degree: int, *, constant: bool = True, names: Sequence[str] | None = None,
                  only: Sequence[tuple[int, ...]] | None = None) -> ObservableDictionary:
        """All monomials of total degree ``<= degree`` (or the exponent tuples in ``only``); the linear ones come
        first after the constant."""
        nm = list(names or [f"x{i}" for i in range(d)])
        exps = list(only) if only is not None else [e for k in range(0 if constant else 1, degree + 1)
                                                    for e in _exponents(d, k)]
        funcs, labels, state = [], [], []
        for e in exps:
            e = tuple(int(v) for v in e)
            funcs.append(lambda X, e=e: np.prod(X ** np.array(e)[None, :], axis=1))
            labels.append("1" if sum(e) == 0 else "*".join(f"{nm[i]}^{p}" if p > 1 else nm[i]
                                                         for i, p in enumerate(e) if p))
            if sum(e) == 1:
                state.append((len(labels) - 1, e.index(1)))
        si = [i for i, _ in sorted(state, key=lambda s: s[1])] if len(state) == d else None
        return cls(funcs, labels, si, f"monomials(deg {degree})")

    @classmethod
    def rbf(cls, centers, epsilon: float, *, kind: str = "gaussian", with_state: bool = True) -> ObservableDictionary:
        """Radial basis functions at ``centers`` (``(k, d)``): ``exp(-(eps r)^2)`` (gaussian) or
        ``sqrt(1 + (eps r)^2)`` (multiquadric); the state coordinates are prepended when ``with_state``."""
        C = np.atleast_2d(np.asarray(centers, dtype=float))
        if kind == "gaussian":
            f = lambda X, c: np.exp(-(epsilon ** 2) * np.sum((X - c) ** 2, axis=1))        # noqa: E731
        elif kind == "multiquadric":
            f = lambda X, c: np.sqrt(1 + (epsilon ** 2) * np.sum((X - c) ** 2, axis=1))     # noqa: E731
        else:
            raise ValueError("kind must be gaussian or multiquadric")
        rb = cls([(lambda X, c=c: f(X, c)) for c in C], [f"rbf{k}" for k in range(len(C))], None, f"rbf({kind})")
        return cls.identity(C.shape[1]) + rb if with_state else rb

    @classmethod
    def fourier(cls, d: int, n_freq: int, period: float | Sequence[float] = 2 * np.pi, *, with_state: bool = False
                ) -> ObservableDictionary:
        """``1, cos(2 pi k x_i / P), sin(2 pi k x_i / P)`` for ``k = 1..n_freq`` per component (periodic states)."""
        P = np.broadcast_to(np.asarray(period, dtype=float), (d,))
        funcs, names = [lambda X: np.ones(X.shape[0])], ["1"]
        for i in range(d):
            for k in range(1, n_freq + 1):
                w = 2 * np.pi * k / P[i]
                funcs += [lambda X, i=i, w=w: np.cos(w * X[:, i]), lambda X, i=i, w=w: np.sin(w * X[:, i])]
                names += [f"cos{k}(x{i})", f"sin{k}(x{i})"]
        out = cls(funcs, names, None, "fourier")
        return cls.identity(d) + out if with_state else out

    @classmethod
    def custom(cls, funcs: Sequence[Callable], names: Sequence[str] | None = None,
               state_index: Sequence[int] | None = None) -> ObservableDictionary:
        return cls(list(funcs), list(names or [f"psi{k}" for k in range(len(funcs))]),
                   None if state_index is None else list(state_index))

    @classmethod
    def neural(cls, model: Callable, n_out: int, *, with_state: bool = True, d: int | None = None,
               name: str = "nn") -> ObservableDictionary:
        """Observables given by a trained network (e.g. the encoder of a Koopman autoencoder): ``model`` maps a
        PyTorch tensor ``(n, d)`` to ``(n, n_out)``. Evaluated without gradients; the dictionary is then fixed (the
        network is not trained here)."""
        def col(X, k):
            import torch

            with torch.no_grad():
                out = model(torch.as_tensor(X, dtype=torch.float32))
            return to_numpy(out)[:, k]

        nnd = cls([(lambda X, k=k: col(X, k)) for k in range(n_out)], [f"{name}{k}" for k in range(n_out)], None,
                  "neural")
        if with_state:
            if d is None:
                raise ValueError("give d (state dimension) to prepend the state")
            return cls.identity(d) + nnd
        return nnd


def _exponents(d: int, k: int):
    for c in itertools.combinations_with_replacement(range(d), k):
        e = [0] * d
        for i in c:
            e[i] += 1
        yield tuple(e)


# =========================================================================== results
@dataclass
class DynamicMode:
    """One eigen-triple: discrete eigenvalue ``lam``, continuous ``omega = log(lam)/dt`` (when ``dt`` is known),
    frequency (cycles per unit time), growth rate, the mode in state space, its amplitude in the data, and the
    ResDMD residual of the eigenpair."""
    index: int
    lam: complex
    omega: complex | None
    frequency: float | None
    growth_rate: float | None
    mode: np.ndarray
    amplitude: float
    residual: float | None

    @property
    def stable(self) -> bool:
        return abs(self.lam) < 1.0 + 1e-12


@dataclass
class KoopmanEigenvalues:
    """Discrete eigenvalues (sorted by decreasing modulus) and, with ``dt``, the continuous ones."""
    discrete: np.ndarray
    dt: float | None = None

    @property
    def continuous(self) -> np.ndarray | None:
        if self.dt is None:
            return None
        with np.errstate(divide="ignore"):
            return np.log(self.discrete.astype(complex)) / self.dt

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.discrete))) if self.discrete.size else 0.0

    def match(self, reference: Sequence[complex], *, continuous: bool = False) -> np.ndarray:
        """For each reference eigenvalue, the distance to the nearest computed one (optimal one-to-one matching
        when the counts allow)."""
        from scipy.optimize import linear_sum_assignment

        ev = self.continuous if continuous else self.discrete
        ref = np.asarray(reference, dtype=complex)
        Dm = np.abs(ref[:, None] - ev[None, :])
        if ref.size <= ev.size:
            r, c = linear_sum_assignment(Dm)
            out = np.empty(ref.size)
            out[r] = Dm[r, c]
            return out
        return Dm.min(axis=1)


@dataclass
class KoopmanEigenfunctions:
    """``phi_k(x) = psi(x) xi_k``: coefficients ``xi`` (``m x r``) on the dictionary."""
    coefficients: np.ndarray
    dictionary: ObservableDictionary

    def __call__(self, X) -> np.ndarray:
        return self.dictionary(X) @ self.coefficients


class KoopmanOperator:
    """A fitted finite-dimensional Koopman approximation; see the module docstring. Built by :class:`DMD` and
    :class:`EDMD`, not directly.

    ``K`` acts on row vectors of dictionary values, ``psi(F(x)) ~ psi(x) K``; ``B`` maps dictionary values to the
    state, ``x ~ psi(x) B``."""

    def __init__(self, K: np.ndarray, dictionary: ObservableDictionary | None, B: np.ndarray, *, dt: float | None,
                 method: str, gram: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                 basis: np.ndarray | None = None, mean: np.ndarray | None = None, info: dict | None = None):
        self.K, self.dictionary, self.B, self.dt, self.method = K, dictionary, B, dt, method
        self.gram, self.basis, self.mean = gram, basis, mean
        self.info = dict(info or {})
        lam, W = np.linalg.eig(K)
        order = np.argsort(-np.abs(lam), kind="stable")
        self._lam, self._xi = lam[order], W[:, order]
        self._left = np.linalg.pinv(self._xi)

    # ------------------------------------------------------------------ spectrum
    @property
    def eigenvalues(self) -> KoopmanEigenvalues:
        return KoopmanEigenvalues(self._lam, self.dt)

    @property
    def eigenfunctions(self) -> KoopmanEigenfunctions:
        if self.dictionary is None:
            raise ValueError("DMD in a reduced basis: eigenfunctions are linear, phi(x) = (x - mean) U xi")
        return KoopmanEigenfunctions(self._xi, self.dictionary)

    def lift(self, X) -> np.ndarray:
        """Dictionary values (EDMD) or reduced coordinates (DMD) of the states ``X``."""
        X = np.atleast_2d(np.asarray(to_numpy(X), dtype=float))
        if self.dictionary is not None:
            return self.dictionary(X)
        return (X - self.mean) @ self.basis

    def residuals(self) -> np.ndarray | None:
        """ResDMD residual of every eigenpair (``None`` without the Gram matrices)."""
        if self.gram is None:
            return None
        G, A, L = self.gram
        out = np.empty(self._lam.size)
        for k in range(self._lam.size):
            xi, lam = self._xi[:, k], self._lam[k]
            num = xi.conj() @ (L - lam * A.conj().T - np.conj(lam) * A + abs(lam) ** 2 * G) @ xi
            den = xi.conj() @ G @ xi
            out[k] = math.sqrt(max(float(np.real(num / den)), 0.0)) if abs(den) > 0 else math.inf
        return out

    def modes(self, X0: np.ndarray | None = None) -> list[DynamicMode]:
        """Dynamic modes: ``x ~ sum_k phi_k(x) v_k`` with ``V = xi^+ B``; amplitudes from ``X0`` (the training
        states when stored)."""
        V = self._left @ self.B.astype(complex)
        if self.basis is not None and self.dictionary is None:
            V = V @ self.basis.T.astype(complex)
        res = self.residuals()
        amps = np.full(self._lam.size, np.nan)
        if X0 is not None:
            Phi = self.lift(X0) @ self._xi
            amps = np.sqrt(np.mean(np.abs(Phi) ** 2, axis=0)) * np.linalg.norm(V, axis=1)
        out = []
        for k, lam in enumerate(self._lam):
            om = (np.log(complex(lam)) / self.dt) if (self.dt and lam != 0) else None
            out.append(DynamicMode(k, complex(lam), om, None if om is None else float(om.imag / (2 * np.pi)),
                                   None if om is None else float(om.real), V[k], float(amps[k]),
                                   None if res is None else float(res[k])))
        return out

    # ------------------------------------------------------------------ prediction
    def predict(self, x0, steps: int) -> np.ndarray:
        """States ``x_0 .. x_steps`` from ``x0`` by propagating the lifted state linearly (``(steps + 1, d)``)."""
        z = self.lift(np.atleast_2d(x0))[0].astype(complex)
        out = [z]
        for _ in range(steps):
            z = z @ self.K
            out.append(z)
        Z = np.array(out)
        X = Z @ self.B
        if self.dictionary is None:
            X = X @ self.basis.T + self.mean
        return np.real_if_close(X, tol=1e6).real

    def stability(self, tol: float = 1e-9) -> dict[str, Any]:
        mags = np.abs(self._lam)
        return {"spectral_radius": float(mags.max()) if mags.size else 0.0,
                "n_unstable": int(np.sum(mags > 1 + tol)), "n_neutral": int(np.sum(np.abs(mags - 1) <= tol)),
                "stable": bool(np.all(mags <= 1 + tol))}

    def generator(self) -> np.ndarray:
        """Continuous-time generator on the dictionary, ``logm(K) / dt`` (principal branch)."""
        from scipy.linalg import logm

        if self.dt is None:
            raise ValueError("dt unknown: pass dt to the estimator or use a Trajectory")
        return logm(self.K) / self.dt


# =========================================================================== estimators
_A_AUTONOMOUS = Assumption("autonomous dynamics", "the snapshot pairs come from one time-invariant map (no forcing "
                           "or parameter drift across the data)")
_A_SPAN = Assumption("dictionary span", "the dictionary (nearly) spans an invariant subspace; otherwise eigenvalues "
                     "can be spurious, which the residuals measure")
_A_SAMPLING = Assumption("sampling", "the time step resolves the fastest relevant frequency (no aliasing beyond "
                         "pi/dt) and the data cover the region of state space where the result is used")

CARD_DMD = MethodCard(
    "koopman.dmd", "trajectory_operator",
    inputs={"data": "Trajectory / TrajectoryEnsemble / (n, d) array / (X0, X1) pairs; PDE snapshots via "
                    "from_snapshots"},
    assumptions=[_A_AUTONOMOUS, _A_SAMPLING, Assumption("linear observables", "the state itself is a good "
                                                                              "observable (linear dynamics)")],
    config={"rank": "int, 'gavish-donoho', or an energy fraction in (0, 1)", "tls": "total-least-squares debiasing",
            "center": "subtract the mean state"},
    outputs={"operator": "KoopmanOperator: eigenvalues, modes, prediction", "rank": "rank used"},
    failure_modes=["nonlinear dynamics: DMD fits a linear map and its spectrum mixes harmonics",
                   "noise biases eigenvalues towards damping (use tls=True)",
                   "too few snapshots for the rank", "aliasing when dt is too coarse"],
    references=["Schmid 2010, J. Fluid Mech. 656:5", "Tu et al. 2014, J. Comput. Dyn. 1:391",
                "Gavish & Donoho 2014, IEEE Trans. Inf. Theory 60:5040", "Hemati et al. 2017, Theor. Comput. Fluid "
                "Dyn. 31:349"])

CARD_EDMD = MethodCard(
    "koopman.edmd", "trajectory_operator",
    inputs={"data": "as DMD", "dictionary": "ObservableDictionary"},
    assumptions=[_A_AUTONOMOUS, _A_SPAN, _A_SAMPLING],
    config={"ridge": "Tikhonov term on the Gram matrix", "dictionary": "observables"},
    outputs={"operator": "KoopmanOperator with eigenfunctions and residuals"},
    failure_modes=["dictionary larger than the data (ill-conditioned Gram matrix)",
                   "spurious eigenvalues from a non-invariant dictionary (large residuals)",
                   "extrapolation outside the sampled region"],
    references=["Williams, Kevrekidis & Rowley 2015, J. Nonlinear Sci. 25:1307",
                "Colbrook & Townsend 2024, Comm. Pure Appl. Math. 77:221 (ResDMD)",
                "Klus et al. 2020, Physica D 406:132416 (generator EDMD)"])


def gavish_donoho_rank(s: np.ndarray, shape: tuple[int, int]) -> int:
    """Optimal hard threshold for singular values with unknown noise level (Gavish & Donoho 2014):
    ``tau = omega(beta) * median(s)``, ``omega(beta) ~ 0.56 beta^3 - 0.95 beta^2 + 1.82 beta + 1.43``."""
    m, n = shape
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return max(1, int(np.sum(s > omega * np.median(s))))


def _diagnostics(op: KoopmanOperator, n_pairs: int, m: int, cond: float | None) -> list[Diagnostic]:
    d = [Diagnostic("n_pairs", n_pairs, "ok" if n_pairs > 2 * m else "warn", "fewer pairs than twice the dictionary "
                    "size" if n_pairs <= 2 * m else "", threshold=2 * m)]
    if cond is not None:
        d.append(Diagnostic("gram_condition", cond, "ok" if cond < 1e10 else "warn", threshold=1e10))
    st = op.stability()
    d.append(Diagnostic("spectral_radius", st["spectral_radius"], "info"))
    res = op.residuals()
    if res is not None:
        d.append(Diagnostic("max_residual", float(np.max(res)), "info"))
    return d


class DMD:
    """Exact DMD (optionally TLS-debiased) on states; see the module docstring."""

    card = CARD_DMD

    def __init__(self, rank: int | float | str = "gavish-donoho", *, tls: bool = False, center: bool = False,
                 lag: int = 1, dt: float | None = None):
        self.rank, self.tls, self.center, self.lag, self.dt = rank, tls, center, lag, dt

    def _rank(self, s: np.ndarray, shape) -> int:
        if isinstance(self.rank, str):
            if self.rank != "gavish-donoho":
                raise ValueError("rank must be an int, an energy fraction or 'gavish-donoho'")
            return gavish_donoho_rank(s, shape)
        if isinstance(self.rank, float) and 0 < self.rank < 1:
            e = np.cumsum(s ** 2) / np.sum(s ** 2)
            return int(np.searchsorted(e, self.rank) + 1)
        return int(min(self.rank, s.size))

    def fit(self, data) -> DynamicsResult:
        t0 = time.perf_counter()
        X0, X1, dt = snapshot_pairs(data, lag=self.lag)
        dt = self.dt if self.dt is not None else dt
        mean = X0.mean(axis=0) if self.center else np.zeros(X0.shape[1])
        A0, A1 = (X0 - mean).T, (X1 - mean).T                     # (D, N)
        if self.tls:
            Z = np.vstack([A0, A1])
            _, sz, Vz = np.linalg.svd(Z, full_matrices=False)
            rz = self._rank(sz, Z.shape)
            P = Vz[:rz].conj().T @ Vz[:rz]
            A0, A1 = A0 @ P, A1 @ P
        U, s, Vh = np.linalg.svd(A0, full_matrices=False)
        r = self._rank(s, A0.shape)
        Ur, Sr, Vr = U[:, :r], s[:r], Vh[:r].conj().T
        At = Ur.conj().T @ A1 @ Vr / Sr[None, :]                   # reduced operator, column convention
        # row convention for KoopmanOperator: z_{n+1} = z_n K with z = (x - mean) Ur
        K = At.T
        op = KoopmanOperator(K, None, np.eye(r), dt=dt, method="tls-dmd" if self.tls else "dmd",
                             basis=Ur, mean=mean,
                             info={"singular_values": s, "rank": r, "exact_modes": A1 @ Vr / Sr[None, :]})
        Zx = (X0 - mean) @ Ur
        Zy = (X1 - mean) @ Ur
        N = Zx.shape[0]
        op.gram = (Zx.conj().T @ Zx / N, Zx.conj().T @ Zy / N, Zy.conj().T @ Zy / N)
        diags = _diagnostics(op, N, r, None)
        diags.append(Diagnostic("rank", r, "info", f"selected by {self.rank}"))
        if dt is not None:
            nyq = np.pi / dt
            om = op.eigenvalues.continuous
            near = int(np.sum(np.abs(np.abs(om.imag) - nyq) < 0.05 * nyq))
            if near:
                diags.append(Diagnostic("near_nyquist", near, "warn", "eigenvalues close to the Nyquist frequency: "
                                        "possible aliasing"))
        cfg = {"rank": self.rank, "tls": self.tls, "center": self.center, "lag": self.lag, "dt": dt}
        rep = reproducibility(CARD_DMD.name, cfg, [X0, X1], wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_DMD.name, {"eigenvalues": op.eigenvalues.discrete, "rank": r,
                                              "spectral_radius": op.stability()["spectral_radius"]},
                              {"residuals": op.residuals(), "kind": "ResDMD residual of each eigenpair (reduced "
                               "coordinates)"}, list(CARD_DMD.assumptions), cfg, diags, {},
                              limits=["a linear fit in the state: nonlinear dynamics need EDMD"],
                              reproducibility=rep, details={"operator": op, "X0": X0, "X1": X1})


class EDMD:
    """Extended DMD on a dictionary of observables; see the module docstring."""

    card = CARD_EDMD

    def __init__(self, dictionary: ObservableDictionary, *, ridge: float = 0.0, lag: int = 1, dt: float | None = None):
        self.dictionary, self.ridge, self.lag, self.dt = dictionary, ridge, lag, dt

    def _state_map(self, Px: np.ndarray, X0: np.ndarray) -> np.ndarray:
        m = len(self.dictionary)
        if self.dictionary.state_index is not None:
            B = np.zeros((m, X0.shape[1]))
            for j, i in enumerate(self.dictionary.state_index):
                B[i, j] = 1.0
            return B
        B, *_ = np.linalg.lstsq(Px, X0, rcond=None)
        return B

    def _finish(self, G, A, L, Px, X0, X1, dt, t0, how):
        m = G.shape[0]
        Gr = G + self.ridge * np.eye(m)
        K = np.linalg.pinv(Gr) @ A
        op = KoopmanOperator(K, self.dictionary, self._state_map(Px, X0), dt=dt, method=how, gram=(G, A, L))
        cond = float(np.linalg.cond(Gr))
        diags = _diagnostics(op, Px.shape[0], m, cond)
        cfg = {"dictionary": self.dictionary.names, "kind": self.dictionary.kind, "ridge": self.ridge, "lag": self.lag,
               "dt": dt}
        rep = reproducibility(CARD_EDMD.name, cfg, [X0, X1], wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_EDMD.name, {"eigenvalues": op.eigenvalues.discrete, "m": m,
                                               "spectral_radius": op.stability()["spectral_radius"]},
                              {"residuals": op.residuals(), "kind": "ResDMD residual of each eigenpair"},
                              list(CARD_EDMD.assumptions), cfg, diags, {},
                              limits=["eigenvalues with large residuals are not eigenvalues of the Koopman operator",
                                      "valid in the sampled region of state space only"],
                              reproducibility=rep, details={"operator": op, "X0": X0, "X1": X1})

    def fit(self, data) -> DynamicsResult:
        t0 = time.perf_counter()
        X0, X1, dt = snapshot_pairs(data, lag=self.lag)
        dt = self.dt if self.dt is not None else dt
        Px, Py = self.dictionary(X0), self.dictionary(X1)
        N = Px.shape[0]
        G, A, L = Px.T @ Px / N, Px.T @ Py / N, Py.T @ Py / N
        return self._finish(G, A, L, Px, X0, X1, dt, t0, "edmd")

    def fit_generator(self, X, dXdt, *, h: float = 1e-5, dt: float = 1.0) -> DynamicsResult:
        """Generator EDMD from states and their time derivatives: ``dpsi/dt = grad psi . f`` by central differences
        along ``f``; the returned operator is ``expm(L dt)`` for a step ``dt`` so it predicts like the discrete one,
        and ``generator()`` gives ``L``. Residuals are not defined for this fit (``None``)."""
        from scipy.linalg import expm

        t0 = time.perf_counter()
        X = np.atleast_2d(np.asarray(to_numpy(X), dtype=float))
        F = np.atleast_2d(np.asarray(to_numpy(dXdt), dtype=float))
        Px = self.dictionary(X)
        dP = (self.dictionary(X + h * F) - self.dictionary(X - h * F)) / (2 * h)
        N = Px.shape[0]
        G = Px.T @ Px / N
        Ad = Px.T @ dP / N
        Lg = np.linalg.pinv(G + self.ridge * np.eye(G.shape[0])) @ Ad
        res = self._finish(G, G @ expm(Lg * dt), G, Px, X, X, dt, t0, "gedmd")
        op = res.details["operator"]
        op.gram = None
        op.info["generator"] = Lg
        res.uncertainty = {"residuals": None, "kind": "not defined for generator EDMD"}
        res.details["generator"] = Lg
        res.estimate["generator_eigenvalues"] = np.linalg.eigvals(Lg)
        return res


# =========================================================================== reconstruction and analysis
@dataclass
class KoopmanReconstruction:
    """Errors of a fitted operator on trajectories: one-step (data-driven pairs) and multi-step (open-loop from the
    first state of each trajectory), relative to the RMS of the data."""
    one_step: float
    multi_step: np.ndarray
    horizon: int
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def evaluate(cls, operator: KoopmanOperator, trajectories: Sequence[np.ndarray] | np.ndarray,
                 horizon: int | None = None) -> KoopmanReconstruction:
        trajs = [np.atleast_2d(to_numpy(t.x if isinstance(t, Trajectory) else t)) for t in
                 (trajectories if isinstance(trajectories, (list, tuple, TrajectoryEnsemble)) else [trajectories])]
        trajs = [t if t.shape[0] > 1 else t.T for t in trajs]
        scale = math.sqrt(np.mean(np.concatenate([t.ravel() for t in trajs]) ** 2)) or 1.0
        one = []
        H = horizon or min(t.shape[0] for t in trajs) - 1
        multi = np.zeros(H + 1)
        for t in trajs:
            for n in range(t.shape[0] - 1):
                one.append(np.sum((operator.predict(t[n], 1)[1] - t[n + 1]) ** 2))
            pred = operator.predict(t[0], H)
            multi += np.sqrt(np.mean((pred - t[:H + 1]) ** 2, axis=1))
        multi /= len(trajs) * scale
        return cls(float(math.sqrt(np.mean(one) / trajs[0].shape[1]) / scale), multi, H)


class SpectralAnalyzer:
    """Analyses around a Koopman estimator: frequencies and growth rates, ranking modes, sensitivity of the
    eigenvalues to noise, order selection by held-out prediction, and comparison with a nonlinear state-space
    regression on the same dictionary."""

    @staticmethod
    def spectrum(result: DynamicsResult, *, residual_tol: float | None = None) -> list[dict[str, Any]]:
        op: KoopmanOperator = result.details["operator"]
        rows = []
        for mo in op.modes(result.details["X0"]):
            row = {"lambda": mo.lam, "abs": abs(mo.lam), "omega": mo.omega, "frequency": mo.frequency,
                   "growth_rate": mo.growth_rate, "amplitude": mo.amplitude, "residual": mo.residual}
            if residual_tol is not None and mo.residual is not None:
                row["trusted"] = mo.residual <= residual_tol
            rows.append(row)
        return sorted(rows, key=lambda r: -(r["amplitude"] if np.isfinite(r["amplitude"]) else 0))

    @staticmethod
    def noise_sensitivity(estimator, clean: np.ndarray, levels: Sequence[float], reference: Sequence[complex], *,
                          reps: int = 10, seed: int = 0) -> dict[str, Any]:
        """Eigenvalue error (nearest-match, mean over ``reps``) when white noise of relative std ``level`` is added to
        the states."""
        rng = np.random.default_rng(seed)
        scale = float(np.std(clean))
        out = {}
        for lv in levels:
            errs = []
            for _ in range(reps):
                noisy = clean + lv * scale * rng.standard_normal(clean.shape)
                op = estimator.fit(noisy).details["operator"]
                errs.append(float(np.max(op.eigenvalues.match(reference))))
            out[float(lv)] = {"mean_max_error": float(np.mean(errs)), "std": float(np.std(errs))}
        return out

    @staticmethod
    def select_order(make_estimator: Callable[[Any], Any], candidates: Sequence[Any], train, test, *,
                     horizon: int = 10) -> dict[str, Any]:
        """Held-out multi-step error for each candidate order (rank or dictionary); the smallest wins."""
        rows = {}
        for c in candidates:
            op = make_estimator(c).fit(train).details["operator"]
            rec = KoopmanReconstruction.evaluate(op, test, horizon)
            rows[str(c)] = {"one_step": rec.one_step, "horizon_error": float(rec.multi_step[-1])}
        best = min(rows, key=lambda k: rows[k]["horizon_error"])
        return {"rows": rows, "best": best}

    @staticmethod
    def compare_with_state_regression(dictionary: ObservableDictionary, train, test, *, horizon: int = 20,
                                      ridge: float = 0.0) -> dict[str, Any]:
        """EDMD (linear propagation in the lifted space) against a nonlinear state-space model on the same dictionary,
        ``x_{n+1} = psi(x_n) W`` iterated in the state (NARX-type regression)."""
        edmd = EDMD(dictionary, ridge=ridge).fit(train)
        op = edmd.details["operator"]
        X0, X1 = edmd.details["X0"], edmd.details["X1"]
        P = dictionary(X0)
        W, *_ = np.linalg.lstsq(P, X1, rcond=None)
        tests = test if isinstance(test, (list, tuple)) else [test]
        e_k, e_r = np.zeros(horizon + 1), np.zeros(horizon + 1)
        for t in tests:
            t = np.atleast_2d(to_numpy(t.x if isinstance(t, Trajectory) else t))
            pk = op.predict(t[0], horizon)
            x = t[0].copy()
            pr = [x]
            for _ in range(horizon):
                x = dictionary(x[None])[0] @ W
                pr.append(x)
            pr = np.array(pr)
            e_k += np.sqrt(np.mean((pk - t[:horizon + 1]) ** 2, axis=1))
            e_r += np.sqrt(np.mean((pr - t[:horizon + 1]) ** 2, axis=1))
        n = len(tests)
        return {"koopman_error": e_k / n, "state_regression_error": e_r / n, "horizon": horizon,
                "note": "same dictionary and data; the Koopman model is linear in the lifted space, the regression is "
                        "nonlinear in the state"}
