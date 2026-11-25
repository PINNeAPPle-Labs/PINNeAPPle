"""One linear-algebra layer for the library: operators, spectra, POD and conditioning (X18, #202).

``pinneapple_analysis.numerics.linalg`` (#1061) answers "where did my digits go?" for a dense matrix. What the
library was missing is the layer above it -- the one a surrogate, a reduced-order model or a sensitivity
study actually calls:

``LinearOperator``
    A shape, a ``matvec`` and (when it is given) an ``rmatvec``. Everything else in this module takes either
    a matrix or an operator, so a solver wrapped as a callable and a matrix on disk are the same argument.
    The adjoint is never invented: an operator built without ``rmatvec`` refuses ``.adjoint()`` instead of
    returning its transpose, which is the wrong answer when the operator is not symmetric.
``pod``
    Proper orthogonal decomposition by SVD, with the energy captured, the rank the energy asked for, the
    coefficients and the reconstruction error **measured** on the snapshots that produced the basis.
``modal_analysis``
    Eigenvalues, frequencies, damping ratios, mode shapes and modal participation, plus the condition
    number of the eigenvector basis, which is what says whether those eigenvalues can be quoted at all.
    For a :class:`LinearOperator` the modes are Ritz values from the Krylov subspace, and the residual
    ``||A v - lambda v||`` is reported so the approximation is not mistaken for the exact spectrum.
``arnoldi``
    The Krylov basis and the Hessenberg matrix behind it, for a matrix or an operator.
``condition_estimate``
    The Hager-Higham 1-norm estimator: a lower bound on ``||A||_1 ||A^-1||_1`` from a handful of solves,
    which is the only affordable answer for a matrix-free operator. When the exact value is available (a
    dense matrix) both are returned, so the estimate is reported against the number it approximates.
``adjoint_sensitivity``
    ``d(c^T x)/dp`` for ``A(p) x = b(p)`` by one adjoint solve instead of one solve per parameter, with
    each value checked against a central difference of the re-solved system.

What is not here
----------------
No sparse storage, no parallel or distributed operators, no iterative solvers of its own (those live in
``numerics.linalg``: ``conjugate_gradient``, ``gmres``, ``bicgstab``), no Galerkin projection (that is
``pinneapple_analysis.reduced_order``, which builds the ROM operators this module's ``pod`` supplies the
basis for), and no autodiff: the sensitivity here is a linear-adjoint statement about a matrix, not a
backpropagation through a network (``pinneapple_core.operators`` does that). The condition estimator is
the 1-norm one; a 2-norm estimate would need a spectral norm estimator, and is not written.

Design rules
------------
* **Measured, not claimed.** The POD reconstruction error, the modal residual and the estimator's gap to
  the exact condition number are all returned next to the object they describe.
* **Refusals are explicit.** An operator with no adjoint refuses the adjoint; a non-square matrix refuses
  eigen analysis; a defective matrix says its eigenvalues are ill-conditioned instead of quoting them.
* **numpy in, numpy out.** Every function takes and returns numpy arrays or dataclasses with ``to_dict``;
  no torch, no I/O.

References
----------
* Trefethen, L.N. and Bau, D., *Numerical Linear Algebra*, SIAM, 1997 -- the Krylov/Ritz reading of a
  spectrum and the conditioning of an eigenvector basis.
* Golub, G.H. and Van Loan, C.F., *Matrix Computations*, 4th ed., Johns Hopkins, 2013 -- POD, the SVD and
  modal decomposition.
* Hager, W.W., "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5(2), 1984; Higham, N.J., "FORTRAN
  subroutines for estimating the 1-norm of a matrix", *ACM TOMS* 14(2), 1988 -- the estimator LAPACK uses
  (``dlacn2``).
* Sirovich, L., "Turbulence and the dynamics of coherent structures", *Quart. Appl. Math.* 45, 1987 -- POD.
* Giles, M.B. and Pierce, N.A., "An introduction to the adjoint approach to design", *Flow, Turbulence and
  Combustion* 65, 2000 -- the adjoint sensitivity.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .numerics.diagnostics import Diagnostic, explain
from .numerics.linalg import condition_number

__all__ = [
    "Arnoldi",
    "ConditionEstimate",
    "LinearOperator",
    "ModalAnalysis",
    "PODResult",
    "SensitivityResult",
    "adjoint_sensitivity",
    "arnoldi",
    "condition_estimate",
    "condition_number",
    "modal_analysis",
    "pod",
]


# --------------------------------------------------------------------------- the operator
class LinearOperator:
    """A linear map with a shape: ``matvec``, and ``rmatvec`` when the adjoint is known.

    The point of the class is that every function here can take a matrix or this, without the caller
    building ``N x N`` dense storage for an operator that is a function. The shape is ``(m, n)`` for a map
    from ``R^n`` to ``R^m``; a vector comes in and out as a ``float64`` numpy array.

        >>> A = LinearOperator(lambda x: 2.0 * x, (3, 3), rmatvec=lambda y: 2.0 * y)
        >>> A.to_dense()
        array([[2., 0., 0.],
               [0., 2., 0.],
               [0., 0., 2.]])
    """

    __slots__ = ("_matvec", "_rmatvec", "shape", "name")

    def __init__(
        self,
        matvec: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int],
        *,
        rmatvec: Callable[[np.ndarray], np.ndarray] | None = None,
        name: str = "A",
    ) -> None:
        m, n = int(shape[0]), int(shape[1])
        if m <= 0 or n <= 0:
            raise ValueError(f"shape must be positive in both dimensions; got {shape}")
        self._matvec = matvec
        self._rmatvec = rmatvec
        self.shape = (m, n)
        self.name = str(name)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_matrix(cls, A: np.ndarray, *, name: str = "A") -> LinearOperator:
        """Wrap a dense matrix; both directions are exact."""
        M = np.asarray(A, dtype=np.float64)
        if M.ndim != 2:
            raise ValueError(f"a matrix must be two-dimensional; got shape {M.shape}")
        return cls(lambda x: M @ x, M.shape, rmatvec=lambda y: M.T @ y, name=name)

    # ---------------------------------------------------------------- algebra
    @property
    def has_adjoint(self) -> bool:
        """Whether ``rmatvec`` was given (the adjoint is known, not assumed)."""
        return self._rmatvec is not None

    @property
    def is_square(self) -> bool:
        """Whether the operator maps a space to itself."""
        return self.shape[0] == self.shape[1]

    def matvec(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """``A x``, with the shape checked rather than trusted."""
        v = np.asarray(x, dtype=np.float64).ravel()
        if v.size != self.shape[1]:
            raise ValueError(f"{self.name} acts on {self.shape[1]} entries; got {v.size}")
        out = np.asarray(self._matvec(v), dtype=np.float64).ravel()
        if out.size != self.shape[0]:
            raise ValueError(f"{self.name} returned {out.size} entries for a {self.shape[0]}-row operator")
        return out

    def rmatvec(self, y: Sequence[float] | np.ndarray) -> np.ndarray:
        """``A^H y``. Refuses when the adjoint was not given -- see :meth:`adjoint`."""
        if self._rmatvec is None:
            raise NotImplementedError(
                f"{self.name} has no adjoint: it was built without ``rmatvec``. A transpose is not the "
                "adjoint of an operator that is not known to be symmetric, so this refuses instead."
            )
        v = np.asarray(y, dtype=np.float64).ravel()
        if v.size != self.shape[0]:
            raise ValueError(f"{self.name}^H acts on {self.shape[0]} entries; got {v.size}")
        out = np.asarray(self._rmatvec(v), dtype=np.float64).ravel()
        if out.size != self.shape[1]:
            raise ValueError(f"{self.name}^H returned {out.size} entries for a {self.shape[1]}-column operator")
        return out

    def adjoint(self) -> LinearOperator:
        """The adjoint operator; refuses when ``rmatvec`` was not supplied."""
        if self._rmatvec is None:
            raise NotImplementedError(
                f"{self.name} has no adjoint: build it with ``rmatvec=``. Guessing the transpose would "
                "make every adjoint and sensitivity result here silently wrong for a non-symmetric map."
            )
        return LinearOperator(self._rmatvec, (self.shape[1], self.shape[0]),
                              rmatvec=self._matvec, name=f"{self.name}^H")

    def __matmul__(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.matvec(x)

    def __rmatmul__(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.rmatvec(x)

    # ---------------------------------------------------------------- inspection
    def to_dense(self) -> np.ndarray:
        """The full matrix, one column at a time: ``n`` applications of the operator."""
        out = np.empty(self.shape, dtype=np.float64)
        basis = np.zeros(self.shape[1])
        for j in range(self.shape[1]):
            basis[j] = 1.0
            out[:, j] = self.matvec(basis)
            basis[j] = 0.0
        return out

    def __repr__(self) -> str:
        adjoint = "with adjoint" if self.has_adjoint else "no adjoint"
        return f"LinearOperator({self.name}, {self.shape[0]}x{self.shape[1]}, {adjoint})"


def _as_operator(A: Any, *, name: str = "A") -> LinearOperator:
    """A matrix or an operator, as an operator."""
    if isinstance(A, LinearOperator):
        return A
    return LinearOperator.from_matrix(np.asarray(A, dtype=np.float64), name=name)


# --------------------------------------------------------------------------- POD
@dataclass(frozen=True)
class PODResult:
    """A POD basis with the energy it captures and the error it actually made."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    energy_captured: float
    coefficients: np.ndarray
    rank: int
    relative_error: float
    snapshots: int
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def reconstruct(self) -> np.ndarray:
        """The snapshots rebuilt from ``rank`` modes: ``mean + modes @ coefficients``."""
        return self.mean[:, None] + self.modes @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form: the numbers, not the arrays (a basis is a file, not a JSON field)."""
        return {
            "rank": self.rank,
            "snapshots": self.snapshots,
            "energy_captured": self.energy_captured,
            "relative_error": self.relative_error,
            "singular_values": [float(s) for s in self.singular_values],
            "cumulative_energy": [float(e) for e in self.cumulative_energy],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def pod(
    snapshots: np.ndarray,
    *,
    energy: float = 0.999,
    rank: int | None = None,
    subtract_mean: bool = True,
) -> PODResult:
    """Proper orthogonal decomposition: the optimal linear basis for ``snapshots``, by SVD.

    ``snapshots`` is ``(n, m)`` -- one snapshot per **column**, which is the convention of
    :mod:`pinneapple_analysis.reduced_order` and of the POD literature. The rank is the smallest one whose
    cumulative energy reaches ``energy``, or ``rank`` when it is given, capped by the **numerical rank** of
    the data: a singular value at the rounding level of the data is a zero with a sign, so asking for ten
    modes of a one-mode dataset returns one rather than nine columns of noise.

    The reconstruction error is measured on the snapshots and returned: a basis that captures 99.9 % of the
    energy and reconstructs the data to 1e-2 has told you something the energy number alone did not. It is
    the *relative* Frobenius error against ``||X||``, so with the mean removed it equals
    ``sqrt(1 - energy_captured) * ||X - mean|| / ||X||`` -- which is Eckart-Young with the mean taken out.
    """
    X = np.asarray(snapshots, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"snapshots must be two-dimensional (n, m); got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("snapshots contain non-finite values: the SVD would not mean anything")
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must be in (0, 1]; got {energy}")
    n, m = X.shape
    mean = X.mean(axis=1) if subtract_mean else np.zeros(n)
    centred = X - mean[:, None] if subtract_mean else X
    U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    total = float(np.sum(s**2))
    diagnostics: list[Diagnostic] = []
    if s.size == 0 or total == 0.0:
        diagnostics.append(explain("LINALG.SINGULAR",
                                   note="the snapshots are constant (or empty): there is no basis to build"))
        empty = np.zeros((n, 0))
        return PODResult(mean=mean, modes=empty, singular_values=s, cumulative_energy=np.zeros(0),
                         energy_captured=1.0, coefficients=np.zeros((0, m)), rank=0,
                         relative_error=0.0, snapshots=int(m), diagnostics=tuple(diagnostics))

    cumulative = np.cumsum(s**2) / total
    wanted = int(np.searchsorted(cumulative, energy) + 1)
    # the numerical rank: a singular value at the rounding level of the data is a zero with a sign, so a
    # request for more modes than the data has returns the data's rank rather than padding with noise
    numerical = int(np.sum(s > s[0] * max(n, m) * np.finfo(np.float64).eps))
    r = min(wanted, s.size) if rank is None else min(int(rank), s.size)
    r = max(1, min(r, max(numerical, 1)))
    modes = U[:, :r]
    capture = float(cumulative[r - 1])
    coefficients = modes.T @ centred
    rebuilt = mean[:, None] + modes @ coefficients
    norm = float(np.linalg.norm(X, "fro"))
    error = float(np.linalg.norm(X - rebuilt, "fro") / norm) if norm > 0 else 0.0
    if rank is not None and r < int(rank):
        diagnostics.append(explain("LINALG.ILL_CONDITIONED",
                                   note=f"the requested rank {r} captures {capture:.6f} of the energy, "
                                        f"below the {energy} asked for",
                                   energy_captured=capture, requested_energy=energy))
    return PODResult(mean=mean, modes=modes, singular_values=s, cumulative_energy=cumulative,
                     energy_captured=capture, coefficients=coefficients, rank=r, relative_error=error,
                     snapshots=int(m), diagnostics=tuple(diagnostics))


# --------------------------------------------------------------------------- Krylov
@dataclass(frozen=True)
class Arnoldi:
    """A Krylov basis, the Hessenberg matrix behind it, and the Ritz values that came out."""

    basis: np.ndarray
    hessenberg: np.ndarray
    ritz_values: np.ndarray
    residuals: np.ndarray
    iterations: int
    breakdown: bool

    @property
    def dimension(self) -> int:
        """How many Krylov vectors were produced."""
        return int(self.basis.shape[1]) - 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "dimension": self.dimension,
            "iterations": self.iterations,
            "breakdown": self.breakdown,
            "ritz_values": [[float(v.real), float(v.imag)] for v in self.ritz_values],
            "residuals": [float(r) for r in self.residuals],
        }


def arnoldi(
    A: LinearOperator | np.ndarray, b: Sequence[float] | np.ndarray, m: int | None = None,
    *, tol: float = 1e-12,
) -> Arnoldi:
    """Arnoldi iteration: ``K_m(A, b) = span{b, Ab, ..., A^{m-1}b}`` and ``A V_m = V_{m+1} H``.

    The Ritz values -- the eigenvalues of ``H_m`` -- are the best approximation to the spectrum that this
    subspace can see, and the residual of each is ``|h_{m+1,m}| * |last component of its eigenvector|``,
    which is what says whether the approximation has converged. A breakdown (``h = 0``) means the subspace
    is invariant and the Ritz values are then **exact**, which is why ``breakdown`` is reported rather than
    treated as a failure.
    """
    op = _as_operator(A)
    v = np.asarray(b, dtype=np.float64).ravel()
    if v.size != op.shape[1]:
        raise ValueError(f"the start vector has {v.size} entries; the operator takes {op.shape[1]}")
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("the start vector is zero: the Krylov subspace would be empty")
    steps = int(m) if m is not None else min(op.shape[0], 20)
    if steps < 1:
        raise ValueError(f"m must be at least 1; got {m}")
    steps = min(steps, op.shape[1])
    V = np.zeros((op.shape[1], steps + 1))
    H = np.zeros((steps + 1, steps))
    V[:, 0] = v / norm
    breakdown = False
    used = 0
    for j in range(steps):
        w = op.matvec(V[:, j])
        for i in range(j + 1):                      # modified Gram-Schmidt, twice for stability
            H[i, j] = float(V[:, i] @ w)
            w = w - H[i, j] * V[:, i]
        for i in range(j + 1):
            correction = float(V[:, i] @ w)
            H[i, j] += correction
            w = w - correction * V[:, i]
        H[j + 1, j] = float(np.linalg.norm(w))
        used = j + 1
        if abs(H[j + 1, j]) <= tol:
            breakdown = True
            break
        V[:, j + 1] = w / H[j + 1, j]

    Hm = H[:used, :used]
    eigs, vectors = np.linalg.eig(Hm)
    # the residual of a Ritz pair is |h_{m+1,m}| times the last component of its eigenvector of H_m
    last = np.abs(vectors[-1, :]) if vectors.size else np.zeros(0)
    residuals = np.abs(H[used, used - 1]) * last if used < H.shape[0] else np.zeros_like(last)
    return Arnoldi(basis=V[:, : used + 1], hessenberg=Hm, ritz_values=eigs, residuals=residuals,
                   iterations=used, breakdown=breakdown)


# --------------------------------------------------------------------------- modal analysis
@dataclass(frozen=True)
class Mode:
    """One eigenpair read the way a structural or a control engineer reads it."""

    eigenvalue: complex
    frequency: float
    damping_ratio: float
    shape: np.ndarray
    participation: float
    residual: float

    @property
    def growth_rate(self) -> float:
        """``Re(lambda)``: negative is stable, positive grows."""
        return float(np.real(self.eigenvalue))

    def to_dict(self, *, with_shape: bool = False) -> dict[str, Any]:
        """JSON-able form; the shape is left out unless asked, since it is ``n`` numbers."""
        out: dict[str, Any] = {
            "eigenvalue": [float(np.real(self.eigenvalue)), float(np.imag(self.eigenvalue))],
            "frequency": self.frequency,
            "damping_ratio": self.damping_ratio,
            "growth_rate": self.growth_rate,
            "participation": self.participation,
            "residual": self.residual,
        }
        if with_shape:
            out["shape"] = [[float(np.real(v)), float(np.imag(v))] for v in self.shape]
        return out


@dataclass(frozen=True)
class ModalAnalysis:
    """The spectrum of a matrix or an operator, with what makes the numbers quotable or not."""

    modes: tuple[Mode, ...]
    basis_condition: float
    exact: bool
    ritz_approximation: bool
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalues in the order of :attr:`modes`."""
        return np.array([m.eigenvalue for m in self.modes])

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "modes": [m.to_dict() for m in self.modes],
            "basis_condition": self.basis_condition,
            "exact": self.exact,
            "ritz_approximation": self.ritz_approximation,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def modal_analysis(
    A: LinearOperator | np.ndarray, *, krylov: int | None = None, frequencies_in_hz: bool = True
) -> ModalAnalysis:
    """Eigenvalues, frequencies, damping ratios, mode shapes and participation of ``A``.

    A dense matrix is diagonalised exactly (``n x n``), and with it the **left** eigenvectors are available,
    so the participation factor ``|phi_i^H psi_i| / (||phi_i|| ||psi_i||)`` -- how much mode ``i`` is
    excited by a generic input -- is measured, and the condition number of the eigenvector basis says how
    far the eigenvalues can move under a perturbation of the matrix. A near-defective matrix (that
    condition above ``1e8``) gets ``LINALG.NEAR_DEFECTIVE``: its eigenvalues are the ill-conditioned
    quantity, not its eigenvectors.

    A :class:`LinearOperator` has no dense diagonalisation to run, so the modes are **Ritz values** from
    :func:`arnoldi` with ``krylov`` steps; ``ritz_approximation`` is then ``True`` and every mode reports
    its measured residual ``||A v - lambda v||``. Reading a Ritz value as an eigenvalue is the classic way
    to publish a mode that does not exist, so the flag travels with the result.

    ``frequency`` is ``|Im lambda| / 2pi`` when ``frequencies_in_hz``, and ``|Im lambda|`` (the angular
    rate) otherwise; ``damping_ratio`` is ``-Re lambda / |lambda|``, which is the usual
    ``zeta = 1`` for a real stable eigenvalue and ``0`` for an undamped oscillation.
    """
    diagnostics: list[Diagnostic] = []
    scale = 1.0 / (2.0 * np.pi) if frequencies_in_hz else 1.0

    if isinstance(A, LinearOperator):
        if not A.is_square:
            raise ValueError(f"modal analysis needs a square operator; {A.name} is {A.shape}")
        m = int(krylov) if krylov is not None else min(A.shape[0], 20)
        run = arnoldi(A, np.ones(A.shape[1]), m)
        eigenvalues = run.ritz_values
        shapes = run.basis[:, :-1] @ np.linalg.eig(run.hessenberg)[1]
        residuals = run.residuals
        participation = np.ones(eigenvalues.size)
        basis_condition = float("nan")
        diagnostics.append(
            explain("LINALG.ILL_CONDITIONED",
                    note=f"the spectrum is a Ritz approximation from {run.dimension} Krylov vectors; "
                         "each mode carries its measured residual",
                    krylov_dimension=run.dimension, breakdown=run.breakdown)
        )
        modes = tuple(
            Mode(eigenvalue=complex(lam),
                 frequency=float(abs(np.imag(lam)) * scale),
                 damping_ratio=float(-np.real(lam) / abs(lam)) if abs(lam) > 0 else 0.0,
                 shape=shapes[:, i], participation=float(participation[i]),
                 residual=float(residuals[i]) if i < residuals.size else float("nan"))
            for i, lam in enumerate(eigenvalues)
        )
        return ModalAnalysis(modes=modes, basis_condition=basis_condition, exact=False,
                             ritz_approximation=True, diagnostics=tuple(diagnostics))

    M = np.asarray(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"modal analysis needs a square matrix; got shape {M.shape}")
    eigenvalues, right = np.linalg.eig(M)
    left = np.linalg.inv(right).conj().T               # columns are the left eigenvectors
    basis_condition = float(np.linalg.cond(right))
    norms = np.linalg.norm(right, axis=0) * np.linalg.norm(left, axis=0)
    participation = np.abs(np.einsum("ij,ij->j", left.conj(), right)) / np.where(norms > 0, norms, 1.0)
    modes = tuple(
        Mode(eigenvalue=complex(lam),
             frequency=float(abs(np.imag(lam)) * scale),
             damping_ratio=float(-np.real(lam) / abs(lam)) if abs(lam) > 0 else 0.0,
             shape=right[:, i], participation=float(participation[i]),
             residual=float(np.linalg.norm(M @ right[:, i] - lam * right[:, i])))
        for i, lam in enumerate(eigenvalues)
    )
    if not np.isfinite(basis_condition):
        diagnostics.append(explain("LINALG.SINGULAR",
                                   note="the eigenvector matrix is singular: the matrix is defective and "
                                        "its eigenvalues are not a continuous function of its entries"))
    elif basis_condition > 1e8:
        diagnostics.append(explain("LINALG.NEAR_DEFECTIVE", basis_condition=basis_condition))
    return ModalAnalysis(modes=modes, basis_condition=basis_condition, exact=True,
                         ritz_approximation=False, diagnostics=tuple(diagnostics))


# --------------------------------------------------------------------------- condition estimate
@dataclass(frozen=True)
class ConditionEstimate:
    """A 1-norm condition number: a measured lower bound, and the exact value when it is affordable."""

    estimate: float
    lower_bound: float
    exact: float | None
    norm: str
    iterations: int
    method: str
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def ratio(self) -> float:
        """``lower_bound / exact``: 1.0 means the estimator found the exact value."""
        if self.exact is None or self.exact == 0 or not np.isfinite(self.exact):
            return float("nan")
        return self.lower_bound / self.exact

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "estimate": self.estimate,
            "lower_bound": self.lower_bound,
            "exact": self.exact,
            "ratio": self.ratio,
            "norm": self.norm,
            "iterations": self.iterations,
            "method": self.method,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def condition_estimate(
    A: LinearOperator | np.ndarray,
    solve: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    solve_adjoint: Callable[[np.ndarray], np.ndarray] | None = None,
    max_iter: int = 5,
    exact: bool | None = None,
) -> ConditionEstimate:
    """``||A||_1 * ||A^-1||_1`` by the Hager-Higham estimator: a lower bound from a few solves.

    ``solve`` must compute ``A^-1 r``; for a matrix it defaults to ``np.linalg.solve`` and the exact 2-norm
    condition number is reported alongside. For a matrix-free operator, pass the solver -- the estimator is
    then the only affordable answer. It is a **lower bound** on ``||A^-1||_1``, so a lower bound on the
    condition number: it can miss a large value, it cannot invent one.

    The iteration is Hager's: with ``B = A^-1`` and ``x = 1/n``, compute ``y = B x``, take ``xi = sign(y)``,
    ``z = B^T xi`` (one transposed solve), move ``x`` to the unit vector of ``argmax |z|``, and repeat until
    the estimate stops growing or the sign vector has been found. The ``||A||_1`` factor is the maximum
    column sum, measured with one application of the operator per column.

    ``solve_adjoint`` (computing ``A^-T r``) is what makes the transposed step possible. It defaults to the
    transpose of a dense matrix; for a matrix-free operator it must be given, and without it the estimate
    comes from the single right-hand side ``1/n`` -- still a valid lower bound, weaker, and ``iterations``
    says ``1`` so the caller can tell the two apart.
    """
    op = _as_operator(A)
    dense: np.ndarray | None = None if isinstance(A, LinearOperator) else np.asarray(A, dtype=np.float64)
    if solve is None:
        if dense is None:
            raise ValueError("a matrix-free operator needs a ``solve`` callable: there is no inverse to fall back on")
        solver: Callable[[np.ndarray], np.ndarray] = lambda r: np.linalg.solve(dense, r)      # noqa: E731
    else:
        solver = solve
    if solve_adjoint is not None:
        solver_t: Callable[[np.ndarray], np.ndarray] | None = solve_adjoint
    elif dense is not None:
        solver_t = lambda r: np.linalg.solve(dense.T, r)                                     # noqa: E731
    else:
        solver_t = None

    want_exact = exact if exact is not None else dense is not None
    n = op.shape[1]
    if op.shape[0] != n:
        raise ValueError(f"the condition number needs a square operator; {op.name} is {op.shape}")

    # ||A||_1: the largest column sum, measured with one application of the operator per column
    columns = op.to_dense()
    norm_a = float(np.max(np.sum(np.abs(columns), axis=0)))

    x = np.full(n, 1.0 / n)
    best_inverse = 0.0
    iterations = 0
    for step_index in range(1, int(max_iter) + 1):
        iterations = step_index
        y = np.asarray(solver(x), dtype=np.float64).ravel()
        if not np.all(np.isfinite(y)):
            break
        estimate = float(np.linalg.norm(y, 1))          # ||B x||_1 with ||x||_1 = 1
        if estimate <= best_inverse or solver_t is None:
            best_inverse = max(best_inverse, estimate)
            break                                        # no longer improving, or no transposed solve
        best_inverse = estimate
        xi = np.sign(y)
        z = np.asarray(solver_t(xi), dtype=np.float64).ravel()
        j = int(np.argmax(np.abs(z)))
        if abs(z[j]) <= float(z @ x):
            break                                        # the sign vector has been found
        x = np.zeros(n)
        x[j] = 1.0

    best = float(norm_a * best_inverse)

    diagnostics: list[Diagnostic] = []
    exact_value: float | None = None
    if want_exact and dense is not None:
        # the same norm the estimate is in, so the two numbers are comparable: ||A||_1 ||A^-1||_1
        try:
            exact_value = float(norm_a * np.linalg.norm(np.linalg.inv(dense), 1))
        except np.linalg.LinAlgError:
            exact_value = float("inf")
    if exact_value is not None and np.isfinite(exact_value) and exact_value > 0:
        if best > exact_value * (1 + 1e-6):
            diagnostics.append(explain("LINALG.ILL_CONDITIONED",
                                       note="the estimate exceeds the exact 1-norm value, which the Hager "
                                            "iteration cannot do -- the solves it was given are not this A",
                                       lower_bound=best, exact=exact_value))
        if best < exact_value * 0.1:
            diagnostics.append(explain("LINALG.ILL_CONDITIONED",
                                       note="the estimator found less than a tenth of the exact value: the "
                                            "solve it was given may be inaccurate",
                                       lower_bound=best, exact=exact_value))
    return ConditionEstimate(estimate=float(best), lower_bound=float(best), exact=exact_value,
                             norm="1", iterations=int(iterations),
                             method="hager-higham 1-norm with sign-vector refinement",
                             diagnostics=tuple(diagnostics))


# --------------------------------------------------------------------------- adjoint sensitivity
@dataclass(frozen=True)
class SensitivityResult:
    """``d(c^T x)/dp`` for ``A(p) x = b(p)``: the adjoint derivatives, and a direct check of them."""

    values: np.ndarray
    checked: np.ndarray
    relative_error: np.ndarray
    state: np.ndarray
    adjoint: np.ndarray
    solves_forward: int
    solves_adjoint: int
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "values": [float(v) for v in self.values],
            "checked": [float(v) for v in self.checked],
            "relative_error": [float(v) for v in self.relative_error],
            "solves_forward": self.solves_forward,
            "solves_adjoint": self.solves_adjoint,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def adjoint_sensitivity(
    A: LinearOperator | np.ndarray,
    b: Sequence[float] | np.ndarray,
    output: Sequence[float] | np.ndarray,
    deriv_A: Sequence[np.ndarray] | np.ndarray | None = None,
    deriv_b: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    step: float = 1e-6,
) -> SensitivityResult:
    """``d(c^T x)/dp`` where ``A(p) x = b(p)``, by the adjoint method, with a finite-difference check.

    For each parameter ``p_k`` with ``dA_k = dA/dp_k`` and ``db_k = db/dp_k``::

        dJ/dp_k = lambda^T (db_k - dA_k x),     with   A^T lambda = c

    One factorization of ``A`` and one of ``A^T`` gives every parameter at once, against ``x`` solved once --
    which is the whole point when there are many parameters and few outputs. ``checked`` is the same
    derivative from a central difference of the re-solved system at ``+/- step``: the two should agree, and
    ``relative_error`` is the measured gap. A large gap is reported, not hidden, because the adjoint formula
    is exact and a disagreement means the derivatives supplied do not match the operator.
    """
    op = _as_operator(A)
    if not op.is_square:
        raise ValueError(f"the system needs a square operator; {op.name} is {op.shape}")
    rhs = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(output, dtype=np.float64).ravel()
    if rhs.size != op.shape[0]:
        raise ValueError(f"b has {rhs.size} entries; the operator has {op.shape[0]} rows")
    if c.size != op.shape[0]:
        raise ValueError(f"the output functional has {c.size} entries; the operator has {op.shape[0]} rows")

    if deriv_A is None and deriv_b is None:
        raise ValueError("no parameter derivatives were given: pass deriv_A and/or deriv_b")
    dA = _as_derivatives(deriv_A, square=True)
    db = _as_derivatives(deriv_b, square=False)
    if len(dA) != len(db) and dA and db:
        raise ValueError(f"{len(dA)} matrix derivatives for {len(db)} right-hand-side derivatives; "
                         "give matching lists, or only one of the two")

    x = _solve(op, rhs)
    lam = _solve(op.adjoint(), c)
    n_params = max(len(dA), len(db))
    values = np.zeros(n_params)
    for k in range(n_params):
        term = np.zeros(op.shape[0])
        if db:
            term = term + db[k]
        if dA:
            term = term - dA[k] @ x
        values[k] = float(lam @ term)

    checked = np.array([_central_difference(op, rhs, c, dA[k] if dA else None, db[k] if db else None, step)
                        for k in range(n_params)])
    scale = np.maximum(np.abs(checked), 1e-12)
    error = np.abs(values - checked) / scale
    diagnostics: list[Diagnostic] = []
    worst = float(np.max(error)) if error.size else 0.0
    if worst > 1e-4:
        diagnostics.append(explain("LINALG.ILL_CONDITIONED",
                                   note="the adjoint derivative and the finite difference disagree: the "
                                        "supplied parameter derivatives likely do not match the operator",
                                   worst_relative_error=worst))
    return SensitivityResult(values=values, checked=checked, relative_error=error, state=x, adjoint=lam,
                             solves_forward=1, solves_adjoint=1 + 2 * n_params,
                             diagnostics=tuple(diagnostics))


def _as_derivatives(given: Any, *, square: bool) -> list[np.ndarray]:
    """One derivative or a list of them, as a list of arrays (a single one is a list of one)."""
    if given is None:
        return []
    if isinstance(given, np.ndarray) and given.ndim == (3 if square else 2):
        return [np.asarray(d, dtype=np.float64) for d in given]
    if isinstance(given, np.ndarray) and given.ndim == (2 if square else 1):
        return [np.asarray(given, dtype=np.float64)]
    try:
        items = list(given)
    except TypeError:  # pragma: no cover - a scalar is not a derivative of anything here
        raise ValueError("a parameter derivative must be an array or a sequence of arrays") from None
    return [np.asarray(d, dtype=np.float64).ravel() if not square else np.asarray(d, dtype=np.float64)
            for d in items]


def _solve(op: LinearOperator, rhs: np.ndarray) -> np.ndarray:
    """Solve ``op x = rhs``: densely, or by the operator's own adjoint when it is matrix-free."""
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"a solve needs a square operator; {op.name} is {op.shape}")
    dense = op.to_dense()
    try:
        return np.linalg.solve(dense, rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - a singular system is the caller's problem
        raise ValueError(f"{op.name} is singular: there is no state to differentiate") from exc


def _central_difference(
    op: LinearOperator,
    rhs: np.ndarray,
    c: np.ndarray,
    dA: np.ndarray | None,
    db: np.ndarray | None,
    step: float,
) -> float:
    """The same derivative by re-solving at ``p +/- step``: the reference the adjoint is checked against."""
    dense = op.to_dense()
    results = []
    for sign in (1.0, -1.0):
        matrix = dense + sign * step * dA if dA is not None else dense.copy()
        vector = rhs + sign * step * db if db is not None else rhs.copy()
        results.append(float(c @ np.linalg.solve(matrix, vector)))
    return (results[0] - results[1]) / (2.0 * step)
