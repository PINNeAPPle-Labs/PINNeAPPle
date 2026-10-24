"""Condensation with a learned block operator, error indicators and an error-driven fallback to exact condensation.

When every block differs a little (voided microbumps, a TIM contact that degrades across the die, TSV quality), the
identical-block reuse of :mod:`.substructure` is gone: each block would need its own factorisation. Here the
reduced block operator (the interface Schur complement ``S_r``, the condensed load map ``W_r`` and the observable
maps ``O_t``, ``O_p`` on ``m`` interface modes per face) is learned as a function of the block parameters ``theta``
from a few exact condensations: a POD of the operator entries (as a correction to the operator of the block at the
centre of the parameter box, so the large, parameter-independent part is exact) and a Gaussian-process regression of
the POD coefficients (:class:`pinneapple_neural.architectures.rom.parametric.ParametricPOD`, which gives a predictive standard
deviation).

The learned operator is never trusted blindly. :func:`solve_adaptive`:

1. predicts every block; blocks outside the training box or with a relative predictive deviation above ``tol_std``
   are condensed exactly instead;
2. solves the global interface problem;
3. checks a set of blocks against their exact condensation *at the solved state*: the hottest blocks (the goal is
   the peak junction temperature), plus a random audit sample. The check is the junction-temperature discrepancy
   of the block (K), and the global effect of the heat-flow residual ``r_b`` the learned operator injects at its ports:
   ``A^-1 r_b`` with the already factorised global matrix ``A`` (one triangular solve per checked block), mapped to
   junction temperatures. Blocks whose discrepancy exceeds ``tol_T`` are replaced by their exact operator and the
   problem is solved again;
4. stops when every checked block passes. The audit sample gives an estimate for the blocks never checked: their
   number times the mean (and the largest) per-block effect seen in the audit, an additive estimate of the error
   that remains (reported, not a guarantee).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import fv
from .package import BASE, Stack, Variant
from .substructure import FLUXES, block_model, interface_modes

#: default block parameters: microbump conductivity factor (voids), contact-resistance factor of both TIM contacts,
#: TSV conductivity factor
DEFAULT_BOUNDS = {"ubump_k": (0.3, 1.0), "tim_R": (0.5, 3.0), "tsv_k": (0.5, 1.0)}


def default_variant(theta: np.ndarray) -> Variant:
    a, b, c = (float(x) for x in theta)
    return Variant(inclusion_k=(("ubump", a), ("interposer", c)), contact=(("die", "tim", b), ("tim", "lid", b)))


def reduced_block(stack: Stack, variant: Variant, Phi: np.ndarray) -> dict[str, np.ndarray]:
    """Exact reduced operator of one block on the interface modes ``Phi`` (no cache: one factorisation)."""
    bm = block_model(stack, variant, cache=False)
    return {"Sr": Phi.T @ bm.S @ Phi, "Wr": Phi.T @ bm.W, "Otr": bm.O_t @ Phi, "Op": bm.O_p}


_PARTS = ("Sr", "Wr", "Otr", "Op")


def _flatten(ops: list[dict], scales: dict | None = None):
    d = ops[0]["Sr"].shape[0]
    iu = np.triu_indices(d)
    cols = {"Sr": np.stack([o["Sr"][iu] for o in ops])}
    for k in _PARTS[1:]:
        cols[k] = np.stack([o[k].ravel() for o in ops])
    if scales is None:
        scales = {k: float(np.sqrt((v ** 2).mean())) for k, v in cols.items()}
    return np.concatenate([cols[k] / scales[k] for k in _PARTS], 1), scales


@dataclass
class LearnedBlocks:
    """A learned map ``theta -> reduced block operator``. ``fit()`` condenses ``n_train`` Latin-hypercube samples
    of ``bounds`` exactly and trains the regression."""
    stack: Stack
    m: int = 6
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    to_variant: Callable[[np.ndarray], Variant] = default_variant
    n_train: int = 48
    regressor: str = "gpr"
    seed: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        base = block_model(self.stack, BASE)
        basis = {o: interface_modes(base, o, self.m) for o in ("x", "y")}
        self.Phi = sla.block_diag(basis["x"], basis["x"], basis["y"], basis["y"])
        self.d = self.Phi.shape[1]
        self.n2 = self.stack.n ** 2
        self.n_obs = base.O.shape[0]

    @property
    def lo(self) -> np.ndarray:
        return np.array([v[0] for v in self.bounds.values()])

    @property
    def hi(self) -> np.ndarray:
        return np.array([v[1] for v in self.bounds.values()])

    def exact(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        return reduced_block(self.stack, self.to_variant(theta), self.Phi)

    def fit(self) -> LearnedBlocks:
        from pinneapple_neural.architectures.rom.parametric import ParametricPOD, latin_hypercube
        t0 = time.perf_counter()
        P, _ = latin_hypercube(self.n_train, self.bounds, seed=self.seed)
        self.centre = 0.5 * (self.lo + self.hi)
        self.base_ops = self.exact(self.centre)
        ops = [self._minus(self.exact(th), self.base_ops) for th in P]
        self.timings["train_condensations"] = time.perf_counter() - t0
        X, self.scales = _flatten(ops)
        t1 = time.perf_counter()
        self.rom = ParametricPOD(r=24, energy=1 - 1e-12, regressor=self.regressor, seed=self.seed).fit(P, X)
        self.timings["fit"] = time.perf_counter() - t1
        self.P_train = P
        return self

    @staticmethod
    def _minus(a, b):
        return {k: a[k] - b[k] for k in _PARTS}

    def _unflatten(self, X: np.ndarray) -> dict[str, np.ndarray]:
        d, n2, no = self.d, self.n2, self.n_obs
        nu = d * (d + 1) // 2
        sizes = [nu, d * n2, no * d, no * n2]
        parts = np.split(X, np.cumsum(sizes)[:-1], axis=1)
        N = len(X)
        Sr = np.zeros((N, d, d))
        iu = np.triu_indices(d)
        Sr[:, iu[0], iu[1]] = parts[0] * self.scales["Sr"]
        Sr = Sr + np.triu(Sr, 1).transpose(0, 2, 1)
        b = self.base_ops
        return {"Sr": b["Sr"] + Sr, "Wr": b["Wr"] + parts[1].reshape(N, d, n2) * self.scales["Wr"],
                "Otr": b["Otr"] + parts[2].reshape(N, no, d) * self.scales["Otr"],
                "Op": b["Op"] + parts[3].reshape(N, no, n2) * self.scales["Op"]}

    def predict(self, theta: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Operators of ``(N, p)`` blocks, their relative predictive deviation (norm of the std over the norm of the
        predicted correction to the centre block, in the scaled entries) and an out-of-box flag."""
        theta = np.atleast_2d(theta)
        X, std = self.rom.predict(theta, return_std=True)
        rel = (np.linalg.norm(std, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-300)
               if std is not None else np.zeros(len(theta)))
        ood = ((theta < self.lo - 1e-12) | (theta > self.hi + 1e-12)).any(1)
        return self._unflatten(X), rel, ood


def _face_dofs(shape: tuple[int, int], m: int) -> np.ndarray:
    nbx, nby = shape
    I, J = np.meshgrid(np.arange(nbx), np.arange(nby), indexing="ij")
    nxf = (nbx + 1) * nby
    F = np.stack([I * nby + J, (I + 1) * nby + J, nxf + I * (nby + 1) + J, nxf + I * (nby + 1) + J + 1], -1)
    return (F[..., :, None] * m + np.arange(m)).reshape(nbx * nby, 4 * m), nxf + nbx * (nby + 1)


def assemble_blocks(Sr: np.ndarray, dofs: np.ndarray, n: int) -> sp.csc_matrix:
    """Global interface matrix from per-block reduced operators ``(N, d, d)`` and their dofs ``(N, d)``."""
    N, d = dofs.shape
    rows = np.repeat(dofs, d, axis=1).ravel()
    cols = np.tile(dofs, (1, d)).ravel()
    return sp.csc_matrix((Sr.reshape(N, -1).ravel(), (rows, cols)), shape=(n, n))


def solve_blocks(ops: dict[str, np.ndarray], pb: np.ndarray, dofs: np.ndarray, n: int):
    """Assemble, factorise, solve; return the per-block reduced port values ``(N, d)`` and the timings."""
    t0 = time.perf_counter()
    A = assemble_blocks(ops["Sr"], dofs, n)
    g = np.zeros(n)
    np.add.at(g, dofs.ravel(), np.einsum("bdk,bk->bd", ops["Wr"], pb).ravel())
    t1 = time.perf_counter()
    lu = fv.factor(A)
    t = lu.solve(g)
    return t[dofs], {"assemble": t1 - t0, "factor_solve": time.perf_counter() - t1, "lu": lu}


def block_observables(ops: dict[str, np.ndarray], tb: np.ndarray, pb: np.ndarray, n2: int) -> dict[str, np.ndarray]:
    vals = np.einsum("bod,bd->bo", ops["Otr"], tb) + np.einsum("bok,bk->bo", ops["Op"], pb)
    res = {"Tj": vals[:, :n2]}
    for i, name in enumerate(FLUXES):
        res[name] = vals[:, n2 + i]
    return res


def _replace(ops, idx, exact_ops):
    for k in _PARTS:
        ops[k][idx] = np.stack([e[k] for e in exact_ops])


@dataclass
class AdaptiveResult:
    obs: dict[str, np.ndarray]
    tb: np.ndarray
    exact_blocks: np.ndarray
    checked: np.ndarray
    failed: list[np.ndarray]
    rounds: int
    timings: dict[str, float]
    flagged_std: int
    flagged_ood: int
    estimate: dict = field(default_factory=dict)


def solve_adaptive(learned: LearnedBlocks, theta: np.ndarray, power: np.ndarray, *, tol_std: float = 0.05,
                   tol_T: float = 0.01, n_hot: int = 8, n_audit: int = 16,
                   max_rounds: int = 5, seed: int = 0) -> AdaptiveResult:
    """Learned condensation with the verification loop of the module docstring. ``theta``: ``(nbx, nby, p)`` block
    parameters; ``power``: ``(nbx * n, nby * n)`` W per active cell."""
    st = learned.stack
    nbx, nby = theta.shape[:2]
    N = nbx * nby
    n, n2 = st.n, st.n ** 2
    pb = power.reshape(nbx, n, nby, n).transpose(0, 2, 1, 3).reshape(N, n2)
    th = theta.reshape(N, -1)
    tim = {}
    t0 = time.perf_counter()
    ops, rel, ood = learned.predict(th)
    tim["predict"] = time.perf_counter() - t0
    exact = np.zeros(N, bool)
    first = np.nonzero((rel > tol_std) | ood)[0]
    t1 = time.perf_counter()
    if len(first):
        _replace(ops, first, [learned.exact(th[i]) for i in first])
        exact[first] = True
    tim["exact_condensations"] = time.perf_counter() - t1
    dofs, nf = _face_dofs((nbx, nby), learned.m)
    rng = np.random.default_rng(seed)
    checked = np.zeros(N, bool)
    failed = []
    tim["solve"] = tim["verify"] = 0.0
    audit_effects: list[float] = []
    for rnd in range(1, max_rounds + 1):
        t2 = time.perf_counter()
        tb, info = solve_blocks(ops, pb, dofs, nf * learned.m)
        obs = block_observables(ops, tb, pb, n2)
        tim["solve"] += time.perf_counter() - t2
        t3 = time.perf_counter()
        hot = np.argsort(-obs["Tj"].max(1))
        cand = [i for i in hot if not exact[i] and not checked[i]][:n_hot]
        pool = np.setdiff1d(np.nonzero(~exact & ~checked)[0], cand)
        audit = rng.choice(pool, min(n_audit, len(pool)), replace=False) if len(pool) else np.array([], int)
        todo = np.array(list(cand) + list(audit), int)
        ex_ops = {i: learned.exact(th[i]) for i in todo}
        R = np.zeros((nf * learned.m, len(todo)))
        local = np.zeros(len(todo))
        for c, i in enumerate(todo):
            e = ex_ops[i]
            local[c] = np.abs((e["Otr"] - ops["Otr"][i]) @ tb[i] + (e["Op"] - ops["Op"][i]) @ pb[i])[:n2].max()
            np.add.at(R[:, c], dofs[i], (e["Sr"] - ops["Sr"][i]) @ tb[i] - (e["Wr"] - ops["Wr"][i]) @ pb[i])
        effect = local.copy()
        if len(todo):
            dlt = -info["lu"].solve(R)                                    # (n_dofs, n_todo): A e = -r
            dTj = np.einsum("bod,bdc->bco", ops["Otr"][:, :n2], dlt[dofs])   # (N, n_todo, n2)
            effect = np.maximum(local, np.abs(dTj).max(axis=(0, 2)))
        bad = list(todo[effect > tol_T])
        audit_effects += list(effect[len(cand):])
        checked[todo] = True
        tim["verify"] += time.perf_counter() - t3
        failed.append(np.array(bad, int))
        if bad:
            _replace(ops, np.array(bad), [ex_ops[i] for i in bad])
            exact[bad] = True
            continue
        break
    unchecked = int((~exact & ~checked).sum())
    ae = np.array(audit_effects) if audit_effects else np.zeros(1)
    estimate = {"unchecked_blocks": unchecked, "mean_effect_K": float(ae.mean()), "max_effect_K": float(ae.max()),
                "additive_estimate_K": float(unchecked * ae.mean())}
    tim["total"] = time.perf_counter() - t0
    return AdaptiveResult({k: v.reshape(nbx, nby, *v.shape[1:]) for k, v in obs.items()}, tb, exact, checked,
                          failed, rnd, tim, int(((rel > tol_std) & ~ood).sum()), int(ood.sum()), estimate)


def solve_exact(learned: LearnedBlocks, theta: np.ndarray, power: np.ndarray) -> tuple[dict, float]:
    """Every block condensed exactly (one factorisation per block): the reference of the learned path."""
    st = learned.stack
    nbx, nby = theta.shape[:2]
    N = nbx * nby
    n, n2 = st.n, st.n ** 2
    pb = power.reshape(nbx, n, nby, n).transpose(0, 2, 1, 3).reshape(N, n2)
    t0 = time.perf_counter()
    exs = [learned.exact(t) for t in theta.reshape(N, -1)]
    ops = {k: np.stack([e[k] for e in exs]) for k in _PARTS}
    dofs, nf = _face_dofs((nbx, nby), learned.m)
    tb, _ = solve_blocks(ops, pb, dofs, nf * learned.m)
    obs = block_observables(ops, tb, pb, n2)
    return {k: v.reshape(nbx, nby, *v.shape[1:]) for k, v in obs.items()}, time.perf_counter() - t0


def variant_field(shape: tuple[int, int], seed: int = 0, *, defects: int = 0, defect_k: float = 0.05,
                  bounds: dict | None = None) -> np.ndarray:
    """Smooth random block parameters inside ``bounds`` (a degraded-TIM gradient, scattered bump voids, TSV
    quality) and ``defects`` nearly open microbumps (``ubump_k = defect_k``, outside the default training box)."""
    b = bounds or DEFAULT_BOUNDS
    rng = np.random.default_rng(seed)
    nbx, nby = shape
    X, Y = np.meshgrid(np.linspace(0, 1, nbx), np.linspace(0, 1, nby), indexing="ij")
    out = np.empty((nbx, nby, 3))
    lo = np.array([v[0] for v in b.values()])
    hi = np.array([v[1] for v in b.values()])
    for j in range(3):
        a = rng.normal(size=4)
        f = a[0] * X + a[1] * Y + 0.5 * np.sin(np.pi * (a[2] * X + a[3] * Y))
        f = (f - f.min()) / max(np.ptp(f), 1e-12)
        noise = rng.uniform(0, 1, shape)
        out[..., j] = lo[j] + (hi[j] - lo[j]) * np.clip(0.7 * f + 0.3 * noise, 0, 1)
    for _ in range(defects):
        i, j = rng.integers(nbx), rng.integers(nby)
        out[i, j, 0] = defect_k
    return out
