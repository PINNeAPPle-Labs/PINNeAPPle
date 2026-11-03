"""The standard reduced-order-modeling workflow (#1208): snapshots -> POD basis -> POD-Galerkin, DMD and operator
inference on one protocol -> checks against the full model on parameters and times never used for training.

Protocol
--------
1. **Snapshots** from a library full-order model (:mod:`.fom`, ``n_params`` parameter values over the stated range)
   or from a file (``.npz`` with ``snapshots`` ``(P, T, n)``, ``times`` ``(T,)`` and ``params`` ``(P,)``). The
   parameters at odd positions (1, 3, ...) are held out (interpolation); the training window is the first
   ``train_fraction`` of the times, the rest is held out (extrapolation in time).
2. **Basis**: when the invariant is the mass (``sum(u) dx``), the constant mode comes first (so the invariant is
   inside the basis), then the POD modes of the training snapshots with it removed. The rank is the smallest that
   captures ``energy`` of the singular-value energy; the projection error of the held-out FOM trajectories on that
   basis is reported: it is the floor no ROM in that basis can beat.
3. **ROMs** in that basis, each run with the same time step as the FOM:

   * ``pod_galerkin`` (intrusive; needs the FOM operator): ``A_r(mu) = V^T (L0 + mu L1) V`` and the quadratic tensor
     ``V^T Q(V_j * V_k)``, integrated with the FOM's own IMEX scheme.
   * ``dmd`` (non-intrusive): exact DMD of the reduced coordinates per training parameter (a discrete operator at
     the snapshot spacing), interpolated linearly in the parameter (operator interpolation in a common basis).
   * ``opinf`` (non-intrusive operator inference, Peherstorfer & Willcox 2016): ``da/dt = (A0 + mu A1) a + H (a x a)``
     fitted by regularized least squares to the reduced coordinates and their 4th-order central time derivatives,
     the same parametric structure the FOM has, integrated with the same IMEX scheme.

4. **Checks** per ROM: relative space-time L2 error against the FOM on the held-out parameters in the training
   window (``interpolation``) and after it (``extrapolation``); stability of the reduced operator from its
   eigenvalues (growth factor over the horizon); drift of the invariant; and the speed-up of the online solve
   (operator assembly, time integration and lifting to the full field) against the FOM solve at the same
   parameter, both timed here. The recommendation is the passing ROM with the lowest extrapolation error.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fom import PeriodicTransportFOM, make_fom


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _time_best(fn: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    best, out = np.inf, None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, float(best)


def _ddt(A: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """4th-order central differences of the rows of ``A`` (spacing ``h``); returns (states, derivatives)."""
    d = (A[:-4] - 8 * A[1:-3] + 8 * A[3:-1] - A[4:]) / (12 * h)
    return A[2:-2], d


def _kron_c(a: np.ndarray) -> np.ndarray:
    """Compressed (upper-triangle) quadratic features of the rows of ``a``."""
    r = a.shape[-1]
    iu = np.triu_indices(r)
    return (a[..., :, None] * a[..., None, :])[..., iu[0], iu[1]]


def _compress_quad(H: np.ndarray) -> np.ndarray:
    """Full quadratic operator (r, r, r) acting on ``a_j a_k`` -> compressed (r, r(r+1)/2) on the upper triangle."""
    r = H.shape[1]
    iu = np.triu_indices(r)
    return np.where(iu[0] == iu[1], H[:, iu[0], iu[1]], H[:, iu[0], iu[1]] + H[:, iu[1], iu[0]])


def _reduced_rollout(a0: np.ndarray, L: np.ndarray, Hc: np.ndarray | None, dt: float, steps: int,
                     save_every: int) -> np.ndarray:
    """The FOM's IMEX CN/AB2 scheme on the reduced system, with the implicit matrix inverted once (it is r x r).
    A linear model advances one snapshot interval per product (the CN step matrix to the power ``save_every``),
    which is the same scheme, not an approximation."""
    r = a0.size
    eye = np.eye(r)
    M = np.linalg.inv(eye - 0.5 * dt * L)
    G = M @ (eye + 0.5 * dt * L)
    out = [a0]
    a = a0
    if Hc is None:
        Gs = np.linalg.matrix_power(G, save_every)
        for _ in range(steps // save_every):
            a = Gs @ a
            out.append(a)
        return np.array(out)
    i0, i1 = np.triu_indices(r)
    MH = dt * (M @ Hc)
    qp = None
    for k in range(1, steps + 1):
        qn = MH @ (a[i0] * a[i1])
        a = G @ a + (qn if qp is None else 1.5 * qn - 0.5 * qp)
        qp = qn
        if k % save_every == 0:
            out.append(a)
    return np.array(out)


@dataclass
class _ROM:
    """A reduced model: ``lin(mu)`` is the continuous linear operator (the discrete one for ``discrete=True``) and
    ``Hc`` the compressed quadratic operator."""

    name: str
    lin: Callable[[float], np.ndarray]
    Hc: np.ndarray | None
    discrete: bool
    note: str = ""

    def rollout(self, a0: np.ndarray, mu: float, dt: float, steps: int, save_every: int) -> np.ndarray:
        if self.discrete:
            Ad = self.lin(mu)
            out = [a0]
            for _ in range(steps // save_every):
                out.append(Ad @ out[-1])
            return np.array(out)
        return _reduced_rollout(a0, self.lin(mu), self.Hc, dt, steps, save_every)

    def jacobian(self, mu: float, a_eq: np.ndarray) -> np.ndarray:
        J = np.array(self.lin(mu), float)
        if self.Hc is not None:
            r = a_eq.size
            for k, (i, j) in enumerate(zip(*np.triu_indices(r), strict=True)):
                J[:, i] += self.Hc[:, k] * a_eq[j]
                J[:, j] += self.Hc[:, k] * a_eq[i]
        return J

    def growth(self, mu: float, a_eq: np.ndarray, horizon: float, n_saved: int, skip: int) -> tuple[float, float]:
        """Eigenvalues of the model linearized about the equilibrium ``a_eq`` (the uniform state of the same mass
        when the invariant is the mass), on the modes after the first ``skip`` (the invariant's own mode):
        (max Re(lambda) or max |lambda| for a discrete model, growth factor over the horizon)."""
        J = self.jacobian(mu, a_eq)[skip:, skip:]
        lam = np.linalg.eigvals(J)
        if self.discrete:
            m = float(np.max(np.abs(lam)))
            return m, float(m ** n_saved)
        m = float(np.max(lam.real))
        return m, float(np.exp(m * horizon))


def _basis(train: np.ndarray, energy: float, max_rank: int, constant_mode: bool) -> dict[str, Any]:
    n = train.shape[-1]
    X = train.reshape(-1, n).T                      # (n, m)
    e = np.ones((n, 1)) / np.sqrt(n)
    if constant_mode:
        X = X - e @ (e.T @ X)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    r = int(min(np.searchsorted(cum, energy) + 1, max_rank, U.shape[1]))
    V = np.hstack([e, U[:, :r]]) if constant_mode else U[:, :r]
    return {"V": V, "rank": V.shape[1], "pod_rank": r, "energy_captured": float(cum[r - 1]),
            "singular_values": s[: min(len(s), 40)].tolist()}


def _galerkin(fom: PeriodicTransportFOM, V: np.ndarray) -> _ROM:
    A0, A1 = V.T @ (fom.L0() @ V), V.T @ (fom.L1() @ V)
    H = None
    if fom.q:
        n, r = V.shape
        M = -0.5 * fom.q * (fom.D1 @ (V[:, :, None] * V[:, None, :]).reshape(n, r * r))
        H = _compress_quad((V.T @ M).reshape(r, r, r))
    return _ROM("pod_galerkin", lambda mu: A0 + mu * A1, H, False, "intrusive: projected FOM operators")


def _dmd(coords: Sequence[np.ndarray], mus: np.ndarray) -> _ROM:
    ops = []
    for A in coords:
        X0, X1 = A[:-1], A[1:]
        ops.append(np.linalg.lstsq(X0, X1, rcond=None)[0].T)
    ops = np.array(ops)
    order = np.argsort(mus)
    mus, ops = mus[order], ops[order]

    def lin(mu):
        j = int(np.clip(np.searchsorted(mus, mu) - 1, 0, len(mus) - 2)) if len(mus) > 1 else 0
        if len(mus) == 1:
            return ops[0]
        w = (mu - mus[j]) / (mus[j + 1] - mus[j])
        return (1 - w) * ops[j] + w * ops[j + 1]
    return _ROM("dmd", lin, None, True, "exact DMD per training parameter, operators interpolated in mu")


def _opinf(coords: Sequence[np.ndarray], mus: np.ndarray, h: float, quadratic: bool, reg: float) -> _ROM:
    F, Y = [], []
    for A, mu in zip(coords, mus, strict=True):
        a, da = _ddt(A, h)
        cols = [a, mu * a] + ([_kron_c(a)] if quadratic else [])
        F.append(np.hstack(cols))
        Y.append(da)
    F, Y = np.vstack(F), np.vstack(Y)
    r = coords[0].shape[1]
    scale = np.linalg.norm(F, axis=0) ** 2 / len(F)
    lam = reg * np.maximum(scale, 1e-300) * len(F)
    if quadratic:
        lam[2 * r:] *= 1e2                          # heavier on the quadratic block (standard OpInf practice)
    W = np.linalg.solve(F.T @ F + np.diag(lam), F.T @ Y).T
    A0, A1 = W[:, :r], W[:, r:2 * r]
    H = W[:, 2 * r:] if quadratic else None
    return _ROM("opinf", lambda mu: A0 + mu * A1, H, False,
                "non-intrusive: (A0 + mu A1) a + H(a x a) by regularized least squares")


def _load_snapshots(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    for k in ("snapshots", "times", "params"):
        if k not in d:
            raise ValueError(f"{path}: the .npz needs 'snapshots' (P, T, n), 'times' (T,) and 'params' (P,); "
                             f"'{k}' is missing")
    S = np.asarray(d["snapshots"], float)
    t, p = np.asarray(d["times"], float), np.asarray(d["params"], float)
    if S.ndim != 3 or S.shape[0] != p.size or S.shape[1] != t.size:
        raise ValueError(f"{path}: snapshots {S.shape} do not match params {p.shape} and times {t.shape}")
    return {"snapshots": S, "times": t, "params": p}


def reduced_order_workflow(*, fom: str = "burgers", mu_range: Sequence[float] = (0.05, 0.15), n_params: int = 9,
                           n: int = 512, dt: float = 1e-3, t_end: float = 2.0, save_every: int = 10,
                           snapshots_file: str = "", train_fraction: float = 0.75, energy: float = 0.99999,
                           max_rank: int = 40, methods: Sequence[str] = ("pod_galerkin", "dmd", "opinf"),
                           invariant: str = "mass", tolerance: float = 0.02, extrapolation_tolerance: float = 0.05,
                           invariant_tolerance: float = 1e-6, opinf_regularization: float = 1e-10,
                           repeats: int = 3) -> dict[str, Any]:
    """Run the protocol; see the module docstring. Returns the basis, the per-ROM results, the checks and the
    recommendation. ``invariant``: ``"mass"`` (``sum(u) dx``) or ``""`` (none stated)."""
    if not 0.1 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.1, 1)")
    model: PeriodicTransportFOM | None = None
    fom_seconds: dict[int, float] = {}
    if snapshots_file:
        data = _load_snapshots(snapshots_file)
        S, times, mus = data["snapshots"], data["times"], data["params"]
        dx = 2 * np.pi / S.shape[-1]
        source = f"{S.shape[0]} trajectories of {S.shape[1]} snapshots x {S.shape[2]} values from {snapshots_file}"
    else:
        model = make_fom(fom, n=int(n), dt=float(dt), t_end=float(t_end), save_every=int(save_every))
        mus = np.linspace(float(mu_range[0]), float(mu_range[1]), int(n_params))
        trajs = []
        for i, mu in enumerate(mus):
            U, sec = _time_best(lambda mu=mu: model.solve(float(mu)), 1)
            trajs.append(U)
            fom_seconds[i] = sec
        S, times, dx = np.array(trajs), model.times, model.dx
        source = (f"FOM {fom} (n = {model.n}, dt = {model.dt}, {model.steps} steps): {len(mus)} parameter values "
                  f"mu in [{mus[0]:g}, {mus[-1]:g}]")
    P, T, N = S.shape
    if P < 3:
        raise ValueError("need at least 3 parameter values: the odd positions are held out")
    test_idx = list(range(1, P - 1, 2))
    train_idx = [i for i in range(P) if i not in test_idx]
    n_train_t = max(5, int(round(train_fraction * (T - 1))) + 1)
    if n_train_t >= T:
        raise ValueError("no time left for the extrapolation window")
    t_train = float(times[n_train_t - 1])
    h = float(times[1] - times[0])
    if not np.allclose(np.diff(times), h, rtol=1e-6, atol=1e-12):
        raise ValueError("the snapshot times must be equally spaced")
    if model is not None:
        steps_per_save, dt_rom = model.save_every, model.dt
    else:
        steps_per_save, dt_rom = 1, h
    steps_total = (T - 1) * steps_per_save
    conserve = invariant == "mass"
    train = S[train_idx, :n_train_t]
    basis = _basis(train, float(energy), int(max_rank), conserve)
    V = basis["V"]

    def mass(U):
        return U.sum(axis=-1) * dx

    # projection floor on the held-out trajectories
    floor_i, floor_e = [], []
    for i in test_idx:
        Pu = (S[i] @ V) @ V.T
        floor_i.append(_rel(Pu[:n_train_t], S[i, :n_train_t]))
        floor_e.append(_rel(Pu[n_train_t:], S[i, n_train_t:]))
    basis.update(projection_error_interpolation=float(max(floor_i)),
                 projection_error_extrapolation=float(max(floor_e)))
    basis.pop("V")

    coords = [S[i, :n_train_t] @ V for i in train_idx]
    quadratic = bool(model.q) if model is not None else True
    builders: dict[str, Callable[[], _ROM]] = {
        "dmd": lambda: _dmd(coords, mus[train_idx]),
        "opinf": lambda: _opinf(coords, mus[train_idx], h, quadratic, float(opinf_regularization)),
    }
    skipped: dict[str, str] = {}
    if model is not None:
        builders["pod_galerkin"] = lambda: _galerkin(model, V)
    elif "pod_galerkin" in methods:
        skipped["pod_galerkin"] = "not run: intrusive, needs the FOM operator (snapshots came from a file)"

    results: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name in methods:
        if name in skipped:
            results[name] = {"skipped": skipped[name]}
            checks.append({"code": f"{name}.run", "status": "skip", "detail": skipped[name]})
            continue
        if name not in builders:
            raise ValueError(f"unknown ROM method {name!r}: use pod_galerkin, dmd, opinf")
        rom, build_s = _time_best(builders[name], 1)
        err_i, err_e, drift, growth, eig, speed, rom_s = [], [], [], [], [], [], []
        finite = True
        for i in test_idx:
            mu = float(mus[i])
            a0 = S[i, 0] @ V

            def online(rom=rom, a0=a0, mu=mu):
                return rom.rollout(a0, mu, dt_rom, steps_total, steps_per_save) @ V.T
            Uh, sec = _time_best(online, int(repeats))
            if Uh.shape[0] != T or not np.all(np.isfinite(Uh)):
                finite = False
                err_i.append(np.inf)
                err_e.append(np.inf)
                continue
            err_i.append(_rel(Uh[:n_train_t], S[i, :n_train_t]))
            err_e.append(_rel(Uh[n_train_t:], S[i, n_train_t:]))
            m = mass(Uh)
            drift.append(float(np.max(np.abs(m - m[0])) / max(abs(m[0]), 1e-300)))
            a_eq = np.zeros_like(a0)
            if conserve:
                a_eq[0] = a0[0]                     # the uniform state of the same mass: the flow's equilibrium
            e, g = rom.growth(mu, a_eq, float(times[-1]), T - 1, 1 if conserve else 0)
            eig.append(e)
            growth.append(g)
            rom_s.append(sec)
            if model is not None:
                fom_sec = _time_best(lambda mu=mu: model.solve(mu), int(repeats))[1]
                fom_seconds[i] = min(fom_seconds.get(i, np.inf), fom_sec)
                speed.append(fom_seconds[i] / sec)
        res = {"note": rom.note, "build_seconds": build_s,
               "interpolation_error": float(max(err_i)), "extrapolation_error": float(max(err_e)),
               "per_parameter": [{"mu": float(mus[i]), "interpolation_error": ei, "extrapolation_error": ee}
                                 for i, ei, ee in zip(test_idx, err_i, err_e, strict=True)],
               "finite": finite}
        if finite:
            res.update(max_eigenvalue=float(max(eig)), growth_factor=float(max(growth)),
                       invariant_drift=float(max(drift)) if conserve else None,
                       online_seconds=float(np.median(rom_s)),
                       speedup=float(min(speed)) if speed else None)
        results[name] = res
        ok_i = finite and res["interpolation_error"] <= tolerance
        ok_e = finite and res["extrapolation_error"] <= extrapolation_tolerance
        checks.append({"code": f"{name}.interpolation", "status": "pass" if ok_i else "fail",
                       "detail": f"max rel. L2 error {res['interpolation_error']:.3g} on mu = "
                                 f"{[round(float(mus[i]), 6) for i in test_idx]}, t <= {t_train:g} "
                                 f"(<= {tolerance:g}; projection floor {basis['projection_error_interpolation']:.2g})"})
        checks.append({"code": f"{name}.extrapolation", "status": "pass" if ok_e else "fail",
                       "detail": f"max rel. L2 error {res['extrapolation_error']:.3g} for {t_train:g} < t <= "
                                 f"{float(times[-1]):g} (<= {extrapolation_tolerance:g}; projection floor "
                                 f"{basis['projection_error_extrapolation']:.2g})"})
        if finite:
            kind = ("max |lambda| (discrete)" if rom.discrete else "max Re(lambda)") + \
                " linearized about the uniform state" + (", fluctuation modes" if conserve else "")
            stable = res["growth_factor"] <= 1.01
            checks.append({"code": f"{name}.stability", "status": "pass" if stable else "fail",
                           "detail": f"{kind} = {res['max_eigenvalue']:.4g}; growth of the linearized model over the "
                                     f"horizon {res['growth_factor']:.4g} (<= 1.01)"})
            if conserve:
                okc = res["invariant_drift"] <= invariant_tolerance
                checks.append({"code": f"{name}.invariant", "status": "pass" if okc else "fail",
                               "detail": f"mass drift {res['invariant_drift']:.2g} (<= {invariant_tolerance:g})"})
            if res["speedup"] is not None:
                checks.append({"code": f"{name}.speedup", "status": "pass" if res["speedup"] > 1 else "fail",
                               "detail": f"online {res['online_seconds'] * 1e3:.3g} ms against the FOM: "
                                         f"x{res['speedup']:.3g} (worst held-out parameter, best of {repeats})"})
            else:
                checks.append({"code": f"{name}.speedup", "status": "skip",
                               "detail": "not measured: no FOM to time (snapshots came from a file)"})
        else:
            checks.append({"code": f"{name}.stability", "status": "fail", "detail": "the rollout diverged"})
        res["passed"] = all(c["status"] != "fail" for c in checks if c["code"].startswith(name + "."))

    passing = [k for k, v in results.items() if v.get("passed")]
    best = min(passing, key=lambda k: results[k]["extrapolation_error"]) if passing else None
    out = {"source": source, "fom": fom if model is not None else None, "params": mus.tolist(),
           "train_params": [float(mus[i]) for i in train_idx], "test_params": [float(mus[i]) for i in test_idx],
           "t_train": t_train, "t_end": float(times[-1]), "n_snapshots_train": int(train.shape[0] * train.shape[1]),
           "basis": basis, "methods": results, "checks": checks, "recommended": best,
           "fom_seconds": {float(mus[i]): s for i, s in fom_seconds.items() if i in test_idx}}
    out["summary"] = (f"rank {basis['rank']} ({basis['energy_captured']:.6g} energy); " +
                      "; ".join(f"{k}: {v['interpolation_error']:.2g} / {v['extrapolation_error']:.2g}"
                                + (f", x{v['speedup']:.3g}" if v.get("speedup") else "")
                                for k, v in results.items() if "skipped" not in v)
                      + (f"; recommended {best}" if best else "; no ROM passes every check"))
    return out


def write_snapshots(path: str, snapshots: np.ndarray, times: Sequence[float], params: Sequence[float]) -> str:
    """Write a snapshot file the workflow reads (``.npz``)."""
    np.savez_compressed(path, snapshots=np.asarray(snapshots, float), times=np.asarray(times, float),
                        params=np.asarray(params, float))
    return path


def as_table(result: Mapping[str, Any]) -> list[list[Any]]:
    """One row per ROM: method, interpolation and extrapolation error, stability, drift, speed-up, passed."""
    rows = []
    for k, v in result["methods"].items():
        if "skipped" in v:
            rows.append([k, None, None, None, None, None, v["skipped"]])
            continue
        rows.append([k, v["interpolation_error"], v["extrapolation_error"], v.get("growth_factor"),
                     v.get("invariant_drift"), v.get("speedup"), "pass" if v["passed"] else "fail"])
    return rows
