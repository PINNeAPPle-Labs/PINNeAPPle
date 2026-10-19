"""Parameter identifiability: can these parameters be recovered from these measurements, before anyone estimates them?

:func:`analyze_identifiability` answers it with four independent methods on the same model and measurements:

1. **Fisher information** ``F = J^T Sigma^-1 J`` at the analysis point (``J``: sensitivities of the predictions to
   the scaled parameters). Its numerical rank gives *local structural* identifiability (a zero eigenvalue is a
   direction in parameter space that changes no prediction at all); its inverse is the Cramér-Rao bound on the
   standard deviation of any unbiased estimate (*practical* identifiability) and gives the expected correlations.
   Eigen-directions with zero or small eigenvalues are reported as **ambiguities**: the parameter combinations the
   data cannot tell apart, and the combination they do fix (e.g. ``h/k``).
2. **Local sensitivity** of every prediction to every parameter (``dy/dz``, for a log parameter the change of the
   prediction for a 100 % change of the parameter), and the **value of each observation**: its leverage
   ``(J_i/sigma_i) F^+ (J_i/sigma_i)^T``, the share of the identifiable information it carries.
3. **Profile likelihood** (Raue et al., Bioinformatics 25, 2009): each parameter is fixed on a grid and the others
   re-fitted; the profile ``chi2`` crossing ``chi2_min + 3.84`` on both sides gives a 95 % interval. A flat profile
   is structural non-identifiability, a profile open on one side is practical non-identifiability. Unlike the
   Fisher bound it is not a linearisation.
4. **Bayesian posterior** by adaptive Metropolis (flat priors inside the bounds), for posterior correlations and
   intervals, when the model is cheap enough (``model.cost == "cheap"``).

Then **optimal experimental design**: from candidate measurements (another sensor position, time, observable),
the greedy D-optimal choice: the candidate that most increases ``log det F`` (determinant lemma
``log(1 + g^T F^-1 g)``), repeated ``n_suggest`` times, with the rank and the bounds each one would give.

Before an experiment the measured values may be missing: the analysis then uses the model's own predictions at the
nominal parameters ("expected data", noise-free), the standard design-stage setting. With measured values the model
is fitted first and analysed at the best fit.

Limits: the Fisher rank is a local, numerical test at one point (not a symbolic proof of global structural
identifiability); the profile and the posterior are only as good as the parameter bounds given.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimation import least_squares_fit, metropolis
from .forward import ForwardModel, Observations, Parameter, ParameterSet, jacobian
from .sensitivity import IdentifiabilityAnalyzer

__all__ = ["IdentifiabilityReport", "analyze_identifiability", "d_optimal_design", "profile_likelihood",
           "CHI2_95"]

CHI2_95 = 3.841458820694124          # chi-square quantile, 1 degree of freedom, 95 %
STATUS = ("identifiable", "practically non-identifiable", "structurally non-identifiable")


def _fmt_exp(e: float) -> str:
    r = round(e, 2)
    return "" if abs(r - 1) < 1e-9 else f"^{r:g}"


def _combination(names: list[str], log: list[bool], v: np.ndarray) -> dict[str, Any]:
    """Describe a sloppy direction ``v`` (unit vector in z) involving 2+ parameters: what moves, and for a pair what
    stays fixed (``a^vb * b^-va`` for log parameters, ``vb*a - va*b`` for linear ones)."""
    idx = [i for i in np.argsort(-np.abs(v)) if abs(v[i]) > 0.02]
    weights = {names[i]: round(float(v[i]), 3) for i in idx}
    out: dict[str, Any] = {"parameters": [names[i] for i in idx], "direction": weights}
    if len(idx) == 1:
        out["fixed_combination"] = None
        out["text"] = f"{names[idx[0]]} changes no prediction (within the noise) on its own"
        return out
    moves = " and ".join(f"{names[i]} {'up' if v[i] > 0 else 'down'}" for i in idx)
    if len(idx) == 2:
        a, b = idx
        va, vb = float(v[a]), float(v[b])
        if log[a] and log[b]:
            ea, eb = vb / abs(vb), -va / abs(vb)
            if abs(eb) < 1e-12:
                comb = names[a]
            else:
                comb = f"{names[a]}{_fmt_exp(ea)}·{names[b]}{_fmt_exp(abs(eb))}" if eb > 0 else \
                    f"{names[a]}{_fmt_exp(ea)}/{names[b]}{_fmt_exp(abs(eb))}"
        else:
            comb = f"{vb:+.2f}·{names[a]} {(-va):+.2f}·{names[b]}"
        out["fixed_combination"] = comb
        out["text"] = (f"moving {moves} together leaves the predictions unchanged: the data fix {comb}, "
                       f"not {names[a]} and {names[b]} separately")
    else:
        out["fixed_combination"] = None
        out["text"] = f"moving {moves} together (direction {weights}) leaves the predictions nearly unchanged"
    return out


@dataclass
class IdentifiabilityReport:
    parameters: list[dict[str, Any]]          # one dict per free parameter: status, bounds, sensitivity, ...
    analysis_point: dict[str, float]
    data: str                                  # "measured" or "expected (noise-free predictions at nominal values)"
    n_observations: int
    rank: int
    n_parameters: int
    condition_number: float
    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    d_criterion: float                         # log det F (identifiable subspace if rank-deficient)
    correlation: list[list[float]]
    ambiguities: list[dict[str, Any]]
    observations: dict[str, Any]               # leverage per row, value per observable
    suggestions: list[dict[str, Any]]
    profiles: dict[str, dict[str, Any]]
    posterior: dict[str, Any] | None
    derivatives: str
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    fit: dict[str, Any] | None = None

    def status(self, name: str) -> str:
        for p in self.parameters:
            if p["name"] == name:
                return p["status"]
        raise KeyError(name)

    def estimable(self, name: str) -> bool:
        return self.status(name) == "identifiable"

    @property
    def summary(self) -> str:
        ok = [p["name"] for p in self.parameters if p["status"] == STATUS[0]]
        bad = [p for p in self.parameters if p["status"] != STATUS[0]]
        s = f"{len(ok)} of {len(self.parameters)} parameters identifiable"
        if ok:
            s += f" ({', '.join(ok)})"
        for p in bad:
            s += f"; {p['name']}: {p['status']}"
        if self.ambiguities:
            s += ". " + self.ambiguities[0]["text"][0].upper() + self.ambiguities[0]["text"][1:]
        if self.suggestions and bad:
            first = self.suggestions[0]
            s += f". Most informative extra measurement: {first['label']}"
        return s

    def to_dict(self) -> dict[str, Any]:
        from pinneapple_core.jsonable import to_jsonable

        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["summary"] = self.summary
        return to_jsonable(d)


def _label(obs: Observations, i: int) -> str:
    parts = [f"{k}={obs.inputs[k][i]:.4g}" for k in obs.inputs if np.isfinite(obs.inputs[k][i])]
    o = str(obs.observable[i])
    return (f"{o} at " if o else "") + ", ".join(parts)


def d_optimal_design(F: np.ndarray, Jc_w: np.ndarray, n_pick: int = 3, distinct: bool = True) -> list[dict[str, Any]]:
    """Greedy D-optimal selection. ``F``: current Fisher information (p, p); ``Jc_w``: sensitivities of the
    candidate measurements divided by their sigma (m, p). Returns the picks in order with ``log det`` gains."""
    p = F.shape[0]
    scale = max(float(np.trace(F)) / p, 1e-30)
    ridge = 1e-9 * scale * np.eye(p)
    Fc = F.copy()
    chosen: list[dict[str, Any]] = []
    used = set()
    for _ in range(min(n_pick, len(Jc_w))):
        Finv = np.linalg.inv(Fc + ridge)
        gains = np.log1p(np.einsum("ij,jk,ik->i", Jc_w, Finv, Jc_w))
        if distinct:
            for u in used:
                gains[u] = -np.inf
        k = int(np.argmax(gains))
        if not np.isfinite(gains[k]) or gains[k] <= 1e-12:
            break
        used.add(k)
        Fc = Fc + np.outer(Jc_w[k], Jc_w[k])
        chosen.append({"index": k, "gain_logdet": float(gains[k]), "F_after": Fc.copy()})
    return chosen


def _std_from_F(F: np.ndarray, tol: float) -> np.ndarray:
    """Cramér-Rao standard deviations in z; inf for parameters touching the null space."""
    w, V = np.linalg.eigh(F)
    lam_max = max(w[-1], 1e-300)
    null = w <= tol * lam_max
    std = np.sqrt(np.maximum(np.diag(np.linalg.pinv(F, rcond=tol, hermitian=True)), 0.0))
    if null.any():
        touch = (np.abs(V[:, null]) > 0.01).any(axis=1)
        std = np.where(touch, np.inf, std)
    return std


def _crossing(d: np.ndarray, zs: np.ndarray, pin: np.ndarray, pairs) -> tuple:
    """Where the profile ``d`` first reaches the 95 % threshold walking ``pairs`` outward: ``(z, None)``, or
    ``(None, z_bound)`` when it rises there only because another parameter sits on its bound, or ``(None, None)``."""
    for a, b in pairs:
        if d[a] < CHI2_95 <= d[b]:
            if pin[b]:
                return None, float(zs[b])
            t = (CHI2_95 - d[a]) / (d[b] - d[a])
            return float(zs[a] + t * (zs[b] - zs[a])), None
    return None, None


def profile_likelihood(model: ForwardModel, params: ParameterSet, obs: Observations, z_best: np.ndarray,
                       chi2_min: float, std_z: np.ndarray, *, n_points: int = 15, max_nfev: int = 60,
                       time_budget: float = 60.0, fisher: np.ndarray | None = None) -> dict[str, dict[str, Any]]:
    """Profile of every free parameter around ``z_best`` (see the module docstring). Returns, per parameter, the
    grid (physical units), ``delta_chi2``, the 95 % interval (``None`` on an open side) and the verdict.

    Each re-fit starts from the better of two points: the previous grid point's optimum, and that optimum moved
    along the ridge the Fisher information predicts (``dz_others = -F_oo^+ F_oi dz_i``), so a strongly correlated
    partner follows the profiled parameter instead of stalling where the model saturates."""
    out: dict[str, dict[str, Any]] = {}
    lo_all, hi_all = params.bounds()
    t0 = time.time()
    p = len(params)
    for i, par in enumerate(params.free):
        s = std_z[i]
        half = 4 * s if np.isfinite(s) and s > 0 else 3.0
        half = min(max(half, 0.05), 3.0)
        lo = max(lo_all[i], z_best[i] - half) if np.isfinite(lo_all[i]) else z_best[i] - half
        hi = min(hi_all[i], z_best[i] + half) if np.isfinite(hi_all[i]) else z_best[i] + half
        if not np.isfinite(s) and np.isfinite(lo_all[i]) and np.isfinite(hi_all[i]):
            lo, hi = lo_all[i], hi_all[i]
        left = np.linspace(z_best[i], lo, n_points // 2 + 1)[1:]
        right = np.linspace(z_best[i], hi, n_points // 2 + 1)[1:]
        others = [j for j in range(p) if j != i]
        ridge = np.zeros(p)
        if fisher is not None and others:
            Foo = fisher[np.ix_(others, others)]
            ridge[others] = -np.linalg.pinv(Foo, rcond=1e-10, hermitian=True) @ fisher[others, i]
        pts = {float(z_best[i]): chi2_min}
        pinned = {float(z_best[i]): False}             # another parameter on its bound at this grid point
        for side in (left, right):
            zw = z_best.copy()
            for zi in side:
                if time.time() - t0 > time_budget:
                    break
                step = zi - zw[i]
                zw[i] = zi
                if others:
                    r = least_squares_fit(model, params, obs, z0=zw, free=others, max_nfev=max_nfev)
                    if np.any(ridge):
                        zp = params.clip(zw + ridge * step)
                        zp[i] = zi
                        r2 = least_squares_fit(model, params, obs, z0=zp, free=others, max_nfev=max_nfev)
                        r = r2 if r2.chi2 < r.chi2 else r
                    zw, c = r.z.copy(), r.chi2
                    pinned[float(zi)] = bool(r.at_bound)
                else:
                    from .forward import _predict

                    pred = _predict(model, params.theta(zw), obs)
                    c = float(np.sum(((pred - obs.values) / obs.sigma) ** 2))
                    pinned[float(zi)] = False
                pts[float(zi)] = c
        zs = np.array(sorted(pts))
        c = np.array([pts[z] for z in zs])
        pin = np.array([pinned[z] for z in zs])
        cmin = min(chi2_min, float(c.min()))
        d = c - cmin
        ib = int(np.argmin(d))

        zl, bl = _crossing(d, zs, pin, [(k, k - 1) for k in range(ib, 0, -1)])
        zr, br = _crossing(d, zs, pin, [(k, k + 1) for k in range(ib, len(zs) - 1)])
        bounded = [float(par.from_z(b)) for b in (bl, br) if b is not None]
        free_d = d[~pin] if (~pin).any() else d
        flat = float(free_d.max()) < 0.05 * CHI2_95
        verdict = STATUS[2] if flat else (STATUS[0] if zl is not None and zr is not None else STATUS[1])
        out[par.name] = {"grid": [par.from_z(z) for z in zs], "delta_chi2": d.round(6).tolist(),
                         "ci95": [None if zl is None else par.from_z(zl), None if zr is None else par.from_z(zr)],
                         "verdict": verdict, "threshold": CHI2_95, "closed_by_bounds_at": bounded,
                         "other_at_bound": pin.tolist(),
                         "complete": bool(time.time() - t0 <= time_budget)}
    return out


def analyze_identifiability(model: ForwardModel, obs: Observations,
                            parameters: Sequence[Parameter] | None = None, *,
                            candidates: Observations | None = None, n_suggest: int = 3,
                            profile: bool | str = "auto", n_profile: int = 15, posterior: bool | str = "auto",
                            mcmc_steps: int = 4000, rank_tol: float = 1e-8, practical_threshold: float = 0.5,
                            fit_first: bool = True, seed: int = 0, time_budget: float = 90.0
                            ) -> IdentifiabilityReport:
    """Identifiability of ``parameters`` (default: the model's) from ``obs``; see the module docstring.

    ``practical_threshold``: a parameter is practically non-identifiable when its Cramér-Rao standard deviation in
    ``z`` exceeds this (for a log parameter 0.5 is about ±65 % at one sigma) or when its profile is open.
    ``profile`` / ``posterior``: ``True``, ``False`` or ``"auto"`` (only for cheap models, at most 6 parameters).
    """
    t0 = time.time()
    params = ParameterSet(parameters if parameters is not None else model.default_parameters())
    names = params.names
    p = len(params)
    notes: list[str] = []
    cost = getattr(model, "cost", "cheap")
    cheap = cost == "cheap"
    fit_info = None
    if obs.has_values and fit_first:
        fit = least_squares_fit(model, params, obs)
        z = fit.z
        data = obs
        mode = "measured"
        fit_info = {"theta": fit.theta, "chi2": fit.chi2, "dof": fit.dof, "nfev": fit.nfev, "at_bound": fit.at_bound}
        if fit.at_bound:
            notes.append(f"The best fit sits on a bound for {', '.join(fit.at_bound)}: the bounds, not the data, "
                         f"decide that value.")
    else:
        z = params.z0()
        pred0, _, _ = jacobian(model, params, z, obs, method="fd" if not cheap else "auto")
        data = obs.with_values(pred0)
        mode = "expected (noise-free predictions at the nominal values)"
    pred, J, how = jacobian(model, params, z, data)
    Jw = J / data.sigma[:, None]
    F = Jw.T @ Jw
    analyzer = IdentifiabilityAnalyzer(tol=rank_tol).analyze(F, names)
    w, V = analyzer.eigenvalues, analyzer.eigenvectors
    lam_max = max(float(w[-1]), 1e-300)
    rank = int(np.sum(w > rank_tol * lam_max))
    std_z = _std_from_F(F, rank_tol)
    reg = F + max(rank_tol * lam_max, 1e-300) * np.eye(p)
    C = np.linalg.inv(reg)
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    logs = [bool(par.log) for par in params.free]

    ambiguities = []
    structural = np.zeros(p, bool)
    for j in range(p):
        ev = float(w[j])
        kind = None
        if ev <= rank_tol * lam_max:
            kind = "structural"
        elif 1 / math.sqrt(ev) > practical_threshold:
            kind = "practical"
        if kind is None:
            continue
        v = V[:, j] * np.sign(V[np.argmax(np.abs(V[:, j])), j])
        desc = _combination(names, logs, v)
        if kind == "structural":
            structural |= np.abs(v) > 0.01
        desc.update(kind=kind, eigenvalue=ev, std_along=None if ev <= 0 else 1 / math.sqrt(ev))
        ambiguities.append(desc)
    col = np.sqrt(np.diag(F))
    structural |= col <= math.sqrt(rank_tol) * math.sqrt(lam_max)

    # value of each observation
    Fp = np.linalg.pinv(F, rcond=rank_tol, hermitian=True)
    lev = np.einsum("ij,jk,ik->i", Jw, Fp, Jw)
    share = (Jw**2) / np.maximum(np.diag(F)[None, :], 1e-300)
    by_obs: dict[str, dict[str, Any]] = {}
    for o in dict.fromkeys(str(x) for x in data.observable):
        m = np.array([str(x) == o for x in data.observable])
        by_obs[o or "y"] = {"n": int(m.sum()), "leverage": float(lev[m].sum()),
                            "information_share": {n: float(share[m, k].sum()) for k, n in enumerate(names)}}
    order = np.argsort(-lev)
    rows = [{"index": int(i), "label": _label(data, i), "leverage": float(lev[i]),
             "inputs": {k: float(v[i]) for k, v in data.inputs.items()}, "observable": str(data.observable[i]),
             "sensitivity": {n: float(J[i, k]) for k, n in enumerate(names)}} for i in range(data.n)]
    observations = {"rows": rows, "ranking": [int(i) for i in order], "by_observable": by_obs,
                    "note": "leverage = share of the identifiable information carried by each measurement "
                            "(sums to the rank); a measurement with leverage near 1 cannot be removed without "
                            "losing a parameter"}

    # experimental design
    suggestions: list[dict[str, Any]] = []
    try:
        cand = candidates if candidates is not None else model.candidate_measurements(data)
        _, Jc, _ = jacobian(model, params, z, cand)
        Jcw = Jc / cand.sigma[:, None]
        for pick in d_optimal_design(F, Jcw, n_pick=n_suggest):
            Fa = pick["F_after"]
            wa = np.linalg.eigvalsh(Fa)
            ra = int(np.sum(wa > rank_tol * max(wa[-1], 1e-300)))
            sa = _std_from_F(Fa, rank_tol)
            k = pick["index"]
            suggestions.append({
                "label": _label(cand, k), "observable": str(cand.observable[k]),
                "inputs": {kk: float(v[k]) for kk, v in cand.inputs.items()}, "sigma": float(cand.sigma[k]),
                "gain_logdet": pick["gain_logdet"], "rank_after": ra,
                "std_after": {n: (None if not np.isfinite(sa[i]) else float(sa[i])) for i, n in enumerate(names)}})
        if suggestions:
            notes.append(f"Extra measurements chosen greedily (D-optimal) among {cand.n} candidates; each line "
                         f"assumes the ones above it are taken too.")
    except (ValueError, NotImplementedError) as exc:
        notes.append(f"No experimental design: {exc}")

    # profile likelihood
    run_profile = profile is True or (profile == "auto" and cost in ("cheap", "moderate") and p <= 6)
    profiles: dict[str, dict[str, Any]] = {}
    if run_profile:
        chi2_min = float(np.sum(((pred - data.values) / data.sigma) ** 2))
        profiles = profile_likelihood(model, params, data, z, chi2_min, std_z, n_points=n_profile,
                                      time_budget=max(5.0, time_budget - (time.time() - t0)) * 0.6, fisher=F)
    elif profile == "auto":
        notes.append("Profile likelihood skipped (expensive model or more than 6 parameters).")

    # posterior
    post = None
    run_post = posterior is True or (posterior == "auto" and cheap and p <= 6)
    if run_post:
        lo, hi = params.bounds()
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ps = metropolis(model, params, data, z0=z, cov0=np.linalg.inv(reg + 1e-6 * np.eye(p)),
                            n_steps=mcmc_steps, seed=seed,
                            max_seconds=max(3.0, (time_budget - (time.time() - t0)) * 0.8))
            post = ps.summary()
            post["note"] = ("adaptive Metropolis, flat priors inside the bounds; a structurally non-identifiable "
                            "parameter spreads over its bounds")
        else:
            notes.append("Posterior skipped: it needs finite bounds on every parameter.")
    elif posterior == "auto":
        notes.append("Posterior (MCMC) skipped (expensive model or more than 6 parameters).")

    # per-parameter verdict
    out_params = []
    for i, par in enumerate(params.free):
        if structural[i]:
            status = STATUS[2]
            reason = "the Fisher information has a zero eigenvalue in a direction that moves this parameter"
            if col[i] <= math.sqrt(rank_tol) * math.sqrt(lam_max):
                reason = "no measurement is sensitive to it"
        elif std_z[i] > practical_threshold:
            status, reason = STATUS[1], (f"Cramér-Rao standard deviation {std_z[i]:.2f} in "
                                         f"{'ln units' if par.log else 'units of the nominal value'} exceeds "
                                         f"{practical_threshold}")
        else:
            status, reason = STATUS[0], "full-rank Fisher information and a small Cramér-Rao bound"
        prof = profiles.get(par.name)
        if prof and status != STATUS[2] and prof["verdict"] != STATUS[0]:
            status, reason = prof["verdict"], "the 95 % profile likelihood interval is open within the bounds"
        if prof and status == STATUS[2] and prof["verdict"] == STATUS[0]:
            notes.append(f"{par.name}: the Fisher information is rank-deficient at the analysis point but the "
                         f"profile likelihood closes; the dependence is nonlinear, check the profile plot.")
        if prof and prof.get("closed_by_bounds_at"):
            notes.append(f"{par.name}: the profile rises only where another parameter reaches its bound (at "
                         f"{', '.join(f'{v:.4g}' for v in prof['closed_by_bounds_at'])} {par.unit}): the bounds, "
                         f"not the data, limit it.")
        theta_i = params.theta(z)[par.name]
        if np.isfinite(std_z[i]):
            rel = (math.exp(std_z[i]) - 1) if par.log else std_z[i] * par.scale / max(abs(theta_i), 1e-300)
            crlb = std_z[i] * (theta_i if par.log else par.scale)
        else:
            rel, crlb = None, None
        out_params.append({
            "name": par.name, "unit": par.unit, "value": theta_i, "nominal": par.value, "lower": par.lower,
            "upper": par.upper, "log": par.log, "status": status, "reason": reason,
            "sensitivity": float(col[i]), "sensitivity_share": float(col[i] / max(col.sum(), 1e-300)),
            "crlb_std": crlb, "crlb_relative": rel,
            "profile_ci95": prof["ci95"] if prof else None})
    pos = w[w > rank_tol * lam_max]
    report = IdentifiabilityReport(
        parameters=out_params, analysis_point=params.theta(z), data=mode, n_observations=data.n, rank=rank,
        n_parameters=p, condition_number=math.inf if rank < p else float(lam_max / float(w[0])),
        eigenvalues=w.tolist(), eigenvectors=V.T.tolist(), d_criterion=float(np.sum(np.log(pos))) if len(pos) else
        -math.inf, correlation=np.clip(corr, -1, 1).round(4).tolist(), ambiguities=ambiguities,
        observations=observations, suggestions=suggestions, profiles=profiles, posterior=post, derivatives=how,
        notes=notes, seconds=time.time() - t0, fit=fit_info)
    return report

