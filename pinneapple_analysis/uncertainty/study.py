"""One uncertainty study of one model output: is the prediction reliable here, and which cases need the solver?

:func:`uncertainty_study` composes the pieces of :mod:`pinneapple_analysis.uncertainty`:

1. split-conformal intervals calibrated on ``(X_cal, y_cal)`` with their coverage **measured** on ``(X_test, y_test)``
   (never used to calibrate), plus the coverage of the raw ensemble interval when the model has a spread;
2. an applicability gate fitted on reference inputs (the training inputs when known) and the declared validity range;
3. screening of ``cases`` (scenarios, a batch of designs): prediction, interval, gate, route, and optionally the
   high-fidelity answer for the routed cases (or all), which shows whether the rule sent the right cases;
4. propagation of uncertain inputs around a nominal operating point (MC / LHS / Sobol, and polynomial chaos when
   cheap), with the probability of meeting the limits with and without the model's own error, and the share of the
   propagated samples that leave the region where the model was checked.

Every number in the returned dict comes from these runs; nothing is assumed.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from pinneapple_analysis.uncertainty.applicability import (
    ApplicabilityDomain,
    route_cases,
    routing_summary,
)
from pinneapple_analysis.uncertainty.conformal_study import conformal_interval, conformal_study
from pinneapple_analysis.uncertainty.distributions import as_space
from pinneapple_analysis.uncertainty.tolerance import propagate_space, wilson_interval

SCALES = ("absolute", "relative", "ensemble")


def _scale(kind: str, model, X, pred, out_idx: int):
    if kind == "absolute":
        return None
    if kind == "relative":
        s = np.abs(pred)
        if np.any(s <= 0):
            raise ValueError("relative scale needs predictions different from zero; use 'absolute'")
        return s
    if kind == "ensemble":
        S = model.std(X)
        if S is None:
            raise ValueError(f"{model.name} has no ensemble spread; use 'absolute' or 'relative'")
        s = S[:, out_idx]
        return np.maximum(s, 1e-12 * max(float(np.abs(pred).max()), 1.0))
    raise ValueError(f"scale must be one of {SCALES}")


def _hist(y, bins: int = 30) -> dict:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    counts, edges = np.histogram(y, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def uncertainty_study(model, output: str, X_cal, y_cal, X_test, y_test, *, alpha: float = 0.1,
                      scale: str = "absolute", X_ref=None, bounds: Mapping[str, tuple] | None = None,
                      cases=None, case_names: Sequence[str] | None = None,
                      tolerance: float | None = None, tolerance_pct: float | None = None,
                      limits: tuple[float | None, float | None] | None = None,
                      high_fidelity: Callable[[np.ndarray], np.ndarray] | None = None, hf_scope: str = "routed",
                      max_hf: int = 40, nominal: Mapping[str, float] | None = None, uncertain=None,
                      n_propagation: int = 4096, propagation_method: str = "lhs", pce_degree: int = 3,
                      level: float = 0.95, k: int = 5, gate_level: float = 0.99, seed: int = 0) -> dict:
    """Run the study (see the module docstring). ``model`` is a
    :class:`pinneapple_analysis.study_models.StudyModel`; arrays are ``(n, d)`` inputs in ``model.input_names``
    order and ``(n,)`` true outputs. ``tolerance`` (output units) or ``tolerance_pct`` (percent of the prediction)
    is the interval half-width a decision can accept; ``limits = (lo, hi)`` the acceptance limits of the output.
    ``uncertain`` lists the uncertain inputs (distributions) and ``nominal`` the values of the others."""
    j = model.output_index(output)
    names = model.input_names
    Xc, Xt = np.asarray(X_cal, dtype=float), np.asarray(X_test, dtype=float)
    yc, yt = np.asarray(y_cal, dtype=float).ravel(), np.asarray(y_test, dtype=float).ravel()
    ok_c, ok_t = np.isfinite(yc), np.isfinite(yt)
    Xc, yc, Xt, yt = Xc[ok_c], yc[ok_c], Xt[ok_t], yt[ok_t]
    pc, pt = model.predict(Xc)[:, j], model.predict(Xt)[:, j]
    sc, st = _scale(scale, model, Xc, pc, j), _scale(scale, model, Xt, pt, j)
    study = conformal_study(yc, pc, yt, pt, alpha=alpha, scale_cal=sc, scale_test=st, level=level, group_by=pt,
                            group_name=f"predicted {output}")
    err = pt - yt
    accuracy = {"n_test": int(yt.size), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "max_abs": float(np.max(np.abs(err)))}
    # percent errors only when no true value is near zero (an output that crosses zero, e.g. a lift coefficient,
    # makes them meaningless)
    if np.min(np.abs(yt)) >= 0.05 * np.median(np.abs(yt)) > 0:
        rel = np.abs(err) / np.abs(yt)
        accuracy.update(mean_abs_pct=float(100 * rel.mean()), p95_abs_pct=float(100 * np.quantile(rel, 0.95)))
    ensemble = None
    S = model.std(Xt)
    if S is not None:
        z = norm.ppf(1 - alpha / 2)
        cov = np.abs(err) <= z * S[:, j]
        kk = int(cov.sum())
        ensemble = {"interval": f"prediction +- {z:.3g} x ensemble std (Gaussian assumption)",
                    "nominal": 1 - alpha, "coverage": kk / cov.size,
                    "coverage_ci": list(wilson_interval(kk, cov.size, level)),
                    "mean_width": float(np.mean(2 * z * S[:, j]))}

    ref = np.asarray(X_ref, dtype=float) if X_ref is not None else np.vstack([Xc, Xt])
    domain = ApplicabilityDomain.fit(ref, names, bounds=bounds, k=k, level=gate_level)
    test_gate = domain.assess(Xt)
    covered = study.covered
    by_status = []
    for s in ("inside", "sparse", "outside"):
        m = test_gate["status"] == s
        if m.any():
            kk = int(covered[m].sum())
            by_status.append({"status": s, "n": int(m.sum()), "coverage": kk / int(m.sum()),
                              "coverage_ci": list(wilson_interval(kk, int(m.sum()), level))})
    out = {"model": model.info(), "output": output, "unit": model.units.get(output, ""), "alpha": alpha,
           "scale": scale, "conformal": study.to_dict(), "accuracy": accuracy, "ensemble": ensemble,
           "domain": domain.to_dict(), "test_coverage_by_gate": by_status}

    tol_abs = tolerance
    if cases is not None:
        Xq = np.atleast_2d(np.asarray(cases, dtype=float))
        pq = model.predict(Xq)[:, j]
        sq = _scale(scale, model, Xq, pq, j)
        lo, hi = conformal_interval(pq, study.q, sq)
        gate = domain.assess(Xq)
        if tolerance_pct is not None:
            tol_case = np.abs(pq) * tolerance_pct / 100.0
        else:
            tol_case = np.full(pq.size, np.inf if tol_abs is None else tol_abs)
        routes = []
        for i in range(pq.size):
            r = route_cases(gate["status"][i:i + 1], pq[i:i + 1], lo[i:i + 1], hi[i:i + 1],
                            tolerance=None if not np.isfinite(tol_case[i]) else float(tol_case[i]), limits=limits)[0]
            routes.append(r)
        rows = []
        names_c = list(case_names) if case_names is not None else [f"case {i + 1}" for i in range(pq.size)]
        for i in range(pq.size):
            rows.append({"name": names_c[i], "inputs": dict(zip(names, Xq[i].tolist(), strict=True)), "prediction": float(pq[i]),
                         "lower": float(lo[i]), "upper": float(hi[i]), "status": str(gate["status"][i]),
                         "out_of_range": gate["out_of_range"][i], "knn_distance": float(gate["knn_distance"][i]),
                         "density_ratio": float(gate["density_ratio"][i]),
                         "neighbours_within": int(gate["neighbours_within"][i]), **routes[i]})
        hf = None
        if high_fidelity is not None:
            pick = [i for i, r in enumerate(rows) if hf_scope == "all" or r["route"] == "high_fidelity"][:max_hf]
            if pick:
                yh = np.asarray(high_fidelity(Xq[pick]), dtype=float).ravel()
                for i, v in zip(pick, yh, strict=True):
                    rows[i]["high_fidelity"] = float(v) if np.isfinite(v) else None
                    if np.isfinite(v):
                        rows[i]["hf_error"] = float(pq[i] - v)
                        rows[i]["hf_in_interval"] = bool(lo[i] <= v <= hi[i])
            groups = {}
            for route in ("surrogate", "high_fidelity"):
                e = [abs(r["hf_error"]) for r in rows if r["route"] == route and "hf_error" in r]
                ins = [r["hf_in_interval"] for r in rows if r["route"] == route and "hf_error" in r]
                if e:
                    groups[route] = {"n": len(e), "mean_abs_error": float(np.mean(e)), "max_abs_error": float(np.max(e)),
                                     "in_interval": float(np.mean(ins))}
            hf = {"scope": hf_scope, "n_run": len(pick), "n_failed": int(sum(rows[i].get("high_fidelity") is None
                                                                             for i in pick)),
                  "capped": len([r for r in rows if hf_scope == "all" or r["route"] == "high_fidelity"]) > max_hf,
                  "by_route": groups}
        out["cases"] = {"rows": rows, "summary": routing_summary(routes), "tolerance": tol_abs,
                        "tolerance_pct": tolerance_pct, "limits": list(limits) if limits else None,
                        "high_fidelity": hf}

    if uncertain:
        sp = as_space(uncertain)
        unknown = [p.name for p in sp if p.name not in names]
        if unknown:
            raise ValueError(f"uncertain inputs not in the model: {unknown}")
        base = np.array([float((nominal or {}).get(n, np.nan)) for n in names])
        free = [names.index(p.name) for p in sp]
        fixed_missing = [names[i] for i in range(len(names)) if i not in free and not np.isfinite(base[i])]
        if fixed_missing:
            raise ValueError(f"give a nominal value for {', '.join(fixed_missing)}")

        def full(Xu):
            X = np.repeat(base[None], Xu.shape[0], axis=0)
            X[:, free] = Xu
            return X

        def fn(Xu):
            return model.predict(full(Xu))[:, j]

        res = propagate_space(fn, sp, n=n_propagation, method=propagation_method, limits=limits, seed=seed,
                              level=level)
        Y = res.outputs[:, 0]
        Xall = full(res.inputs)
        gate = domain.assess(Xall)
        slo, shi = conformal_interval(Y, study.q, _scale(scale, model, Xall, Y, j))
        prop = {"n": int(res.n), "method": propagation_method, "mean": float(res.mean[0]), "std": float(res.std[0]),
                "percentiles": {f"p{int(p) if p == int(p) else p}": float(v[0]) for p, v in res.percentiles.items()},
                "histogram": _hist(Y), "share_outside": float(np.mean(gate["status"] == "outside")),
                "share_sparse": float(np.mean(gate["status"] == "sparse")),
                "nominal": {n: float(base[i]) for i, n in enumerate(names) if i not in free},
                "uncertain": [p.to_dict() for p in sp]}
        if limits is not None:
            lo_l, hi_l = limits
            sure = np.ones(Y.size, bool)
            if lo_l is not None:
                sure &= slo >= lo_l
            if hi_l is not None:
                sure &= shi <= hi_l
            ks = int(sure.sum())
            prop.update(p_within=res.p_within, p_within_ci=list(res.p_within_ci),
                        p_within_with_model_error=ks / Y.size,
                        p_within_with_model_error_ci=list(wilson_interval(ks, Y.size, level)))
        if len(sp) <= 8:
            from pinneapple_analysis.uncertainty.pce import n_terms, polynomial_chaos
            if n_terms(len(sp), pce_degree) <= 200:
                p = polynomial_chaos(fn, sp, degree=pce_degree, seed=seed)
                prop["pce"] = {"degree": p.degree, "n_evals": p.n_evals, "mean": float(p.mean[0]),
                               "std": float(p.std[0]), "loo_q2": float(p.loo_q2[0]),
                               "S1": dict(zip(p.names, p.S1[0].tolist(), strict=True)), "ST": dict(zip(p.names, p.ST[0].tolist(), strict=True))}
        out["propagation"] = prop
    return out


__all__ = ["uncertainty_study", "SCALES"]
