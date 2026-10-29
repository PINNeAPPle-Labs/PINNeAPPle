"""The measured studies of the condensation benchmark (each returns plain JSON-able numbers).

Every number in ``docs/research/physics_condensation.md`` comes from one of these functions, run by
``examples/physics_condensation/run_benchmark.py``. Peak memory is the peak resident set size of a fresh process
(:func:`isolated`), so one study does not inherit another's allocations.

* :func:`mesh_convergence` -- the reference solver under joint lateral/vertical refinement, observed order.
* :func:`accuracy_vs_modes` -- condensed vs reference: Tmax, junction field, interface heats, energy balance, full
  field, the error estimate and its effectivity, for a range of interface modes ``m``.
* :func:`refinement_study` -- error-driven face refinement against uniformly more modes, error vs interface dofs.
* :func:`homogenization_study` -- what Keff erases: smooth power, a sub-pitch hotspot, voided bumps.
* :func:`scaling_point` -- one package size: reference (if it fits), flat condensed, hierarchy; time and memory.
* :func:`configurations` -- cost to build vs cost per new power map.
* :func:`learned_study` -- learned block operators with verification vs exact per-block condensation.
* :func:`transient_study` -- Craig-Bampton transient vs the fine reference.
* :func:`root_compressibility` -- numerical ranks of the off-diagonal blocks of the top Schur complement.
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

import numpy as np

from . import fv
from . import homogenize as H
from .hierarchy import Hierarchy
from .package import Package, Variant, hotspot_power, reference_stack, uniform_package
from .substructure import Condensed, energy_norm, errors, reference_observables, tmax_cells


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def isolated(func: str, timeout: float = 3600, module: str = "pinneapple_condense.benchmark", **kwargs) -> dict:
    """Run ``<module>.<func>(**kwargs)`` (default: this module) in a fresh Python process; its result plus the
    peak RSS of that process (``peak_rss_gb``)."""
    code = (f"import json, sys, importlib; from pinneapple_condense import benchmark as b; "
            f"f = getattr(importlib.import_module({module!r}), {func!r}); "
            "r = f(**json.loads(sys.argv[1])); r['peak_rss_gb'] = b.peak_rss_gb(); "
            "print('@@' + json.dumps(r))")
    out = subprocess.run([sys.executable, "-c", code, json.dumps(kwargs)], capture_output=True, text=True,
                         timeout=timeout)
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError(f"{func} failed:\n{out.stderr[-3000:]}")


def _ref(pkg: Package, power=None):
    T, info = fv.solve_package(pkg, power)
    return T, reference_observables(pkg, T), info


def mesh_convergence(shape=(8, 8), levels=((4, 1), (8, 2), (16, 4)), seed: int = 1) -> dict:
    """Reference solutions at increasing resolution (cells per block side, vertical multiplier), the observed order
    from the last three levels, the Richardson extrapolation and the error of the default resolution (8, 1)."""
    rows = []
    for n, v in tuple(levels) + ((8, 1),):
        st = reference_stack(n=n, vertical=v)
        pkg = uniform_package(shape, st, seed=seed)
        T, info = fv.solve_package(pkg)
        obs = reference_observables(pkg, T)
        tmax, tmax_active = tmax_cells(T, st.active_k)
        rows.append({"n": n, "vertical": v, "cells": int(T.size), "seconds": info.seconds, "method": info.method,
                     "Tmax": tmax, "Tmax_active_row": tmax_active, "Tj_mean": float(obs["Tj"].mean()),
                     "q_tim": float(obs["q_tim"].sum()), "q_ubump": float(obs["q_ubump"].sum()),
                     "q_bottom": float(obs["q_bottom"].sum()),
                     "energy_balance": float((obs["q_top"].sum() + obs["q_bottom"].sum()) / pkg.power.sum() - 1)})
    default = rows.pop()
    out = {"levels": rows, "default": default}
    for key in ("Tmax", "Tj_mean", "q_tim", "q_ubump", "q_bottom"):
        a, b, c = (r[key] for r in rows[-3:])
        if (b - a) * (c - b) > 0 and abs(c - b) > 0:
            p = float(np.log(abs(b - a) / abs(c - b)) / np.log(2))
            out[f"order_{key}"] = p
            out[f"richardson_{key}"] = float(c + (c - b) / (2 ** p - 1))
            out[f"err_default_{key}"] = float(abs(default[key] - out[f"richardson_{key}"]))
    return out


def accuracy_vs_modes(shape=(12, 12), ms=(1, 2, 4, 6, 8, 12, 16, 24), seed: int = 1, modes: str = "pod") -> dict:
    pkg = uniform_package(shape, seed=seed)
    T, ref, info = _ref(pkg)
    s = fv.assemble_package(pkg)
    rows = []
    for m in ms:
        c = Condensed(pkg, m, modes=modes).prepare()
        st = c.solve()
        e = errors(c.observables(st), ref, pkg.power.sum())
        Tf = c.field(st)
        eta = c.estimate(st)
        en = energy_norm(s.K, (Tf - T).ravel())
        e.update({"m": m, "dofs": c.n_dofs, "field_max_err": float(np.abs(Tf - T).max()),
                  "field_rel_l2": float(np.linalg.norm(Tf - T) / np.linalg.norm(T)), "energy_err": en,
                  "estimate": float(np.sqrt((eta ** 2).sum())),
                  "effectivity": float(np.sqrt((eta ** 2).sum()) / max(en, 1e-300)),
                  "prepare_s": c.timings["prepare"], "solve_s": st.seconds})
        rows.append(e)
    return {"shape": list(shape), "cells": int(T.size), "reference_s": info.seconds, "reference": info.method,
            "Tmax_ref": float(ref["Tj"].max()), "modes": modes, "rows": rows}


def refinement_study(shape=(12, 12), m0: int = 2, steps: int = 6, seed: int = 1, hot: bool = True) -> dict:
    """Doerfler face refinement from ``m0`` modes (``p``: double the modes of marked faces; ``full``: un-reduce them)
    vs uniform ``m``; a power map with a sub-pitch hotspot (``hot``) makes the error local."""
    pkg = uniform_package(shape, seed=seed)
    if hot:
        p = pkg.power.copy()
        p[p.shape[0] // 3, p.shape[1] // 2] += 0.2 * p.sum()
        pkg.power = p
    T, ref, _ = _ref(pkg)
    s = fv.assemble_package(pkg)
    adaptive = {}
    for mode in ("p", "full"):
        rows = []
        c = Condensed(pkg, m0).prepare()
        for i in range(steps + 1):
            st = c.solve()
            eta = c.estimate(st)
            Tf = c.field(st)
            e = errors(c.observables(st), ref, pkg.power.sum())
            rows.append({"step": i, "dofs": c.n_dofs, "refined_faces": int((c.face_m > m0).sum()),
                         "faces": int(c.face_m.size), "err_Tmax": e["err_Tmax"], "err_Tj_max": e["err_Tj_max"],
                         "energy_err": energy_norm(s.K, (Tf - T).ravel()),
                         "estimate": float(np.sqrt((eta ** 2).sum()))})
            if i < steps:
                c = c.refine(eta, 0.5, mode=mode).prepare()
        adaptive[mode] = rows
    uniform = []
    for m in (m0, 4, 8, 16, 32, 64):
        c = Condensed(pkg, m).prepare()
        st = c.solve()
        Tf = c.field(st)
        e = errors(c.observables(st), ref, pkg.power.sum())
        uniform.append({"m": m, "dofs": c.n_dofs, "err_Tmax": e["err_Tmax"], "err_Tj_max": e["err_Tj_max"],
                        "energy_err": energy_norm(s.K, (Tf - T).ravel())})
    return {"shape": list(shape), "adaptive": adaptive, "uniform": uniform, "Tmax_ref": float(ref["Tj"].max())}


def correction_study(nb: int = 32, ms=(6, 12), seed: int = 0, power_scale: float = 4.0, rtol: float = 1e-9,
                     every: int = 10) -> dict:
    """The exact correction (:meth:`.substructure.Condensed.correct`) on a large hotspot case: iterations to the
    fine solution from each reduced model, and the Tmax error along the way (every ``every`` iterations)."""
    pkg = uniform_package((nb, nb), seed=seed)
    pkg.power = hotspot_power(pkg.stack, pkg.shape, power_scale * pkg.power.sum(), seed=100, hotspots=1)
    t0 = time.perf_counter()
    T, ref, info = _ref(pkg)
    tref = time.perf_counter() - t0
    tm = float(ref["Tj"].max())
    rows = []
    for m in ms:
        c = Condensed(pkg, m).prepare()
        st = c.solve()
        e0 = abs(float(c.observables(st)["Tj"].max()) - tm)
        trail = []
        for it in (every, 2 * every, 4 * every, 8 * every):
            r = c.correct(st, rtol=0.0, maxiter=it)
            trail.append({"iterations": it, "err_Tmax": abs(float(c.observables_fine(r["U"], r["pb"])["Tj"].max()) - tm),
                          "rel_residual": r["history"][-1], "seconds": r["seconds"]})
        r = c.correct(st, rtol=rtol)
        rows.append({"m": m, "prepare_s": c.timings["prepare"], "err_Tmax_reduced": e0, "trail": trail,
                     "iterations_to_rtol": r["iterations"], "seconds_to_rtol": r["seconds"],
                     "err_Tmax_corrected": abs(float(c.observables_fine(r["U"], r["pb"])["Tj"].max()) - tm)})
    return {"blocks": nb * nb, "cells": int(T.size), "Tmax_ref": tm, "reference_s": tref, "rtol": rtol, "rows": rows}


def homogenization_study(shape=(12, 12), seed: int = 1) -> dict:
    out = {}
    base = uniform_package(shape, seed=seed)
    cases = {"smooth power": base}
    hot = uniform_package(shape, seed=seed)
    p = hot.power.copy()
    p[p.shape[0] // 3, p.shape[1] // 2] += 0.2 * p.sum()
    hot.power = p
    cases["sub-pitch hotspot"] = hot
    void = uniform_package(shape, seed=seed)
    void.variants.append(Variant(inclusion_k=(("ubump", 0.05),)))
    bi, bj = np.unravel_index(np.argmax(void.block_power().sum(-1)), shape)
    void.layout[max(0, bi - 1):bi + 2, max(0, bj - 1):bj + 2] = 1
    cases["voided microbumps at the hotspot"] = void
    for name, pkg in cases.items():
        T, ref, info = _ref(pkg)
        t0 = time.perf_counter()
        Th, sysm, _ = H.solve_homogenized(pkg, 1)
        th = time.perf_counter() - t0
        hom = H.homogenized_observables(pkg, Th, sysm, 1)
        r = H.compare(pkg, ref, hom)
        c = Condensed(pkg, 12).prepare()
        st = c.solve()
        e = errors(c.observables(st), ref, pkg.power.sum())
        if name == "sub-pitch hotspot":
            n = pkg.stack.n
            out["maps"] = {"fine": T[:, :, pkg.stack.active_k].tolist(),
                           "hom": np.kron(hom["Tj_mean"], np.ones((n, n))).tolist()}
        r.update({"reference_s": info.seconds, "homogenized_s": th, "condensed_m12_err_Tmax": e["err_Tmax"],
                  "condensed_m12_err_Tj_max": e["err_Tj_max"], "condensed_m12_err_q_ubump": e["err_q_ubump"],
                  "condensed_m12_s": c.timings["prepare"] + st.seconds})
        out[name] = r
    return out


def scaling_point(nb: int, m: int = 12, reference: bool = True, flat: bool = True, hier: bool = True,
                  seed: int = 2) -> dict:
    """One ``nb x nb`` package: wall time and result of each method that is asked for (call through
    :func:`isolated` per method for its own peak memory)."""
    pkg = uniform_package((nb, nb), seed=seed)
    out = {"blocks": nb * nb, "cells": pkg.n_cells}
    if reference:
        t0 = time.perf_counter()
        T, ref, info = _ref(pkg)
        out["reference"] = {"seconds": time.perf_counter() - t0, "method": info.method, "iterations": info.iterations,
                            "Tmax": float(ref["Tj"].max())}
    if flat:
        c = Condensed(pkg, m).prepare()
        st = c.solve()
        o = c.observables(st)
        out["flat"] = {"prepare_s": c.timings["prepare"], "solve_s": st.seconds, "dofs": c.n_dofs,
                       "Tmax": float(o["Tj"].max()), "timings": c.timings}
    if hier:
        h = Hierarchy(pkg, m).prepare()
        sol = h.solve()
        o = h.observables(sol)
        out["hierarchy"] = {"prepare_s": h.timings["prepare"], "solve_s": sol["seconds"],
                            "Tmax": float(o["Tj"].max()), "timings": h.timings,
                            "energy_balance": float((o["q_top"].sum() + o["q_bottom"].sum()) / pkg.power.sum() - 1)}
    return out


def configurations(nb: int = 64, m: int = 12, n_configs: int = 20, seed: int = 0, check: int = 2) -> dict:
    """Build once, then solve ``n_configs`` new power maps; the first ``check`` are also solved by the reference."""
    pkg = uniform_package((nb, nb), seed=seed)
    t0 = time.perf_counter()
    h = Hierarchy(pkg, m).prepare()
    build = time.perf_counter() - t0
    per, err, ref_s = [], [], []
    for i in range(n_configs):
        p = hotspot_power(pkg.stack, pkg.shape, pkg.power.sum(), seed=100 + i, hotspots=1 + i % 5)
        sol = h.solve(p)
        o = h.observables(sol)
        per.append(sol["seconds"])
        if i < check:
            t1 = time.perf_counter()
            T, ref, _ = _ref(pkg, p)
            ref_s.append(time.perf_counter() - t1)
            err.append(errors(o, ref, p.sum()))
    return {"blocks": nb * nb, "build_s": build, "per_config_s": per, "reference_s": ref_s,
            "errors": err, "timings": h.timings}


def learned_study(nb: int = 32, m: int = 12, n_train: int = 48, seed: int = 1, defects: int = 4,
                  exact_reference: bool = True, tol_T: float = 0.01) -> dict:
    from .learned import LearnedBlocks, solve_adaptive, solve_exact, variant_field
    st = reference_stack()
    t0 = time.perf_counter()
    L = LearnedBlocks(st, m=m, n_train=n_train, seed=seed).fit()
    build = time.perf_counter() - t0
    shape = (nb, nb)
    theta = variant_field(shape, seed=seed, defects=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(nb * nb, defects, replace=False)
    th = theta.reshape(-1, 3)
    th[idx[: defects // 2], 0] = 0.05             # nearly open microbumps
    th[idx[defects // 2:], 1] = 30.0              # delaminated TIM (contact resistance x 30)
    p = hotspot_power(st, shape, 1e6 * nb * nb * st.pitch ** 2, seed=seed + 3)
    r = solve_adaptive(L, theta, p, tol_T=tol_T)
    r0 = solve_adaptive(L, theta, p, tol_std=1e9, max_rounds=1, n_hot=0, n_audit=0)
    out = {"blocks": nb * nb, "n_train": n_train, "build_s": build, "train": L.timings, "rank": L.rom.rank,
           "adaptive": {"timings": r.timings, "exact_blocks": int(r.exact_blocks.sum()),
                        "checked": int(r.checked.sum()), "failed_per_round": [len(f) for f in r.failed],
                        "flagged_std": r.flagged_std, "flagged_ood": r.flagged_ood, "estimate": r.estimate},
           "learned_only_s": r0.timings["total"]}
    if exact_reference:
        o_ex, tex = solve_exact(L, theta, p)
        tm = float(o_ex["Tj"].max())

        def err(o):
            return {"err_Tmax": abs(float(o["Tj"].max()) - tm), "err_Tj_max": float(np.abs(o["Tj"] - o_ex["Tj"]).max()),
                    "err_q_tim": float(np.abs(o["q_tim"] - o_ex["q_tim"]).sum() / np.abs(o_ex["q_tim"]).sum())}
        out.update({"exact_s": tex, "Tmax_exact": tm, "adaptive_err": err(r.obs), "learned_only_err": err(r0.obs)})
        # what the defects do (exact, with vs without them)
        clean = variant_field(shape, seed=seed, defects=0)
        o_c, _ = solve_exact(L, clean, p) if nb <= 16 else (None, None)
        if o_c is not None:
            out["defect_effect_Tj_max"] = float(np.abs(o_ex["Tj"] - o_c["Tj"]).max())
    return out


def transient_study(shape=(6, 6), dt: float = 2e-4, steps: int = 150, mk=((8, 8), (16, 16), (32, 32)),
                    seed: int = 1) -> dict:
    from .transient import CBTransient, reference_transient
    pkg = uniform_package(shape, seed=seed)
    p1 = pkg.power
    p2 = hotspot_power(pkg.stack, shape, p1.sum(), seed=seed + 6)
    powers = np.array([p1 if (i // 50) % 2 == 0 else p2 for i in range(steps)])
    ref = reference_transient(pkg, powers, dt)
    rows = []
    for m, k in mk:
        cb = CBTransient(pkg, m, k).prepare(dt)
        out = cb.run(powers)
        rows.append({"m": m, "k": k, "dofs": cb.n, "prepare_s": cb.timings["prepare"], "run_s": out["seconds"],
                     "err_Tmax_max": float(np.abs(out["Tmax"] - ref["Tmax"]).max()),
                     "err_Tj_max": float(np.abs(out["Tj"] - ref["Tj"]).max()),
                     "err_q_tim_rel": float(np.abs(out["q_tim"] - ref["q_tim"]).max() / np.abs(ref["q_tim"]).max()),
                     "Tmax": out["Tmax"].tolist()})
    return {"shape": list(shape), "dt": dt, "steps": steps, "cells": ref["n_cells"], "reference_s": ref["seconds"],
            "Tmax_ref": ref["Tmax"].tolist(), "q_tim_ref": ref["q_tim"].tolist(), "rows": rows}


def root_compressibility(nb: int = 64, m: int = 12, tol: float = 1e-8) -> dict:
    """Numerical rank (relative ``tol``) of the off-diagonal blocks between the four sides of the root Schur
    complement: how much an H-matrix representation of the top operator could save."""
    pkg = uniform_package((nb, nb))
    h = Hierarchy(pkg, m, keep_root=True).prepare()
    S = h.ops[int(h.keys[0][0])].S
    n = S.shape[0] // 4
    out = {"blocks": nb * nb, "root_ports": int(S.shape[0]), "side_dofs": n, "ranks": {}}
    names = "WESN"
    for i in range(4):
        for j in range(i + 1, 4):
            B = S[i * n:(i + 1) * n, j * n:(j + 1) * n]
            sv = np.linalg.svd(B, compute_uv=False)
            out["ranks"][names[i] + names[j]] = int((sv > tol * sv[0]).sum()) if sv[0] > 0 else 0
    # within one side: far-field interaction of the first and last quarter
    q = n // 4
    sv = np.linalg.svd(S[:q, n - q:n], compute_uv=False)
    out["ranks"]["W_near_far_quarter"] = int((sv > tol * sv[0]).sum())
    out["quarter_dofs"] = q
    return out


if __name__ == "__main__":
    print(json.dumps(isolated(sys.argv[1], **json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")), indent=1))


def meshless_study(nb: int = 2, hfs=(6, 8, 10, 12), methods=("gfdm", "rbf_fd"), interfaces=("pairs", "naive"),
                   ls_hfs=(8, 10), seed: int = 1, keep_fields: int = 8) -> dict:
    """Track 1b: RBF-FD and GFDM point clouds from the implicit geometry against the fine FV reference, for a range
    of spacings ``pitch / hf``, interface treatments (node pairs with flux continuity and the contact law, or the
    naive single-node collocation), and least-squares collocation (``ls_hfs``). Keeps the junction map and a
    vertical slice of the point cloud for ``hf = keep_fields`` (figures)."""
    from . import meshless as ML
    pkg = uniform_package((nb, nb), seed=seed)
    st = pkg.stack
    t0 = time.perf_counter()
    T, info = fv.solve_package(pkg, method="direct")
    t_ref = time.perf_counter() - t0
    ref = reference_observables(pkg, T)
    tj_ref = T[:, :, st.active_k]
    qref = float(ref["q_tim"].sum())
    rows, fields = [], {}
    configs = [(hf, m, itf, False) for hf in hfs for m in methods for itf in interfaces]
    configs += [(hf, m, "pairs", True) for hf in ls_hfs for m in methods]
    for hf, m, itf, ls in configs:
        row = {"hf": hf, "method": m, "interface": itf, "least_squares": ls}
        try:
            cl = ML.point_cloud(pkg, h_xy=st.pitch / hf, interface=itf)
            sol = ML.solve(pkg, cl, method=m, interface=itf, oversample=ls, condition=cl.n <= 60_000 and not ls)
            o = ML.observables(sol, pkg)
            row.update({"nodes": cl.n, "Tmax": o["Tmax"], "err_Tmax": o["Tmax"] - float(tj_ref.max()),
                        "err_Tj_max": float(np.abs(o["Tj"] - tj_ref).max()),
                        "err_q_tim": (o["q_tim"] - qref) / qref, "energy_balance": o["energy_balance"],
                        "points_s": sol.timings["points_s"], "assemble_s": sol.timings["assemble_s"],
                        "solve_s": sol.timings["solve_s"], "cond1_est": sol.timings.get("cond1_est")})
            if hf == keep_fields and itf == "pairs" and not ls:
                X = cl.X
                sl = np.abs(X[:, 1] - st.pitch * nb / 2) < st.pitch / hf / 2
                fields[m] = {"Tj": o["Tj"].tolist(), "slice_x": X[sl, 0].tolist(), "slice_z": X[sl, 2].tolist(),
                             "slice_T": sol.T[sl].tolist(), "slice_kind": cl.kind[sl].tolist()}
        except Exception as e:                       # a failed solve is a result
            row.update({"failed": repr(e)[:200]})
        rows.append(row)
    return {"blocks": nb * nb, "fv_cells": int(T.size), "fv_s": t_ref, "Tmax_ref": float(tj_ref.max()),
            "q_tim_ref": qref, "Tj_ref": tj_ref.tolist(), "rows": rows, "fields": fields}


def exceptions_study(shape=(12, 12), seed: int = 1, budgets=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4)) -> dict:
    """Track 1c: the Keff prior plus residual-selected exception patches against random patches at the same
    budget, on a smooth power map and on one with a sub-pitch hotspot; and the 1-D column prior for comparison."""
    from . import exceptions as EX
    out = {}
    for name in ("smooth power", "sub-pitch hotspot"):
        pkg = uniform_package(shape, seed=seed)
        if name == "sub-pitch hotspot":
            p = pkg.power.copy()
            p[p.shape[0] // 3, p.shape[1] // 2] += 0.2 * p.sum()
            pkg.power = p
        out[name] = EX.pareto(pkg, budgets=budgets, halos=(0, 1))
        col = EX.pareto(pkg, budgets=(0.0,), halos=(0,), selects=("residual",), prior_kind="column")
        out[name]["column_prior"] = col["rows"][0]
    return out
