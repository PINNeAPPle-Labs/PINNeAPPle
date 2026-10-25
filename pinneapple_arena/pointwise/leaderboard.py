"""Global ranking of architectures over many arena experiments (the PINNeAPPle Arena leaderboard).

``aggregate(experiments)`` takes completed experiments (``{"id", "winner", "problem_id", "model_results"}``, the
``model_results`` as written by the arena pipeline) and returns one entry per architecture: mean relative L2, MAPE,
RMSE, R2, max error, training time, convergence rate (relative L2 below 0.1), PDE evaluations, peak memory and
inference time, ranked by mean relative L2 (lower is better), plus 0-100 scores per objective:

- ``accuracy`` is absolute: ``(1 - mean relative L2) * 100``, floored at 0;
- ``training_cost``, ``memory`` and ``inference_speed`` are relative to the other entries of the same call
  (min-max), so they depend on the hardware and on which architectures ran;
- ``convergence`` is the convergence rate in percent.

The mean relative L2 pools experiments on different problems; filter ``experiments`` to one problem (or one domain)
for a like-for-like ranking.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def aggregate(experiments: Iterable[Mapping], families: Mapping[str, str] | None = None) -> dict:
    """``{"total_experiments", "total_architectures", "problems_covered", "entries"}``; ``families`` maps an
    architecture id to its family label (e.g. from ``pinneapple_arena.pointwise.ARCHITECTURES_CATALOG``)."""
    families = dict(families or {})
    raw_experiments = [e for e in experiments if isinstance(e.get("model_results"), list)]
    total_experiments = len(raw_experiments)
    arch_stats: dict[str, dict] = {}
    problems_seen: set[str] = set()

    for raw in raw_experiments:
        model_results = raw["model_results"]
        if not isinstance(model_results, list):
            continue

        winner = raw["winner"]
        pid = raw["problem_id"]
        if pid:
            problems_seen.add(pid)

        for mr in model_results:
            if not isinstance(mr, dict):
                continue
            arch_id = mr.get("architecture_id") or mr.get("arch_id", "")
            if not arch_id:
                continue

            arch_name = mr.get("architecture_name") or arch_id
            metrics = mr.get("metrics") or {}

            family = families.get(arch_id, "")

            if arch_id not in arch_stats:
                arch_stats[arch_id] = {
                    "architecture_id": arch_id,
                    "architecture_name": arch_name,
                    "family": family,
                    "n_experiments": 0,
                    "n_wins": 0,
                    "l2_errors": [],
                    "mapes": [],
                    "rmses": [],
                    "r2_scores": [],
                    "max_errors": [],
                    "training_times": [],
                    "n_converged": 0,
                    "problems": set(),
                    "pde_evaluations": [],
                    "peak_memory_mb": [],
                    "inference_times_ms": [],
                }

            s = arch_stats[arch_id]
            s["n_experiments"] += 1
            if winner == arch_id:
                s["n_wins"] += 1
            if pid:
                s["problems"].add(pid)

            def _first(*keys, metrics=metrics):
                """Return the first non-None value across multiple metric key names."""
                for k in keys:
                    v = metrics.get(k)
                    if v is not None:
                        return v
                return None

            l2    = _first("l2_relative_error", "l2_relative")
            mape  = _first("mape")
            rmse  = _first("rmse")
            r2    = _first("r2_score", "r2")
            max_e = _first("max_error")
            t     = mr.get("training_time_s")
            # converged: explicit flag from pipeline, or inferred from L2 < 10%
            conv  = mr.get("converged") or (l2 is not None and l2 < 0.1)

            if l2 is not None:
                s["l2_errors"].append(float(l2))
            if mape is not None and float(mape) < 1e6:  # skip inf/overflow MAPE
                s["mapes"].append(float(mape))
            if rmse is not None:
                s["rmses"].append(float(rmse))
            if r2 is not None:
                s["r2_scores"].append(float(r2))
            if max_e is not None:
                s["max_errors"].append(float(max_e))
            if t is not None:
                s["training_times"].append(float(t))
            if conv:
                s["n_converged"] += 1

            pde_evals = mr.get("n_pde_evaluations")
            peak_mem  = mr.get("peak_gpu_memory_mb") if mr.get("peak_gpu_memory_mb") is not None else mr.get("peak_cpu_memory_mb")
            infer_ms  = mr.get("inference_time_ms")
            if pde_evals is not None:
                s["pde_evaluations"].append(float(pde_evals))
            if peak_mem is not None:
                s["peak_memory_mb"].append(float(peak_mem))
            if infer_ms is not None:
                s["inference_times_ms"].append(float(infer_ms))

    def _avg(lst: list) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    def _avg_robust(lst: list) -> float | None:
        """Average with outlier rejection for RMSE.
        Returns None when the median is > 100 (all experiments diverged — showing a
        trillion-scale value would be more confusing than a dash)."""
        if not lst:
            return None
        sorted_lst = sorted(lst)
        med = sorted_lst[len(sorted_lst) // 2]
        if med > 100:
            return None  # every result diverged — hide rather than mislead
        if med <= 0:
            clean = [v for v in lst if v >= 0]
        else:
            clean = [v for v in lst if v <= med * 1000]
        if not clean:
            return None
        return round(sum(clean) / len(clean), 4)

    entries = []
    for arch_id, s in arch_stats.items():
        n = s["n_experiments"]
        entries.append({
            "architecture_id": arch_id,
            "architecture_name": s["architecture_name"],
            "family": s["family"],
            "n_experiments": n,
            "n_wins": s["n_wins"],
            "win_rate": round(s["n_wins"] / n, 4) if n else 0.0,
            "avg_l2_error": _avg(s["l2_errors"]),
            "avg_mape": _avg_robust(s["mapes"]),
            "avg_rmse": _avg_robust(s["rmses"]),
            "avg_r2_score": _avg(s["r2_scores"]),
            "avg_max_error": _avg(s["max_errors"]),
            "avg_training_time_s": _avg(s["training_times"]),
            "convergence_rate": round(s["n_converged"] / n, 4) if n else 0.0,
            "problems_count": len(s["problems"]),
            "avg_n_pde_evaluations": _avg(s["pde_evaluations"]),
            "avg_peak_memory_mb": _avg(s["peak_memory_mb"]),
            "avg_inference_time_ms": _avg(s["inference_times_ms"]),
            "scores": None,
        })

    # Sort: lower L2 error is better; entries without L2 data go to the end
    entries.sort(key=lambda e: (
        e["avg_l2_error"] is None,
        e["avg_l2_error"] if e["avg_l2_error"] is not None else float("inf"),
    ))

    for i, entry in enumerate(entries):
        entry["rank"] = i + 1

    # ── Multi-objective scores (0–100, higher = better) ──────────────────────
    # Accuracy: ABSOLUTE — based on L2 relative error, not rank within the group.
    # (1 - L2) * 100: L2=0 → 100, L2=0.5 → 50, L2=1.0+ → 0
    # All others: RELATIVE — normalized within this result set (hardware-dependent).
    def _accuracy_score(l2: float | None) -> float | None:
        if l2 is None:
            return None
        return round(max(0.0, (1.0 - l2) * 100), 1)

    def _norm_lower_better(all_vals: list, val: float | None) -> float | None:
        valid = [v for v in all_vals if v is not None]
        if not valid or val is None:
            return None
        lo, hi = min(valid), max(valid)
        if hi == lo:
            return 100.0
        return round((1.0 - (val - lo) / (hi - lo)) * 100, 1)

    all_time  = [e["avg_training_time_s"] for e in entries]
    all_mem   = [e["avg_peak_memory_mb"] for e in entries]
    all_infer = [e["avg_inference_time_ms"] for e in entries]

    for e in entries:
        e["scores"] = {
            "accuracy":        _accuracy_score(e["avg_l2_error"]),
            "training_cost":   _norm_lower_better(all_time,  e["avg_training_time_s"]),
            "convergence":     round(e["convergence_rate"] * 100, 1),
            "memory":          _norm_lower_better(all_mem,   e["avg_peak_memory_mb"]),
            "inference_speed": _norm_lower_better(all_infer, e["avg_inference_time_ms"]),
        }

    return {
        "total_experiments": total_experiments,
        "total_architectures": len(entries),
        "problems_covered": len(problems_seen),
        "entries": entries,
    }
