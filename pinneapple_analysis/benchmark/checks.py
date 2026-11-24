"""Correctness gates of a benchmark run.

* Architecture gates (before any training): each method's ``architecture_checks`` on its first search configuration:
  forward shapes and dtype, parameter count, FNO modes <= Nyquist and resolution invariance, DeepONet branch/trunk
  shapes, MeshGraphNet permutation equivariance, PINN autograd vs finite differences and the residual of an exact
  solution, and that the model can overfit a tiny batch.
* Training gates (after the runs, from every run file): loss finite and decreasing, divergence, "did not learn"
  (final loss not below half the initial one, or an error no better than the trivial prediction), the budget enforced,
  identical accounting across methods, evaluation off the clock, no test data in the search, train/validation gap,
  seed spread (veriphysics tolerance), and a determinism spot-check (same seed, same steps, same result).

Every check is recorded (pass / FAIL / warn / n/a with its value); none is skipped silently.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget

from .registry import Check

DID_NOT_CONVERGE = 0.9          # a relative error at or above this is no better than predicting zero


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def architecture_gates(plan) -> list[Check]:
    out = []
    for pid, mid, prob, meth in plan.pairs():
        cfg = plan.search_configs(prob, meth)[0]
        try:
            out += meth.architecture_checks(prob, plan.ws.prepared(pid), cfg)
        except Exception as e:                                       # a crashing gate is a failed gate
            out.append(Check("architecture checks", "architecture", f"{pid}/{mid}", False, detail=f"crashed: {e!r}"))
    return out


def _losses(res: dict) -> list[float]:
    return [h.get("loss") for h in res.get("train", {}).get("history", []) if h.get("loss") is not None]


def job_checks(rec: dict) -> list[Check]:
    """Checks of one finished job, from its run file."""
    spec, subj = rec["spec"], f"{rec['spec']['problem']}/{rec['spec']['method']}"
    if not rec.get("ok"):
        return [Check("job finished", "training", subj, False, detail=rec.get("error", "")[-300:], value=spec["id"])]
    res = rec["result"]
    tr = res.get("train", {})
    L = _losses(res)
    C = []
    if not L:
        return [Check("loss recorded", "training", subj, None, value=spec["id"], detail="no loss history")]
    fin = all(_finite(x) for x in L)
    C.append(Check("loss finite", "training", subj, fin, value=spec["id"]))
    k = max(1, len(L) // 5)
    first, last = float(np.mean(L[:k])), float(np.mean(L[-k:]))
    dec = fin and last < first
    C.append(Check("loss decreasing", "training", subj, dec if len(L) >= 3 else None, value=f"{first:.3g} -> {last:.3g}",
                   severity="warning", detail=spec["id"]))
    # Divergence: the final loss is ten times the minimum the run reached *and* still above a thousandth of where it
    # started. Without the second half, a run that reached the noise floor (a loss of 1e-11 in float32 after starting
    # at 0.2) is labelled divergent for a 1e-8 wobble: nothing blew up, the arithmetic ran out of digits.
    div = (not fin) or (_finite(L[-1]) and L[-1] > 10 * max(min(L), 1e-30) and L[-1] > 1e-3 * max(L[0], 1e-30))
    C.append(Check("no divergence", "training", subj, not div, value=f"final {L[-1]:.3g}, min {min(L):.3g}",
                   detail=spec["id"]))
    lvl = spec.get("level")
    full = spec["stage"] == "flops" or (spec["stage"] == "final" and (lvl is None or float(lvl) >= 1.0 - 1e-9))
    if full:
        learned = fin and L[-1] < 0.5 * L[0]
        C.append(Check("learned (final loss < half the initial)", "training", subj, learned, severity="warning",
                       value=f"{L[0]:.3g} -> {L[-1]:.3g}", detail=spec["id"]))
    else:                       # a short run of the error-versus-budget curve is expected to learn less
        C.append(Check("learned (final loss < half the initial)", "training", subj, None, value=spec["id"],
                       severity="warning",
                       detail="n/a: a search run or a level below the full budget (expected to learn less)"))
    b = spec["budget"]
    wall = tr.get("wall_s", 0.0)
    if b.get("wall_s"):
        over = wall - b["wall_s"]
        steps = max(1, tr.get("steps", 1))
        per_step = wall / steps
        ok = over <= max(2 * per_step, 0.05 * b["wall_s"]) or tr.get("stopped_by") in ("flops", "steps", "converged")
        C.append(Check("budget enforced", "training", subj, ok, value=f"{wall:.1f} s of {b['wall_s']:.0f} s",
                       detail=f"{spec['id']}, stopped by {tr.get('stopped_by')}"))
    C.append(Check("evaluation off the clock", "training", subj, "eval_wall_s" in tr, severity="warning",
                   detail="train_within_budget pauses the clock during evaluation"))
    if spec["stage"] == "search":
        touched = [k for k in ("test", "ood", "naca") if k in res]
        C.append(Check("search used validation data only", "training", subj, not touched, value=touched or "val",
                       detail=spec["id"]))
    return C


def training_gates(ws, plan) -> tuple[list[Check], dict[str, list[Check]]]:
    """All job checks, aggregated per (problem, method), plus the cross-run checks."""
    import json
    per_pair: dict[str, list[Check]] = defaultdict(list)
    hw = set()
    current = {f"{p}/{m}" for p, m, _, _ in plan.pairs()}
    for p in sorted(ws.runs.glob("*.json")):
        rec = json.loads(p.read_text())
        sp = rec.get("spec", {})
        if sp.get("stage") not in ("search", "final", "flops") or f"{sp.get('problem')}/{sp.get('method')}" not in current:
            continue
        per_pair[f"{sp['problem']}/{sp['method']}"] += job_checks(rec)
        h = rec.get("hardware", {})
        hw.add((h.get("cpu"), h.get("torch"), sp.get("threads", 1)))
    agg = []
    for subj, cs in sorted(per_pair.items()):
        by = defaultdict(list)
        for c in cs:
            by[c.name].append(c)
        for name, group in by.items():
            fails = [c for c in group if c.passed is False]
            applicable = [c for c in group if c.passed is not None]
            agg.append(Check(name, "training", subj, None if not applicable else not fails,
                             value=f"{len(applicable) - len(fails)}/{len(applicable)} jobs",
                             severity=(applicable or group)[0].severity,
                             detail="; ".join(f"{c.detail} ({c.value})" for c in fails[:3])))
    agg.append(Check("identical accounting across methods (hardware, threads, timer, FLOP counter)", "training",
                     "all", len(hw) <= 1, value=[list(x) for x in hw]))
    return agg, per_pair


def determinism_check(plan, pid: str, mid: str, steps: int = 20) -> Check:
    """Run one job twice in this process with the same seed and a step budget: the results must agree."""
    prob, meth = plan.P[pid], plan.M[mid]
    cfg = plan.best(pid, mid).best_config
    inst = (prob.instances().get("val") or [("", None)])[0][1] if meth.per_instance else None
    outs = []
    for _ in range(2):
        r = meth.train(prob, plan.ws.prepared(pid), cfg, 0, ComputeBudget(steps=steps), final=False, **(inst or {}))
        outs.append(r)
    a, b = (_losses(o)[-1] for o in outs)
    rel = abs(a - b) / max(abs(a), 1e-30)
    return Check("determinism (same seed, same steps, same loss)", "training", f"{pid}/{mid}", rel < 1e-5,
                 value=f"{a:.6g} vs {b:.6g}", threshold=1e-5, detail=f"{steps} steps, seed 0, in-process")


def seed_spread(values: list[float], subj: str, what: str) -> Check:
    from pinneapple_veriphysics.applicability import Tolerances
    tol = Tolerances(plateau_rel=0.05).spread_rel
    v = np.asarray([x for x in values if _finite(x)], float)
    if len(v) < 2:
        return Check(f"seed spread of {what} within the veriphysics tolerance", "training", subj, None,
                     detail="fewer than 2 seeds")
    rel = float(v.std(ddof=1) / max(abs(v.mean()), 1e-30))
    return Check(f"seed spread of {what} within the veriphysics tolerance", "training", subj, rel <= tol,
                 value=round(rel, 3), threshold=tol, severity="warning")


def row_status(arch: list[Check], train: list[Check], skipped: bool, did_not_converge: bool) -> str:
    if skipped:
        return "checks skipped"
    bad = [c for c in arch + train if c.passed is False and c.severity == "error"]
    if bad:
        return "INVALID"
    if did_not_converge:
        return "did not converge"
    warn = [c for c in arch + train if c.passed is False]
    return "pass (warnings)" if warn else "pass"


def summarise(checks: list[Check]) -> dict[str, Any]:
    n = defaultdict(int)
    for c in checks:
        n[c.status] += 1
    return dict(n)
