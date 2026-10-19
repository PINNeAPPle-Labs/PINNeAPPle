"""Red team: search for escapes (wrong but passing) and false alarms (right but rejected) against a defence.

* ``evasion_search``: for one generator and one correct case, maximise the true error of the generated output
  (against the reference solver) while every check stays under its threshold. Objective (minimised on the unit box):
  ``-min(error, 2) + 10 * sum_j max(0, margin_j + 0.02)``, ``margin_j = value_j / threshold_j - 1``; the best
  point that the defence accepts is the escape, if its error is at least ``min_error`` (10 % by default).
* ``false_alarm_search``: over correct cases (the problem's ``variation_space``: coefficient contrast, feature size,
  amplitude, viscosity, solver resolution, for several random fields), maximise the largest margin of the defence:
  the most suspicious-looking correct case, a false alarm when its margin is positive.
* ``mutation_score``: the share of the fault library's mutants that matter (true error >= ``min_error``) which the
  defence rejects, from a fault-injection detection matrix; the mutants that pass are escapes too.

Every search is seeded; a finding carries what it needs to be regenerated (case seed and variation, generator and
parameters).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..fault_injection.matrix import DetectionMatrix
from ..fault_injection.problems import BenchmarkProblem, Case
from .blue import Defence
from .generators import Generator
from .search import minimize

__all__ = ["Finding", "evasion_search", "false_alarm_search", "mutation_score", "mutation_escapes"]


@dataclass
class Finding:
    """An escape (``kind="escape"``: wrong, accepted) or a false alarm (``kind="false_alarm"``: correct, rejected),
    or the best attempt at one (``found`` False)."""
    kind: str
    problem: str
    seed: int
    variation: dict[str, float]
    generator: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    true_error: float = 0.0
    values: dict[str, float] = field(default_factory=dict)
    margin: float = -math.inf                 # largest margin of the defence (> 0: rejected)
    found: bool = False
    method: str = ""
    evaluations: int = 0
    round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "problem": self.problem, "seed": int(self.seed), "variation": self.variation,
                "generator": self.generator, "params": self.params, "true_error": float(self.true_error),
                "values": {k: float(v) for k, v in self.values.items()}, "margin": float(self.margin),
                "found": bool(self.found), "method": self.method, "evaluations": int(self.evaluations),
                "round": int(self.round)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Finding:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def case(self, problem: BenchmarkProblem) -> tuple[Case, np.ndarray]:
        """Regenerate the case (and its correct field)."""
        from .generators import get_generator
        base = problem.case(self.seed, **self.variation)
        if self.generator is None:
            return base, base.field
        return get_generator(self.generator).apply(problem, base, self.params)


def _max_margin(defence: Defence, values: dict[str, float]) -> float:
    m = defence.margins(values)
    return max(m.values()) if m else -math.inf


def evasion_search(problem: BenchmarkProblem, defence: Defence, generator: Generator, base: Case, *,
                   budget: int = 200, method: str = "de", seed: int = 0, min_error: float = 0.1) -> Finding:
    """The worst output of ``generator`` on ``base`` that ``defence`` accepts (see the module docstring)."""
    best: dict[str, Any] = {"err": -1.0}

    def objective(z: np.ndarray) -> float:
        q = generator.params(z)
        try:
            faulty, ref = generator.apply(problem, base, q)
        except Exception:  # noqa: BLE001 - a parameter the generator cannot realise is just a bad point
            return 1e6
        if not np.all(np.isfinite(faulty.field)):
            return 1e6
        err = problem.true_error(faulty, ref)
        vals = defence.scores(faulty)
        marg = defence.margins(vals)
        pen = sum(max(0.0, (m if math.isfinite(m) else 1e3) + 0.02) for m in marg.values())
        if pen == 0.0 and err > best["err"]:
            best.update(err=err, params=q, values=vals, margin=max(marg.values()))
        return -min(err, 2.0) + 10.0 * min(pen, 1e3)

    res = minimize(objective, generator.dim, budget, method=method, seed=seed)
    f = Finding("escape", problem.name, base.seed, dict(base.variation), generator.name, method=method,
                evaluations=res.evaluations)
    if best["err"] >= 0:
        f.params, f.true_error, f.values, f.margin = best["params"], best["err"], best["values"], best["margin"]
        f.found = best["err"] >= min_error
    return f


def false_alarm_search(problem: BenchmarkProblem, defence: Defence, *, seeds: Sequence[int], budget: int = 100,
                       method: str = "de", search_seed: int = 0) -> list[Finding]:
    """For each random field in ``seeds``, the correct case of ``problem.variation_space`` with the largest margin
    of ``defence`` (a false alarm when it is positive). Sorted from the most suspicious."""
    keys = list(problem.variation_space)
    out = []
    for s in seeds:
        best: dict[str, Any] = {"margin": -math.inf}

        def objective(z: np.ndarray, s=s, best=best) -> float:
            var = {k: float(lo + (hi - lo) * zi) for (k, (lo, hi)), zi in zip(problem.variation_space.items(), z,
                                                                               strict=True)}
            try:
                c = problem.case(s, **var)
            except Exception:  # noqa: BLE001
                return 1e6
            vals = defence.scores(c)
            m = _max_margin(defence, vals)
            if m > best["margin"]:
                best.update(margin=m, variation=var, values=vals)
            return -min(m, 1e3)

        res = minimize(objective, len(keys), budget, method=method, seed=search_seed + int(s) % 1000)
        out.append(Finding("false_alarm", problem.name, int(s), best.get("variation", {}), None,
                           values=best.get("values", {}), margin=best["margin"], found=best["margin"] > 0,
                           method=method, evaluations=res.evaluations))
    return sorted(out, key=lambda f: -f.margin)


def mutation_score(matrix: DetectionMatrix, thresholds: dict[str, float], min_error: float = 0.1) -> dict[str, Any]:
    """Share of the mutants (every fault x magnitude x clean case of ``matrix``) with true error >= ``min_error``
    that the thresholds reject; per fault too."""
    killed = total = 0
    per: dict[str, list[int]] = {}
    for (f, m), vals in matrix.scores.items():
        err = matrix.errors[(f, m)]
        rej = np.any([vals[c] > t for c, t in thresholds.items()], axis=0) | ~np.all(
            [np.isfinite(vals[c]) for c in thresholds], axis=0)
        sel = err >= min_error
        k, n = int((rej & sel).sum()), int(sel.sum())
        killed += k
        total += n
        a = per.setdefault(f, [0, 0])
        a[0] += k
        a[1] += n
    return {"score": killed / total if total else math.nan, "killed": killed, "mutants": total,
            "min_error": min_error, "per_fault": {f: (k / n if n else math.nan, n) for f, (k, n) in per.items()}}


def mutation_escapes(matrix: DetectionMatrix, thresholds: dict[str, float], *, seed: int, min_error: float = 0.1,
                     per_fault: int = 1) -> list[Finding]:
    """The mutants that matter and pass, the ``per_fault`` largest-error ones of each fault, as findings that
    regenerate with the ``fault:<name>`` generators."""
    out = []
    for f in matrix.faults:
        cand = []
        for (ff, m), vals in matrix.scores.items():
            if ff != f:
                continue
            err = matrix.errors[(f, m)]
            for i in range(len(err)):
                v = {c: float(vals[c][i]) for c in thresholds}
                if err[i] >= min_error and all(math.isfinite(x) and x <= thresholds[c] for c, x in v.items()):
                    cand.append((float(err[i]), m, i, v))
        for err, m, i, v in sorted(cand, key=lambda t: -t[0])[:per_fault]:
            s, var = matrix.trials[i]
            out.append(Finding("escape", matrix.problem, int(s), dict(var), f"fault:{f}",
                               {"magnitude": float(m), "rng_seed": int(seed)}, err, v,
                               max(x / t - 1 if t > 0 else (1 + x if x > t else -1.0) for (c, x), t in
                                   zip(v.items(), thresholds.values(), strict=True)), True, "mutation"))
    return out
