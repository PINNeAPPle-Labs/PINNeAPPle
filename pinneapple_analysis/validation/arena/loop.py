"""The arena loop: rounds of red team then blue team, and the report.

``run_arena(problem, checks=None, budget=...)``:

* **Round 0 defence**: ``checks`` with their own tolerances (default: ``default_checks(problem)``, what users run
  today).
* **Red** (each round): the mutation score of the defence on the fault library (and the mutants that matter but
  pass), evasion searches with every generator on ``n_bases`` correct cases, and the false-alarm search over correct
  cases. Escapes (true relative error >= ``min_error`` and accepted) and false alarms (correct and rejected) are
  stored as regression cases.
* **Blue** (each round): split-conformal calibration of the combined test on clean calibration cases, hardened with
  checks from ``check_library(problem)`` until every stored escape is rejected; stored false alarms pass.
* **Evaluation** (each round): false-alarm rate on a held-out clean set, stored escapes rejected, stored false
  alarms accepted, mutation score after the round, and per check the detection rate, false-alarm rate, ROC AUC over
  fault magnitude and minimal detectable magnitude (from the fault-injection matrix).

Everything is seeded; ``ArenaReport.save_cases`` writes the regression cases replayed by
``pinneapple check --arena-cases``.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..fault_injection.checks import check_library, default_checks
from ..fault_injection.matrix import DetectionMatrix, run_fault_injection
from ..fault_injection.problems import BenchmarkProblem, get_problem
from .blue import Defence, harden, score_matrix
from .cases import save_cases
from .generators import Generator, generators_for
from .red import Finding, evasion_search, false_alarm_search, mutation_escapes, mutation_score

__all__ = ["ArenaBudget", "RoundReport", "ArenaReport", "run_arena", "audit_model", "ModelAudit"]


@dataclass
class ArenaBudget:
    """How much the red team searches and how many clean cases the blue team calibrates on."""
    n_bases: int = 3                # correct cases each generator starts from
    evals: int = 160                # objective evaluations per generator and base case
    fa_seeds: int = 4               # random fields of the false-alarm search
    fa_evals: int = 80              # evaluations per random field
    n_calibration: int = 400        # clean calibration cases (split in two halves)
    n_holdout: int = 300            # held-out clean cases for the measured false-alarm rate
    n_mutation_cases: int = 20      # clean cases the fault library mutates
    n_datasets: int = 0             # datasets for the data mutants (0: model level only)

    @classmethod
    def small(cls) -> ArenaBudget:
        """CI size: seconds to a few minutes per problem."""
        return cls(n_bases=1, evals=40, fa_seeds=2, fa_evals=24, n_calibration=60, n_holdout=40,
                   n_mutation_cases=4, n_datasets=0)

    @classmethod
    def of(cls, budget: ArenaBudget | dict | str | None) -> ArenaBudget:
        if budget is None:
            return cls()
        if isinstance(budget, ArenaBudget):
            return budget
        if budget == "small":
            return cls.small()
        if isinstance(budget, dict):
            return cls(**budget)
        raise TypeError(f"budget must be an ArenaBudget, a dict or 'small', not {budget!r}")


@dataclass
class RoundReport:
    round: int
    defence_before: dict[str, Any]
    defence_after: dict[str, Any]
    mutation_before: dict[str, Any]
    mutation_after: dict[str, Any]
    escapes: list[Finding]                       # found this round (evasion + mutation)
    attempts: list[Finding]                      # best evasion attempt per generator and base (found or not)
    false_alarms: list[Finding]                  # found this round
    fa_attempts: list[Finding]                   # most suspicious correct case per random field
    heldout_false_alarm_before: float
    heldout_false_alarm_after: float
    stored_escapes_rejected: tuple[int, int]
    stored_false_alarms_accepted: tuple[int, int]
    checks_before: dict[str, Any] = field(default_factory=dict)
    checks_after: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def worst_escape(self) -> Finding | None:
        found = [f for f in self.escapes if f.found]
        return max(found, key=lambda f: f.true_error) if found else None

    @property
    def worst_false_alarm(self) -> Finding | None:
        return max(self.fa_attempts, key=lambda f: f.margin) if self.fa_attempts else None

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("escapes", "attempts", "false_alarms", "fa_attempts")}
        d.update(escapes=[f.to_dict() for f in self.escapes], attempts=[f.to_dict() for f in self.attempts],
                 false_alarms=[f.to_dict() for f in self.false_alarms],
                 fa_attempts=[f.to_dict() for f in self.fa_attempts])
        return d


@dataclass
class ArenaReport:
    problem: str
    alpha: float
    min_error: float
    rounds: list[RoundReport]
    defence: Defence
    stored: list[Finding]
    settings: dict[str, Any]
    _problem: BenchmarkProblem | None = None

    def failures(self) -> list[str]:
        """Acceptance (#707) after the last blue round."""
        r = self.rounds[-1]
        out = []
        if r.stored_escapes_rejected[0] < r.stored_escapes_rejected[1]:
            out.append(f"{r.stored_escapes_rejected[1] - r.stored_escapes_rejected[0]} stored escape(s) still pass")
        if r.stored_false_alarms_accepted[0] < r.stored_false_alarms_accepted[1]:
            out.append("a stored false alarm is still rejected")
        if r.heldout_false_alarm_after > self.alpha:
            out.append(f"held-out false-alarm rate {r.heldout_false_alarm_after:.2%} > alpha = {self.alpha:.2%}")
        return out

    @property
    def passed(self) -> bool:
        return not self.failures()

    def save_cases(self, path: str | Path) -> Path:
        """The regression cases and the hardened defence, for ``pinneapple check --arena-cases``."""
        return save_cases(path, self._problem, self.defence, self.stored,
                          {"alpha": self.alpha, "min_error": self.min_error, **self.settings})

    def check_power(self) -> dict[str, Any]:
        """The "check power" section of a validation report, after the last round."""
        r = self.rounds[-1]
        return {"problem": self.problem, "alpha": self.alpha, "defence": self.defence.describe(),
                "heldout_false_alarm": r.heldout_false_alarm_after, "mutation_score": r.mutation_after.get("score"),
                "checks": r.checks_after}

    def to_dict(self) -> dict[str, Any]:
        return {"problem": self.problem, "alpha": self.alpha, "min_error": self.min_error, "settings": self.settings,
                "defence": self.defence.to_dict(), "rounds": [r.to_dict() for r in self.rounds],
                "stored": [f.to_dict() for f in self.stored], "failures": self.failures()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=1, default=float), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        s = self.settings
        L = [f"## Arena: {self.problem}", "",
             f"alpha = {self.alpha:g} (family-wise false-alarm rate of the combined test), an escape is a case with "
             f"true relative error >= {self.min_error:.0%} that every check accepts. Seed {s.get('seed')}, search "
             f"method {s.get('method')}, budget {s.get('budget')}.", ""]
        for r in self.rounds:
            we, wf = r.worst_escape, r.worst_false_alarm
            L += [f"### Round {r.round}", "",
                  f"Defence before: {r.defence_before['kind']} ({', '.join(r.defence_before['checks'])}).", "",
                  "| | before blue | after blue |", "|---|---|---|",
                  f"| held-out false-alarm rate | {r.heldout_false_alarm_before:.2%} | "
                  f"{r.heldout_false_alarm_after:.2%} |",
                  f"| mutation score (mutants with error >= {self.min_error:.0%}) | "
                  f"{_pct(r.mutation_before)} | {_pct(r.mutation_after)} |",
                  f"| escapes found by the red team | {len(r.escapes)} | stored escapes rejected: "
                  f"{r.stored_escapes_rejected[0]}/{r.stored_escapes_rejected[1]} |",
                  f"| false alarms found by the red team | {len(r.false_alarms)} | stored false alarms accepted: "
                  f"{r.stored_false_alarms_accepted[0]}/{r.stored_false_alarms_accepted[1]} |", ""]
            L.append("Worst escape: " + (f"`{we.generator}` {_params(we.params)} on case seed {we.seed}: true error "
                                         f"{we.true_error:.1%}, largest margin {we.margin:+.3f}." if we else
                                         "none found within the budget."))
            L.append("Worst false alarm (most suspicious correct case): " + (
                f"seed {wf.seed}, variation {_params(wf.variation)}: largest margin {wf.margin:+.3f} "
                f"({'rejected' if wf.found else 'accepted'})." if wf else "none searched."))
            L += ["", "Evasion search (best accepted output per generator and base case):", "",
                  "| generator | physics-consistent | case seed | true error | escape |", "|---|---|---|---|---|"]
            for a in r.attempts:
                L.append(f"| {a.generator} | {'yes' if a.generator in _PC else 'no'} | {a.seed} | "
                         f"{a.true_error:.1%} | {'**yes**' if a.found else 'no'} |")
            L += ["", f"Defence after: {r.defence_after['kind']}.", ""]
            L += [f"- {x}" for x in r.defence_after.get("log", [])]
            L += ["", "Per check after the round (fault-injection matrix, thresholds of the defence):", "",
                  "| check | threshold | false-alarm rate | min. detectable magnitude per fault | best AUC per fault |",
                  "|---|---|---|---|---|"]
            for c, v in r.checks_after.items():
                if c.startswith("_"):
                    continue
                mm = ", ".join(f"{f} {'—' if m is None else f'{m:g}'}" for f, m in v["minimal_magnitude"].items())
                au = ", ".join(f"{f} {a:.2f}" for f, a in v["best_auc"].items())
                L.append(f"| {c} | {v['threshold']:.3g} | {v['false_alarm_rate']:.1%} | {mm} | {au} |")
            L += ["", "ROC AUC over fault magnitude (combined test: detection rate; best single-check AUC):", "",
                  "| fault | magnitude | true error | detection | best AUC |", "|---|---|---|---|---|"]
            for row in r.checks_after.get("_roc", []):
                L.append(f"| {row['fault']} | {row['magnitude']:g} | {row['true_error']:.3g} | {row['detection']:.0%} "
                         f"| {row['best_auc']:.2f} |")
            L.append("")
        L += ["### Catalogue of stored cases", "", "| kind | generator | case seed | true error | params |",
              "|---|---|---|---|---|"]
        for f in self.stored:
            L.append(f"| {f.kind} | {f.generator or 'correct case'} | {f.seed} | "
                     f"{'—' if f.kind != 'escape' else f'{f.true_error:.1%}'} | "
                     f"{_params(f.params or f.variation)} |")
        fails = self.failures()
        L += ["", "**Acceptance:** " + ("every stored escape rejected, every stored false alarm accepted, held-out "
                                        f"false-alarm rate <= {self.alpha:g}." if not fails else "; ".join(fails)), ""]
        return "\n".join(L)


_PC = {"coefficient_field", "source_field", "viscosity", "hard_ic_evolution", "time_warp"}


def _pct(m: dict[str, Any]) -> str:
    sc = m.get("score")
    return "—" if sc is None or (isinstance(sc, float) and math.isnan(sc)) else \
        f"{sc:.1%} ({m['killed']}/{m['mutants']})"


def _params(p: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.3g}" for k, v in p.items()) + "}"


def _check_table(matrix: DetectionMatrix, defence: Defence) -> dict[str, Any]:
    view = matrix.with_tolerances(defence.thresholds)
    out: dict[str, Any] = {}
    for c in view.checks:
        out[c] = {"threshold": defence.thresholds[c], "false_alarm_rate": view.false_alarm_rate(c),
                  "minimal_magnitude": {f: view.minimal_magnitude(f, c) for f in view.faults},
                  "best_auc": {f: max(view.auc(f, m, c) for m in view.faults[f].magnitudes) for f in view.faults},
                  "detection": {f"{f}@{m:g}": view.detection_rate(f, m, c) for (f, m) in view.scores}}
    out["_roc"] = [{"fault": f, "magnitude": m, "true_error": view.true_error(f, m),
                    "detection": view.detection_rate(f, m),
                    "best_auc": max(view.auc(f, m, c) for c in view.checks)} for (f, m) in view.scores]
    return out


def _fa_rate(defence: Defence, values: dict[str, np.ndarray]) -> float:
    n = len(next(iter(values.values())))
    return float(np.mean([defence.rejects_values({c: values[c][i] for c in defence.names}) for i in range(n)]))


def run_arena(problem: str | BenchmarkProblem = "darcy_2d", checks: Sequence | None = None, *,
              budget: ArenaBudget | dict | str | None = None, rounds: int = 1, alpha: float = 0.01,
              safety: float = 0.5, min_error: float = 0.1, method: str = "de", seed: int = 0,
              generators: Sequence[Generator] | None = None, cases_path: str | Path | None = None,
              verbose: bool = False, **problem_options: Any) -> ArenaReport:
    """Run ``rounds`` rounds of red team then blue team on ``problem`` (see the module docstring)."""
    t_start = time.time()
    B = ArenaBudget.of(budget)
    p = get_problem(problem, **problem_options)
    lib = check_library(p)
    checks = list(checks) if checks is not None else default_checks(p)
    gens = list(generators) if generators is not None else generators_for(p)
    log = (lambda *a: print(f"[arena {p.name} {time.time() - t_start:6.0f}s]", *a, flush=True)) if verbose \
        else (lambda *a: None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        allchecks = {**lib, **{c.name: c for c in checks}}
        log("fault-injection matrix (mutation operators)")
        fi = run_fault_injection(p, checks=list(allchecks.values()), n_cases=B.n_mutation_cases,
                                 n_datasets=B.n_datasets, include_data=B.n_datasets > 0, seed=seed)
        log("clean calibration and held-out sets")
        cal = p.sample_varied(B.n_calibration, seed=seed + 101)
        hold = p.sample_varied(B.n_holdout, seed=seed + 202)
        cal_v = score_matrix(list(allchecks.values()), cal)
        hold_v = score_matrix(list(allchecks.values()), hold)
        bases = p.sample(B.n_bases, seed=seed + 303)
        fa_seeds = [int(s) for s in np.random.default_rng(seed + 404).integers(0, 2 ** 31 - 1, B.fa_seeds)]

        defence = Defence.from_checks(checks)
        stored: list[Finding] = []
        reports = []
        for r in range(rounds):
            t0 = time.time()
            before = defence
            mut_before = mutation_score(fi.model, before.thresholds, min_error)
            log(f"round {r}: red team vs {before.kind} ({len(before.checks)} checks)")
            esc = [f for f in mutation_escapes(fi.model, before.thresholds, seed=seed, min_error=min_error)]
            attempts = []
            for gi, g in enumerate(gens):
                for bi, base in enumerate(bases):
                    a = evasion_search(p, before, g, base, budget=B.evals, method=method,
                                       seed=seed + 1000 * r + 10 * gi + bi, min_error=min_error)
                    a.round = r
                    attempts.append(a)
                    log(f"  evasion {g.name} base {bi}: error {a.true_error:.3f} {'ESCAPE' if a.found else ''}")
            esc += [a for a in attempts if a.found]
            for f in esc:
                f.round = r
            fa_att = false_alarm_search(p, before, seeds=fa_seeds, budget=B.fa_evals, method=method,
                                        search_seed=seed + 1000 * r)
            fas = [f for f in fa_att if f.found]
            for f in fa_att:
                f.round = r
            log(f"  {len(esc)} escape(s), {len(fas)} false alarm(s); worst margin of a correct case "
                f"{fa_att[0].margin if fa_att else float('nan'):+.3f}")
            stored += esc + fas
            log(f"round {r}: blue team")
            esc_cases = [f.case(p)[0] for f in stored if f.kind == "escape"]
            fa_cases = [f.case(p)[0] for f in stored if f.kind == "false_alarm"]
            defence = harden(before.checks, lib, cal, escapes=esc_cases, false_alarms=fa_cases, alpha=alpha,
                             safety=safety, seed=seed + r, clean_values=cal_v)
            for x in defence.log:
                log("  " + x)
            rej = sum(defence.rejects(c) for c in esc_cases)
            acc = sum(not defence.rejects(c) for c in fa_cases)
            rep = RoundReport(
                r, before.to_dict(), defence.to_dict(), mut_before, mutation_score(fi.model, defence.thresholds,
                                                                                   min_error),
                esc, attempts, fas, fa_att, _fa_rate(before, hold_v), _fa_rate(defence, hold_v),
                (int(rej), len(esc_cases)), (int(acc), len(fa_cases)), _check_table(fi.model, before),
                _check_table(fi.model, defence), time.time() - t0)
            reports.append(rep)
            log(f"round {r} done: held-out false alarms {rep.heldout_false_alarm_before:.2%} -> "
                f"{rep.heldout_false_alarm_after:.2%}, stored escapes rejected {rej}/{len(esc_cases)}")
    out = ArenaReport(p.name, alpha, min_error, reports, defence, stored,
                      {"seed": seed, "method": method, "rounds": rounds, "safety": safety, "budget": asdict(B),
                       "seconds": round(time.time() - t_start, 1)}, p)
    if cases_path is not None:
        out.save_cases(cases_path)
    return out


@dataclass
class ModelAudit:
    """A model's own outputs judged by a defence against the reference solver."""
    problem: str
    defence: str
    true_error: np.ndarray
    rejected: np.ndarray
    seeds: list[int]
    min_error: float

    @property
    def escapes(self) -> list[int]:
        """Seeds of the cases with true error >= ``min_error`` that the defence accepts."""
        return [s for s, e, r in zip(self.seeds, self.true_error, self.rejected, strict=True)
                if e >= self.min_error and not r]

    @property
    def false_alarms(self) -> list[int]:
        """Seeds of the cases with true error < ``min_error`` / 10 that the defence rejects."""
        return [s for s, e, r in zip(self.seeds, self.true_error, self.rejected, strict=True)
                if e < self.min_error / 10 and r]

    def to_markdown(self) -> str:
        e, r = self.true_error, self.rejected
        bad = e >= self.min_error
        return (f"Model audit on {self.problem} ({len(e)} cases, {self.defence}): median true error "
                f"{np.median(e):.2%}, rejected {r.mean():.0%}; of the {int(bad.sum())} cases with error >= "
                f"{self.min_error:.0%}, {int((bad & r).sum())} rejected and {len(self.escapes)} escaped; "
                f"{len(self.false_alarms)} accurate case(s) rejected.")


def audit_model(problem: str | BenchmarkProblem, predict, defence: Defence | str | Path | None = None, *,
                n_cases: int = 50, seed: int = 0, min_error: float = 0.1, varied: bool = True,
                **problem_options: Any) -> ModelAudit:
    """Run an external model (an FNO, a MeshGraphNet, a PINN...) through a defence.

    ``predict(inputs: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray`` returns the model's field on
    the problem's grid for one case (Darcy: ``inputs["a"]`` (n, n) -> u (n, n); Burgers: ``inputs["u0"]`` (n,) ->
    u (n_t, n)). ``defence``: a :class:`Defence`, a cases file, or None for the hardened defence shipped with the
    library for that problem. Each output is compared with the reference solver (true error) and judged by the
    defence: an escape is a wrong output the defence accepts.
    """
    from .cases import builtin_cases, load_defence
    if defence is None or isinstance(defence, (str, Path)):
        name = get_problem(problem, **problem_options).name
        path = defence or next((f for f in builtin_cases() if f.stem == name), None)
        if path is None:
            raise FileNotFoundError(f"no stored defence for {name}; run the arena first")
        p, defence = load_defence(path)
    else:
        p = get_problem(problem, **problem_options)
    cases = p.sample_varied(n_cases, seed=seed) if varied else p.sample(n_cases, seed=seed)
    errs, rej = [], []
    for c in cases:
        out = c.with_field(np.asarray(predict(c.inputs, c.params), dtype=np.float64).reshape(c.field.shape))
        errs.append(p.true_error(out, c))
        rej.append(defence.rejects(out))
    return ModelAudit(p.name, defence.kind, np.array(errs), np.array(rej, bool), [c.seed for c in cases], min_error)
