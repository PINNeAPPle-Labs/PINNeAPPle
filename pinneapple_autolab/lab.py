"""The autonomous physics lab: plan -> build -> simulate -> learn -> optimise -> verify -> deliver, on any simulator.

    lab = AutonomousLab(mission, simulator, "runs/aero")
    lab.run()                                   # the whole scientific cycle, deterministic
    lab.summary()                               # what was simulated, learned, chosen and verified, with the numbers

Every step is also a method an agent can call on its own (``plan``, ``simulate``, ``learn``, ``optimize``,
``assess``, ``choose_next``, ``verify``); each one writes an event to the ledger, and only the ledger's runs are
evidence. The lab enforces the rules a careless optimiser breaks:

* no design is simulated before its geometry and boundary conditions pass the simulator's checks;
* the operating conditions of the mission are fixed (the simulator owns them; nothing here can change them);
* no simulation beyond the budget (``BudgetExhaustedError``), with ``reserve_verify`` runs kept for the end;
* a relative limit ("keep the baseline's downforce") is only applied once the baseline was simulated;
* the error of the surrogate is measured on designs it had not seen (cross-validation, and each acquisition run's
  prediction recorded *before* the run);
* the recommendation is a simulated design, never a prediction; designs only predicted are labelled as such.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.methods import run as run_optimizer
from pinneapple_design.design_optimizer.study import (
    DesignProblem,
    DesignSpace,
    FunctionEvaluator,
    Objective,
)

from .domain import ValidatedDomain
from .ledger import BudgetExhaustedError, Ledger, RunRecord
from .mission import Mission
from .simulator import Simulator
from .surrogates import FieldSurrogate, ModelBoard, compare_models, field_errors
from .trust import (
    CandidateAssessment,
    NextExperiment,
    assess_candidates,
    choose_next,
    verification_checks,
)

__all__ = ["AutonomousLab", "showcase", "SHOWCASE"]

SHOWCASE = {"A": "uncertain", "B": "favourable", "C": "out_of_domain"}


def showcase(decisions: list[dict[str, Any]], baseline: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Three simulated designs the lab judged differently before simulating them, by a fixed rule:

    * A, "uncertain": the one the reference contradicted most (rejected by the reference first, then the largest
      relative error of any output): what an unsettled prediction would have cost if it had been believed;
    * B, "favourable": the first one;
    * C, "out_of_domain": the first one (the model's optimum without evidence).

    Each with its prediction and std, the reasons and the trust decision before the run, what the reference said."""
    out: dict[str, dict[str, Any]] = {}
    for tag, lab in SHOWCASE.items():
        ds = [d for d in decisions if d["label_before"] == lab and d.get("run")]
        if not ds:
            continue
        if tag == "A":
            d = max(ds, key=lambda d: (d["label_after"] != "verified", max(d["rel_error"].values(), default=0)))
        else:
            d = ds[0]
        out[tag] = {"tag": tag, "label": lab, "run": d["run"], "params": d["params"], "predicted": d["predicted"],
                    "std": d["std"], "reference": d["reference"], "rel_error": d["rel_error"],
                    "reasons": d.get("reasons_before", []), "decision": d.get("trust_before"), "rule": d["rule"],
                    "after": d["label_after"], "field_error": d.get("field_error"), "baseline": baseline,
                    "action": f"simulated ({d['run']}): {d['label_after']}"}
    return out


class AutonomousLab:
    """``critical_region``: name of a simulator region whose field error decides the trust in a constraint
    (e.g. the diffuser for downforce); ``field_tol``: the tolerated predicted std of that field there.
    ``probe(params, **options)`` runs another fidelity (e.g. a finer mesh) that never joins the training data."""

    def __init__(self, mission: Mission, simulator: Simulator, folder: str, *,
                 families: Sequence[str] = ("gp", "mlp_ensemble", "quadratic"), field_family: str = "gp",
                 seed: int = 0, nsga_budget: int = 3000, n_pool: int = 2000, critical_region: str | None = None,
                 field_tol: float | None = None,
                 listeners: list[Callable[[dict], None]] | None = None, cv_folds: int = 5):
        self.mission, self.sim, self.folder = mission, simulator, folder
        self.space = DesignSpace(mission.variables)
        self.names = self.space.names
        self.families, self.field_family, self.seed = tuple(families), field_family, seed
        self.nsga_budget, self.n_pool, self.cv_folds = nsga_budget, n_pool, cv_folds
        self.critical_region, self.field_tol = critical_region, field_tol
        self.ledger = Ledger(folder, mission.budget.max_runs, mission.budget.max_wall_s, listeners)
        self.fields: dict[str, dict[str, np.ndarray]] = {}
        self.board: ModelBoard | None = None
        self.field_model: FieldSurrogate | None = None
        self.field_cv: dict[str, Any] = {}
        self.domain: ValidatedDomain | None = None
        self.candidates: list[CandidateAssessment] = []
        self.cand_X: np.ndarray = np.zeros((0, len(self.names)))
        self.pool: dict[str, np.ndarray] = {}
        self.history: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.verified: list[CandidateAssessment] = []
        self.plan_: dict[str, Any] = {}

    # ================================================================== plan
    def plan(self, n_doe: int | None = None) -> dict[str, Any]:
        """Objectives, limits, hypotheses and how the budget is split: 1 baseline, ``n_doe`` space-filling runs,
        acquisition runs chosen one at a time, ``reserve_verify`` final verifications."""
        b = self.mission.budget
        d = len(self.names)
        n_doe = n_doe if n_doe is not None else min(max(2 * d + 2, 8), b.max_runs - b.reserve_verify - 2)
        n_active = b.max_runs - 1 - n_doe - b.reserve_verify
        if n_active < 0:
            raise ValueError(f"budget {b.max_runs} too small for 1 baseline + {n_doe} DOE + {b.reserve_verify} "
                             "verifications")
        self.plan_ = {"mission": self.mission.to_dict(), "simulator": self.sim.info(),
                      "allocation": {"baseline": 1, "doe": n_doe, "acquisition": n_active,
                                     "verification": b.reserve_verify, "total": b.max_runs},
                      "objectives": [f"{o.sense} {o.label or o.name} [{o.unit}]" for o in self.mission.objectives],
                      "limits": [c.describe() for c in self.mission.constraints],
                      "hypotheses": list(self.mission.hypotheses),
                      "acceptance": dict(self.mission.acceptance), "conditions": dict(self.mission.conditions),
                      "surrogates": list(self.families), "critical_region": self.critical_region}
        self.ledger.event("plan", f"{b.max_runs} simulations: 1 baseline, {n_doe} space-filling, {n_active} chosen "
                          f"one at a time, {b.reserve_verify} final verifications", plan=self.plan_)
        return self.plan_

    # ================================================================== build + simulate
    def check_design(self, params: dict[str, float]) -> list[str]:
        issues = []
        for v in self.mission.variables:
            x = params.get(v.name)
            if x is None:
                issues.append(f"{v.name} missing")
            elif not v.lower - 1e-9 <= x <= v.upper + 1e-9:
                issues.append(f"{v.name} = {x:g} outside the allowed range [{v.lower:g}, {v.upper:g}]")
        return issues + list(self.sim.check(params))

    def simulate(self, params: dict[str, float], purpose: str, reason: str = "",
                 predicted: dict[str, Any] | None = None, **options: Any) -> RunRecord | None:
        """Run the reference simulator on one design (charged to the budget). Returns None, without charging,
        when the design fails its checks; raises BudgetExhaustedError when nothing is left."""
        params = {k: float(params[k]) for k in self.names}
        issues = self.check_design(params)
        if issues:
            self.ledger.event("build", "design refused before simulation: " + "; ".join(issues), params=params,
                              issues=issues)
            return None
        self.ledger.charge()
        rid = self.ledger.next_id(purpose)
        self.ledger.event("simulate", f"{rid}: running {self.sim.name}" + (f" ({reason})" if reason else ""),
                          params=params, purpose=purpose, running=True)
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.perf_counter()
        try:
            out = self.sim.run(params, os.path.join(self.folder, "cases", rid), **options)
        except Exception as e:  # noqa: BLE001 - a crashed solver is a failed run, recorded as such
            from .simulator import SimOutcome
            out = SimOutcome({}, False, f"{type(e).__name__}: {e}", time.perf_counter() - t0)
        rec = RunRecord(rid, purpose, params, bool(out.ok), dict(out.outputs) if out.ok else {}, out.message,
                        float(out.cost_s or (time.perf_counter() - t0)), started, reason, dict(predicted or {}),
                        dict(out.meta))
        if out.ok and out.fields:
            os.makedirs(os.path.join(self.folder, "fields"), exist_ok=True)
            rec.fields_path = os.path.join(self.folder, "fields", f"{rid}.npz")
            np.savez_compressed(rec.fields_path, **{k: np.asarray(v, np.float32) for k, v in out.fields.items()})
            self.fields[rid] = {k: np.asarray(v, float) for k, v in out.fields.items()}
        self.ledger.add(rec)
        self.ledger.save()
        return rec

    def baseline(self) -> RunRecord:
        rec = self.simulate(dict(self.mission.baseline), "baseline", "measure the starting design")
        if rec is None or not rec.ok:
            raise RuntimeError("the baseline could not be simulated: "
                               + ("refused by its checks" if rec is None else rec.message))
        cons = self.mission.resolve(rec.outputs)
        self.ledger.event("plan", "limits fixed by the measured baseline: " + "; ".join(c.describe() for c in cons),
                          constraints=[c.describe() for c in cons], baseline=rec.outputs)
        return rec

    def doe(self, n: int) -> list[RunRecord]:
        X = self.space.sample(n, method="lhs", seed=self.seed)
        out = []
        for k, p in enumerate(self.space.as_dicts(X)):
            r = self.simulate(p, "doe", f"space-filling design {k + 1}/{n}")
            if r is not None:
                out.append(r)
        return out

    # ================================================================== learn
    def learn(self) -> ModelBoard:
        """Cross-validate every surrogate family on every predicted output, keep the best per output; fit the field
        surrogate (with its k-fold error per region) when the simulator returns fields; rebuild the domain."""
        outs = self.mission.outputs()
        X, Y = self.ledger.arrays(self.names, outs)
        if len(X) < 4:
            raise RuntimeError(f"{len(X)} successful simulations: too few to learn from")
        U = self.space.to_unit(X)
        self.ledger.event("learn", f"training {len(self.families)} surrogate families on {len(X)} simulations",
                          training=True)
        self.board = compare_models(U, Y, self.families, k=self.cv_folds, seed=self.seed)
        self.domain = ValidatedDomain(U, self.names, self.space.bounds)
        msg = "; ".join(f"{o}: {self.board.chosen[o]} (held-out rel. error "
                        f"{100 * self.board.table[o][self.board.chosen[o]]['rel_mae']:.1f} %)" for o in outs)
        self.ledger.event("learn", msg, board=self.board.to_dict())
        self._learn_fields()
        return self.board

    def _field_snapshots(self, name: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        ok = [r for r in self.ledger.ok_runs() if r.id in self.fields and name in self.fields[r.id]]
        if len(ok) < 4:
            return None
        X = np.array([[r.params[k] for k in self.names] for r in ok])
        return self.space.to_unit(X), np.stack([self.fields[r.id][name] for r in ok]), [r.id for r in ok]

    def _learn_fields(self) -> None:
        if not self.sim.field_names:
            return
        name = self.sim.field_names[0]
        snap = self._field_snapshots(name)
        if snap is None:
            return
        U, S, ids = snap
        n = len(U)
        k = max(2, min(self.cv_folds, n))
        perm = np.random.default_rng(self.seed).permutation(n)
        P, SD = np.empty_like(S), np.empty_like(S)
        for f in np.array_split(perm, k):
            tr = np.setdiff1d(perm, f)
            fm = FieldSurrogate(self.field_family).fit(U[tr], S[tr])
            P[f], SD[f] = fm.predict(U[f])
        regions = {kk: np.asarray(v) for kk, v in self.sim.regions.items()}
        per = [field_errors(P[i], S[i], regions, SD[i]) for i in range(n)]
        agg = {reg: {m: float(np.mean([p[reg][m] for p in per])) for m in per[0][reg]} for reg in per[0]}
        self.field_model = FieldSurrogate(self.field_family).fit(U, S)
        self.field_cv = {"field": name, "family": self.field_family, "n": n, "k": k, "per_region": agg,
                         "n_modes": self.field_model.n_modes, "energy": self.field_model.energy_kept,
                         "per_run": {ids[i]: {reg: per[i][reg]["rmse"] for reg in per[i]} for i in range(n)}}
        self._field_cv_pred = {ids[i]: (P[i], SD[i]) for i in range(n)}
        self.ledger.event("learn", f"field surrogate ({name}, POD {self.field_model.n_modes} modes + "
                          f"{self.field_family}): held-out RMSE " + ", ".join(f"{r} {v['rmse']:.3f}"
                                                                                for r, v in agg.items()),
                          field_cv={kk: v for kk, v in self.field_cv.items() if kk != "per_run"})

    def predict(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.board is None:
            raise RuntimeError("call learn() first")
        return self.board.predict(self.space.to_unit(X))

    def _critical_std(self, X: np.ndarray) -> np.ndarray | None:
        if self.field_model is None or self.critical_region is None:
            return None
        mask = np.asarray(self.sim.regions[self.critical_region])
        fm = self.field_model
        U = self.space.to_unit(X)
        Cs = np.stack([m.predict(U)[1] for m in fm.models], 1)
        sd = np.sqrt((Cs ** 2) @ (fm.modes[:, mask] ** 2))
        return np.sqrt(np.mean(sd ** 2, 1))

    # ================================================================== optimise + assess
    def pareto_axes(self) -> list[Objective]:
        """The mission's objectives plus every one-sided constraint as an objective in its good direction (the
        trade-off a designer wants to see, e.g. drag against downforce)."""
        axes = list(self.mission.objectives)
        for c in self.mission.constraints_absolute():
            if (c.lower is None) != (c.upper is None) and all(o.name != c.name for o in axes):
                axes.append(Objective(c.name, "max" if c.lower is not None else "min", c.unit, c.label))
        return axes

    def optimize(self) -> dict[str, Any]:
        """NSGA-II on the surrogate means over the Pareto axes, plus a Latin-hypercube pool; every evaluated
        design becomes a candidate (thousands), assessed in :meth:`assess`."""
        if self.board is None:
            self.learn()
        axes = self.pareto_axes()
        outs = [a.name for a in axes]

        def fn(X):
            p = self.predict(X)
            return {k: p[k][0] for k in outs}
        prob = DesignProblem("surrogate", self.space, axes, [], FunctionEvaluator(fn, outs, name="surrogate",
                                                                                   fidelity="surrogate"))
        t0 = time.perf_counter()
        res = run_optimizer(prob, "nsga2", budget=self.nsga_budget, seed=self.seed + len(self.ledger.runs))
        Xn = res.evaluations.X if res.evaluations is not None else np.zeros((0, len(self.names)))
        Xp = self.space.sample(self.n_pool, method="lhs", seed=self.seed + 1000 + len(self.ledger.runs))
        self.cand_X = np.vstack([Xn, Xp])
        wall = time.perf_counter() - t0
        self.ledger.event("optimize", f"NSGA-II explored {len(Xn)} designs on the surrogate (+{len(Xp)} pool) in "
                          f"{wall:.1f} s", n_candidates=len(self.cand_X), wall_s=wall, status=res.status)
        return {"n": len(self.cand_X), "wall_s": wall, "status": res.status}

    def best_simulated(self) -> RunRecord | None:
        """The best simulated design that meets every (resolved) constraint."""
        if not self.mission.resolved:
            return None
        cons = self.mission.constraints_absolute()
        o = self.mission.objectives[0]
        ok = [r for r in self.ledger.ok_runs()
              if all(c.violation(r.outputs.get(c.name, np.nan)) <= 0 for c in cons)]
        if not ok:
            return None
        return min(ok, key=lambda r: (1 if o.sense == "min" else -1) * r.outputs[o.name])

    def assess(self) -> list[CandidateAssessment]:
        if not len(self.cand_X):
            self.optimize()
        X = self.cand_X
        pred = self.predict(X)
        best = self.best_simulated()
        o = self.mission.objectives[0]
        model_err = {k: self.board.table[k][self.board.chosen[k]]["rel_mae"] for k in self.board.chosen}
        dv = self.domain.assess(self.space.to_unit(X))
        self.candidates = assess_candidates(X, self.names, pred, self.mission.objectives,
                                            self.mission.constraints_absolute(), dv,
                                            None if best is None else best.outputs[o.name], self.mission.acceptance,
                                            model_err, self._critical_std(X), self.field_tol)
        self.pool = {"X": X, **{f"{k}_mean": v[0] for k, v in pred.items()},
                     **{f"{k}_std": v[1] for k, v in pred.items()},
                     "label": np.array([c.label for c in self.candidates]),
                     "p_feasible": np.array([c.p_feasible for c in self.candidates])}
        counts = {lab: int(sum(c.label == lab for c in self.candidates)) for lab in
                  ("favourable", "uncertain", "out_of_domain", "unlikely")}
        self.ledger.event("assess", "candidates: " + ", ".join(f"{v} {k}" for k, v in counts.items()), counts=counts)
        self._write_live()
        return self.candidates

    def choose_next(self) -> NextExperiment:
        nxt = choose_next(self.candidates, self.mission.objectives[0])
        self.ledger.event("decide", f"next simulation: {nxt.rule}: {nxt.reason}", decision=nxt.to_dict())
        return nxt

    # ================================================================== active learning
    def step(self) -> dict[str, Any] | None:
        """One cycle: learn, optimise, assess, choose, simulate; records the held-out error of the prediction."""
        self.learn()
        self.optimize()
        self.assess()
        nxt = self.choose_next()
        if nxt.candidate is None:
            return None
        c = nxt.candidate
        rec = self.simulate(c.params, "acquisition", f"{nxt.rule}", predicted={"mean": c.predicted, "std": c.std,
                                                                                 "label": c.label})
        entry = self._record_outcome(c, rec, nxt)
        return entry

    def _record_outcome(self, c: CandidateAssessment, rec: RunRecord | None, nxt: NextExperiment | None,
                        verification: bool = False) -> dict[str, Any]:
        label_before, trust_before, reasons_before = c.label, c.report.decisao, list(c.reasons)
        ok = rec is not None and rec.ok
        verification_checks(c, rec.outputs if ok else {}, ok, self.mission.acceptance,
                            self.mission.constraints_absolute(), "" if ok else (rec.message if rec else "refused"))
        entry = {"run": None if rec is None else rec.id, "label_before": label_before, "label_after": c.label,
                 "rule": None if nxt is None else nxt.rule, "reason": None if nxt is None else nxt.reason,
                 "params": c.params, "predicted": c.predicted, "std": c.std, "reference": c.reference,
                 "rel_error": c.rel_error, "p_feasible": c.p_feasible, "domain_ratio": c.domain.ratio,
                 "trust_before": trust_before, "reasons_before": reasons_before, "trust_after": c.report.decisao,
                 "verification": verification, "critical_field_std": c.critical_field_std,
                 "n_train": self.board.n_train if self.board else 0,
                 "cv": {k: self.board.table[k][self.board.chosen[k]]["rel_mae"] for k in self.board.chosen}
                 if self.board else {}}
        if self.field_model is not None and ok and rec.id in self.fields and self.sim.field_names:
            name = self.sim.field_names[0]
            u = self.space.to_unit(np.array([[c.params[k] for k in self.names]]))
            fp, fs = self.field_model.predict(u)
            np.savez_compressed(os.path.join(self.folder, "fields", f"{rec.id}_pred.npz"),
                                **{name: fp[0].astype(np.float32), f"{name}_std": fs[0].astype(np.float32)})
            entry["field_error"] = field_errors(fp[0], self.fields[rec.id][name],
                                                {k: np.asarray(v) for k, v in self.sim.regions.items()}, fs[0])
        self.history.append(entry)
        self.decisions.append(entry)
        msg = (f"{entry['run']}: predicted {self._fmt(c.predicted)} -> simulated {self._fmt(c.reference)}; "
               f"{label_before} -> {c.label}") if ok else f"run failed or refused; {label_before} -> {c.label}"
        self.ledger.event("verify" if verification else "update", msg, outcome=entry)
        self._write_live()
        return entry

    def _write_live(self) -> None:
        """``snapshot_live.json`` in the session folder: what a live dashboard shows between events."""
        import json
        try:
            path = os.path.join(self.folder, "snapshot_live.json")
            with open(path + ".tmp", "w") as f:
                json.dump(self.snapshot(max_candidates=1500), f, default=float, separators=(",", ":"))
            os.replace(path + ".tmp", path)
        except Exception as e:  # noqa: BLE001 - a dashboard file must never stop a campaign
            self.ledger.event("deliver", f"live snapshot not written: {type(e).__name__}: {e}")

    def _fmt(self, d: dict[str, float]) -> str:
        return ", ".join(f"{k} {v:.4g}" for k, v in d.items() if k in self.mission.outputs())

    # ================================================================== verify + run
    def verify(self, n: int | None = None) -> list[CandidateAssessment]:
        """Simulate the ``n`` best favourable candidates (by predicted objective, distinct from simulated designs);
        returns their updated assessments."""
        n = self.mission.budget.reserve_verify if n is None else n
        self.learn()
        self.optimize()
        self.assess()
        o = self.mission.objectives[0]
        sgn = 1 if o.sense == "min" else -1
        fav = sorted([c for c in self.candidates if c.label == "favourable"], key=lambda c: sgn * c.predicted[o.name])
        if not fav:
            fav = sorted([c for c in self.candidates if c.label == "uncertain"], key=lambda c: -c.cei)
        chosen: list[CandidateAssessment] = []
        sim_U = self.space.to_unit(self.ledger.arrays(self.names, [])[0])
        for c in fav:
            u = self.space.to_unit(np.array([[c.params[k] for k in self.names]]))[0]
            if all(np.linalg.norm(u - self.space.to_unit(np.array([[x.params[k] for k in self.names]]))[0]) > 0.05
                   for x in chosen) and (not len(sim_U) or np.min(np.linalg.norm(sim_U - u, axis=1)) > 0.02):
                chosen.append(c)
            if len(chosen) >= n:
                break
        for c in chosen:
            try:
                rec = self.simulate(c.params, "verification", "final verification of a favourable design",
                                    predicted={"mean": c.predicted, "std": c.std, "label": c.label})
            except BudgetExhaustedError:
                break
            self._record_outcome(c, rec, None, verification=True)
            self.verified.append(c)
        return self.verified

    def probe(self, params: dict[str, float], reason: str, **options: Any) -> RunRecord | None:
        """A reference run at another fidelity (``options`` go to ``simulator.run``, e.g. ``resolution="fine"``):
        charged to the budget, kept in the ledger, never used for training or for the recommendation."""
        return self.simulate(params, "probe", reason, **options)

    def run(self, n_doe: int | None = None) -> dict[str, Any]:
        """The whole cycle with the deterministic planner."""
        plan = self.plan(n_doe)
        self.baseline()
        self.doe(plan["allocation"]["doe"])
        while self.ledger.remaining() > self.mission.budget.reserve_verify:
            try:
                if self.step() is None:
                    break
            except BudgetExhaustedError:
                break
        if self.ledger.remaining() > 0:
            self.verify(min(self.mission.budget.reserve_verify, self.ledger.remaining()))
        s = self.summary()
        self.ledger.event("deliver", s["conclusion"], summary=s)
        self.ledger.save()
        return s

    # ================================================================== deliver
    def snapshot(self, max_candidates: int = 3000) -> dict[str, Any]:
        """The session's state for figures and dashboards: runs, candidates (subsampled) with their predictions
        and labels, the surrogate Pareto front over the Pareto axes among candidates whose other limits are met
        on the mean, the model board with its cross-validated predictions, the decisions."""
        from .report import pareto_front
        snap: dict[str, Any] = {"title": self.mission.title, "runs": [r.to_dict() for r in self.ledger.runs],
                                "decisions": self.decisions, "plan": self.plan_,
                                "budget": {"used": self.ledger.used_runs, "max": self.ledger.max_runs,
                                           "wall_s": self.ledger.used_wall_s}}
        best = self.best_simulated()
        snap["best"] = None if best is None else best.to_dict()
        if self.mission.resolved:
            cons = self.mission.constraints_absolute()
            snap["limits"] = {c.name: c.lower for c in cons if c.lower is not None and c.upper is None}
            snap["limits_all"] = {c.name: (c.lower, c.upper) for c in cons}     # None: no bound (valid JSON)
        if self.pool:
            n = len(self.pool["X"])
            keep = np.arange(n) if n <= max_candidates else np.sort(
                np.random.default_rng(0).choice(n, max_candidates, replace=False))
            cand = {k: np.asarray(v)[keep] for k, v in self.pool.items()}
            snap["candidates"] = {k: (np.round(v, 5).tolist() if v.dtype.kind == "f" else v.tolist())
                                  for k, v in cand.items()}
            snap["candidates"]["names"] = self.names
            axes = self.pareto_axes()
            ok = np.ones(len(keep), bool)
            for c in self.mission.constraints_absolute():
                if any(a.name == c.name for a in axes):
                    continue
                m = cand[f"{c.name}_mean"]
                ok &= (m >= (c.lower if c.lower is not None else -np.inf)) & \
                      (m <= (c.upper if c.upper is not None else np.inf))
            if len(axes) >= 2 and ok.any():
                F = np.stack([cand[f"{a.name}_mean"] * (1 if a.sense == "min" else -1) for a in axes[:2]], 1)
                sub = np.flatnonzero(ok)
                snap["front_idx"] = sub[pareto_front(F[sub])].tolist()
        if self.board is not None:
            snap["board"] = self.board.to_dict()
            snap["cv_pred"] = {o: {"y": self.board.cv_y[o].tolist(),
                                   "families": {f: v.tolist() for f, v in self.board.cv_pred[o].items()}}
                               for o in self.board.table}
        snap["field_cv"] = {k: v for k, v in self.field_cv.items() if k != "per_run"}
        return snap

    def showcase(self) -> dict[str, dict[str, Any]]:
        """Three designs the lab judged differently; see :func:`showcase`. A label never simulated falls back to
        the best current candidate with that label, marked "not simulated"."""
        base = next((r.outputs for r in self.ledger.runs if r.purpose == "baseline" and r.ok), {})
        out = showcase(self.decisions, base)
        for tag, lab in SHOWCASE.items():
            pool = sorted([c for c in self.candidates if c.label == lab], key=lambda c: -c.cei)
            if tag not in out and pool:
                c = pool[0]
                out[tag] = {"tag": tag, "label": lab, "run": None, "params": c.params, "predicted": c.predicted,
                            "std": c.std, "reference": {}, "rel_error": {}, "reasons": c.reasons,
                            "decision": c.report.decisao, "rule": None, "after": "not simulated", "baseline": base,
                            "action": "not simulated (budget spent)"}
        return out

    def summary(self) -> dict[str, Any]:
        o = self.mission.objectives[0]
        base = next((r for r in self.ledger.runs if r.purpose == "baseline" and r.ok), None)
        best = self.best_simulated()
        held = [h for h in self.history if h["rel_error"]]
        out: dict[str, Any] = {"mission": self.mission.to_dict(), "plan": self.plan_,
                               "budget": {"used_runs": self.ledger.used_runs, "max_runs": self.ledger.max_runs,
                                          "solver_wall_s": round(self.ledger.used_wall_s, 1),
                                          "failed_runs": sum(not r.ok for r in self.ledger.runs)},
                               "baseline": None if base is None else {"id": base.id, "params": base.params,
                                                                      "outputs": base.outputs},
                               "best": None if best is None else {"id": best.id, "purpose": best.purpose,
                                                                  "params": best.params, "outputs": best.outputs},
                               "board": None if self.board is None else self.board.to_dict(),
                               "field_cv": {k: v for k, v in self.field_cv.items() if k != "per_run"},
                               "decisions": self.decisions,
                               "held_out": [{"run": h["run"], "n_train": h["n_train"], **{
                                   f"rel_error_{k}": v for k, v in h["rel_error"].items()}} for h in held],
                               "verified": [c.to_dict() for c in self.verified],
                               "showcase": self.showcase(),
                               "environment": self.ledger.env}
        if base is not None and best is not None:
            b, w = base.outputs[o.name], best.outputs[o.name]
            out["improvement"] = {o.name: {"baseline": b, "best": w, "change": w - b,
                                           "change_pct": 100 * (w - b) / abs(b)}}
            for c in self.mission.constraints_absolute():
                out["improvement"][c.name] = {"baseline": base.outputs.get(c.name), "best": best.outputs.get(c.name),
                                              "limit": c.describe()}
            out["conclusion"] = (f"best simulated design {best.id}: {o.name} {w:.4g} {o.unit} vs baseline {b:.4g} "
                                 f"({100 * (w - b) / abs(b):+.1f} %), every limit met in the reference simulation")
        else:
            out["conclusion"] = "no simulated design meets every limit" if base is not None else "no baseline"
        return out
