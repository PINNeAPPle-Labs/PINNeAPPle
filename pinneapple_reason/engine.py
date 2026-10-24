"""The reasoning engine: check every requirement, find the first failure, try the countermeasures, keep the best,
check again, until every requirement passes.

    from pinneapple_reason import reason, cases
    chain = reason(*cases.roof())            # requirements, naive concept, title
    print(chain.explain())                   # version 0 -> version N, and why

At each step the engine

1. runs every failure mode of every requirement on the current concept (in the order the requirements and their
   modes are given) and takes the **first** one that fails;
2. applies each countermeasure the library has for that mode (plus any an advisor proposes) to a copy of the concept,
   and re-runs **every** check on the copy: which requirement each option fixes or breaks, the target margin before and
   after, and what it costs;
3. keeps the cheapest option that fixes the failure without breaking anything that passed (among options within 1 %
   of the lowest cost: the one leaving most checks passing, then the largest margin); if none fixes it alone, the one that improves the target margin most (then the cheapest),
   and goes on; an option that brings back an earlier version is refused (no cycles).

An advisor (an LLM through :class:`pinneapple_reason.llm.LLMAdvisor`) may add candidate countermeasures from the
library; it never supplies a number: every margin, cost and mechanism comes from the checks.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .chain import (
    MAX_STEPS,
    RESOLVED,
    UNRESOLVED,
    Chain,
    Concept,
    Finding,
    Option,
    Requirement,
    Step,
    Verification,
    text_of,
    tr,
)
from .library import LIBRARY, Infeasible, Library

__all__ = ["Reasoner", "reason", "Advisor"]


class Advisor(Protocol):
    def propose_countermeasures(self, finding: Finding, concept: Concept, library: Library) -> list[str]: ...


class Reasoner:
    def __init__(self, library: Library = LIBRARY, advisor: Advisor | None = None, max_steps: int = 12,
                 progress: Callable[[str], None] | None = None):
        self.library = library
        self.advisor = advisor
        self.max_steps = int(max_steps)
        self.progress = progress or (lambda msg: None)
        self._cache: dict[tuple, Verification] = {}
        self.evaluations = 0

    # -- checks ----------------------------------------------------------------------------------------------------
    def verify(self, concept: Concept, requirements: Sequence[Requirement]) -> list[Verification]:
        out = []
        ck = concept.key()
        for r in requirements:
            for m in r.modes:
                key = (ck, r.id, m, tuple(sorted(r.limits.items())))
                if key not in self._cache:
                    self.evaluations += 1
                    self._cache[key] = self.library.mode(m).evaluate(concept, r.id, r.limits)
                out.append(self._cache[key])
        return out

    @staticmethod
    def first_failure(vs: Sequence[Verification], library: Library = LIBRARY) -> Finding | None:
        for v in vs:
            if not v.passed:
                return Finding(v.requirement, v.mode, library.mode(v.mode).title, v)
        return None

    # -- countermeasures -------------------------------------------------------------------------------------------
    def candidates(self, finding: Finding, concept: Concept) -> list[tuple[Any, str]]:
        cms = [(c, "library") for c in self.library.for_mode(finding.mode)]
        if self.advisor is not None:
            try:
                names = self.advisor.propose_countermeasures(finding, concept, self.library)
            except Exception as exc:  # noqa: BLE001 - an advisor failure never stops the reasoning
                self.progress(f"advisor failed: {exc}")
                names = []
            known = {c.name for c, _ in cms}
            for n in names:
                if n in self.library.countermeasures and n not in known:
                    cms.append((self.library.countermeasures[n], "llm"))
                    known.add(n)
        return cms

    def evaluate_options(self, finding: Finding, concept: Concept, requirements: Sequence[Requirement],
                         vs: Sequence[Verification], visited: set[str]) -> tuple[list[Option], list[Concept | None]]:
        limits = next((r.limits for r in requirements if r.id == finding.requirement), {})
        before = {(v.requirement, v.mode): v.passed for v in vs}
        options, trials = [], []
        for cm, by in self.candidates(finding, concept):
            ok, why = cm.applicable(concept)
            base = dict(countermeasure=cm.name, element=cm.element, title=cm.title, effect=cm.effect,
                        proposed_by=by, target_margin_before=finding.verification.margin)
            if not ok:
                options.append(Option(feasible=False, reason=why, **base))
                trials.append(None)
                continue
            try:
                trial = cm.apply(concept, finding.verification, limits)
            except Infeasible as exc:
                options.append(Option(feasible=False, reason=exc.text, **base))
                trials.append(None)
                continue
            if trial.key() in visited:
                options.append(Option(feasible=False, reason=tr("volta a uma versão anterior", "returns to an earlier "
                                                                "version"), **base))
                trials.append(None)
                continue
            vt = self.verify(trial, requirements)
            tgt = next(v for v in vt if v.requirement == finding.requirement and v.mode == finding.mode)
            changes = {k: trial.params[k] for k in trial.params if concept.params.get(k) != trial.params[k]}
            regress = [f"{v.requirement}/{v.mode}" for v in vt if before.get((v.requirement, v.mode)) and not v.passed]
            options.append(Option(feasible=True, changes=changes, cost=cm.cost(concept, trial),
                                  target_margin_after=tgt.margin, fixes_target=tgt.passed,
                                  passing_after=sum(v.passed for v in vt), regressions=regress, **base))
            trials.append(trial)
        return options, trials

    @staticmethod
    def choose(options: Sequence[Option]) -> tuple[int | None, dict[str, str]]:
        feas = [(i, o) for i, o in enumerate(options) if o.feasible]
        fix_clean = [(i, o) for i, o in feas if o.fixes_target and not o.regressions]
        fix_any = [(i, o) for i, o in feas if o.fixes_target]
        pool = fix_clean or fix_any
        if pool:
            cheapest = min(t[1].cost.get("index", 0.0) for t in pool)
            near = [t for t in pool if t[1].cost.get("index", 0.0) <= cheapest * 1.01 + 1e-12]
            i, o = max(near, key=lambda t: (t[1].passing_after, t[1].target_margin_after or 0.0))
            n = len(pool)
            others = [p for _, p in pool if p is not o]
            why_pt = ("resolve a falha" + (" sem quebrar nenhum requisito que passava" if not o.regressions else
                                           f" (mas quebra {', '.join(o.regressions)}, tratado no passo seguinte)"))
            why_en = ("fixes the failure" + (" without breaking any requirement that passed" if not o.regressions else
                                             f" (but breaks {', '.join(o.regressions)}, handled next)"))
            if n > 1:
                why_pt += (f"; entre {n} opções que resolvem, é a de menor custo ({o.cost.get('index', 0):.3g}, "
                           f"{o.passing_after} verificações passando; contra " + ", ".join(
                               f"{p.countermeasure}: {p.passing_after} passando, custo {p.cost.get('index', 0):.3g}"
                               for p in others) + ")")
                why_en += (f"; of {n} options that fix it, it is the cheapest ({o.cost.get('index', 0):.3g}, "
                           f"{o.passing_after} checks passing; vs " + ", ".join(
                               f"{p.countermeasure}: {p.passing_after} passing, cost {p.cost.get('index', 0):.3g}"
                               for p in others) + ")")
            return i, tr(why_pt, why_en)
        better = [(i, o) for i, o in feas if o.target_margin_after is not None and o.target_margin_before is not None
                  and o.target_margin_after > o.target_margin_before]
        if better:
            # the margin before is the same for every option: the largest margin after is the largest gain
            i, o = max(better, key=lambda t: (t[1].target_margin_after, -t[1].cost.get("index", 0.0)))
            return i, tr(f"nenhuma opção resolve sozinha; esta é a que mais melhora a margem "
                         f"({o.target_margin_before:.3g} → {o.target_margin_after:.3g}), e o passo seguinte continua",
                         f"no option fixes it alone; this one improves the margin most "
                         f"({o.target_margin_before:.3g} -> {o.target_margin_after:.3g}), and the next step goes on")
        return None, tr("nenhuma contramedida da biblioteca melhora este modo de falha",
                        "no countermeasure in the library improves this failure mode")

    # -- the loop --------------------------------------------------------------------------------------------------
    def run(self, requirements: Sequence[Requirement], concept: Concept, title: Any = "") -> Chain:
        requirements = list(requirements)
        for r in requirements:
            for m in r.modes:
                self.library.mode(m)                       # fail early on an unknown mode
        t0 = time.perf_counter()
        steps: list[Step] = []
        visited = {concept.key()}
        status = MAX_STEPS
        cur = concept
        for k in range(self.max_steps + 1):
            vs = self.verify(cur, requirements)
            finding = self.first_failure(vs, self.library)
            step = Step(cur.version, cur, vs, finding)
            steps.append(step)
            if finding is None:
                status = RESOLVED
                self.progress(f"v{cur.version}: every requirement passes")
                break
            self.progress(f"v{cur.version}: {finding.mode} fails ({finding.proof})")
            if k == self.max_steps:
                break
            options, trials = self.evaluate_options(finding, cur, requirements, vs, visited)
            idx, why = self.choose(options)
            step.options, step.chosen, step.why = options, idx, why
            if idx is None:
                status = UNRESOLVED
                break
            self.progress(f"  -> {options[idx].countermeasure} ({text_of(why, 'en')[:120]})")
            cur = trials[idx]
            visited.add(cur.key())
        title = title if isinstance(title, dict) else tr(str(title), str(title))
        return Chain(title, requirements, steps, status,
                     meta={"evaluations": self.evaluations, "wall_s": time.perf_counter() - t0,
                           "advisor": type(self.advisor).__name__ if self.advisor else None})


def reason(requirements: Sequence[Requirement], concept: Concept, title: Any = "", *, advisor: Advisor | None = None,
           max_steps: int = 12, library: Library = LIBRARY, progress: Callable[[str], None] | None = None) -> Chain:
    """Build the chain of whys for ``requirements`` starting from the naive ``concept``."""
    return Reasoner(library, advisor, max_steps, progress).run(requirements, concept, title)
