"""The self-modifying loop: mutate a build, simulate every candidate, keep what the physics accepts.

Edits (all on the stud grid):

* ``add``: a new part on top of or under an existing one, overlapping it by at least one stud;
* ``remove``, ``move`` (shift by 1-3 studs or one part height), ``swap`` (another part, same anchor),
  ``rotate`` (non-planar goals) and ``recolour`` (cosmetic: it does not change the score);
* grammar rules that read the last evaluation: ``extend`` (a part past the farthest end, overlapping it),
  ``counterweight`` (a part on the far side of the support), ``brace`` (a part across the joint that failed) and
  ``widen`` (a part next to the base), and ``mirror`` (replace one half of the build by the reflection of the
  other about the goal's target line, for spans).

Two search strategies with the same budget of physics evaluations (duplicates are looked up, not re-simulated):

* :func:`evolve`: (mu + lambda) evolution with NSGA-II survivor selection and Deb's constraint handling
  (:meth:`EvolutionaryDesignOptimizer._nsga2_select`), objectives from the goal (e.g. reach and part count);
* :func:`surrogate_search`: the same edits and selection, but each generation proposes many more children than it
  can afford, a Gaussian-process surrogate (:class:`GaussianProcessSurrogate`) trained on every build simulated so
  far ranks them by an upper confidence bound, and only the best are simulated.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .assembly import Assembly
from .goals import Evaluation, Goal
from .parts import part

EDITS = ("add", "add", "add", "remove", "move", "move", "swap", "recolour", "extend", "extend", "counterweight",
         "brace", "widen", "rotate", "mirror")


def _random_part(goal: Goal, rng) -> str:
    return goal.parts[rng.integers(len(goal.parts))]


def _place_on(goal, asm, rng, anchor, p, above: bool) -> bool:
    b = asm.bricks[anchor]
    rot = 0 if goal.planar else int(rng.integers(2))
    sx, sy = (p.ny, p.nx) if rot else (p.nx, p.ny)
    z = b.top if above else b.z - p.h
    if z < 0:
        return False
    x = int(b.x + rng.integers(-sx + 1, b.sx))
    y = 0 if goal.planar else int(b.y + rng.integers(-sy + 1, b.sy))
    asm.add(p, x, y, z, rot, goal.colours[rng.integers(len(goal.colours))])
    return True


def mutate(goal: Goal, asm: Assembly, rng, last: Evaluation | None = None, edit: str | None = None) -> tuple[Assembly, str]:
    """One random edit (or ``edit``). Returns the new build and the edit name; the build may be invalid."""
    a = asm.copy()
    edit = edit or EDITS[rng.integers(len(EDITS))]
    n = len(a)
    if edit == "rotate" and goal.planar:
        edit = "move"
    if edit == "mirror":
        if goal.target_x is None:
            edit = "add"
        else:
            t2 = 2 * goal.target_x
            left = bool(rng.random() < 0.5)
            keep = [b for b in a.bricks if ((b.x + b.sx / 2) < goal.target_x) == left]
            if keep:
                mirrored = [type(b)(b.part, int(round(t2 - b.x - b.sx)), b.y, b.z, b.rot, b.colour) for b in keep]
                a.bricks = keep + mirrored
                return a, "mirror"
            edit = "add"
    if edit in ("add", "widen") or n == 0:
        if n == 0 or edit == "widen":
            p = part(_random_part(goal, rng))
            ref = a.bricks[rng.integers(n)] if n else None
            x = int((ref.x if ref else 0) + rng.integers(-p.nx, p.nx + 1))
            y = 0 if goal.planar else int((ref.y if ref else 0) + rng.integers(-2, 3))
            a.add(p, x, y, 0, 0, goal.colours[rng.integers(len(goal.colours))])
            return a, "widen" if n else "add"
        _place_on(goal, a, rng, int(rng.integers(n)), part(_random_part(goal, rng)), bool(rng.random() < 0.7))
        return a, "add"
    if edit == "remove" and n > 1:
        a.remove(int(rng.integers(n)))
        return a, "remove"
    if edit == "move":
        b = a.bricks[rng.integers(n)]
        if rng.random() < 0.75:
            d = int(rng.choice([-3, -2, -1, 1, 2, 3]))
            if goal.planar or rng.random() < 0.5:
                b.x += d
            else:
                b.y += d
        else:
            b.z = max(0, b.z + int(rng.choice([-1, 1])) * b.part.h)
        return a, "move"
    if edit == "swap":
        i = int(rng.integers(n))
        b = a.bricks[i]
        a.bricks[i] = type(b)(part(_random_part(goal, rng)), b.x, b.y, b.z, b.rot, b.colour)
        return a, "swap"
    if edit == "rotate":
        b = a.bricks[rng.integers(n)]
        b.rot = 1 - b.rot
        return a, "rotate"
    if edit == "recolour":
        a.bricks[rng.integers(n)].colour = goal.colours[rng.integers(len(goal.colours))]
        return a, "recolour"
    if edit == "extend":                       # a part reaching past a frontier end toward the goal's target
        target = goal.target_x
        if target is None:                     # as far as possible along +x
            ranked = sorted(range(n), key=lambda k: (-a.bricks[k].footprint()[1], -a.bricks[k].z))
        else:                                  # the ends nearest the target (from both sides of a gap)
            def gap_to(k):
                x0, x1 = a.bricks[k].footprint()[:2]
                return 0.0 if x0 <= target <= x1 else min(abs(x0 - target), abs(x1 - target))
            ranked = sorted(range(n), key=gap_to)
        b = a.bricks[ranked[int(rng.integers(min(2, n)))]]
        x0, x1 = b.footprint()[:2]
        right = target is None or x1 <= target or (x0 < target and rng.random() < 0.5)
        p = part(_random_part(goal, rng))
        ov = int(rng.integers(1, max(2, min(p.nx, b.sx))))
        z = b.top if rng.random() < 0.5 else b.z - p.h
        if z < 0:
            z = b.top
        x = x1 - ov if right else x0 + ov - p.nx
        a.add(p, x, b.y, z, 0, goal.colours[rng.integers(len(goal.colours))])
        return a, "extend"
    if edit == "counterweight":                # a part on the side away from the reach (the outer end)
        if goal.target_x is not None and rng.random() < 0.5:
            i = max(range(n), key=lambda k: (a.bricks[k].footprint()[1], -a.bricks[k].top))
        else:
            i = min(range(n), key=lambda k: (a.bricks[k].footprint()[0], -a.bricks[k].top))
        b = a.bricks[i]
        p = part(_random_part(goal, rng))
        a.add(p, b.x - int(rng.integers(0, p.nx)), b.y, b.top, 0, goal.colours[rng.integers(len(goal.colours))])
        return a, "counterweight"
    if edit == "brace":                        # a part across the joint that failed last time
        if last is not None and last.critical:
            lo, up = last.critical[0]
            if 0 <= up < n and lo < n:                       # indices refer to the evaluated parent
                anchor = up if (lo < 0 or rng.random() < 0.5) else lo
                above = anchor == up
                _place_on(goal, a, rng, anchor, part(_random_part(goal, rng)), above)
                return a, "brace"
        _place_on(goal, a, rng, int(rng.integers(n)), part(_random_part(goal, rng)), True)
        return a, "brace"
    return mutate(goal, asm, rng, last, "add")


def repair(goal: Goal, asm: Assembly) -> Assembly:
    """Drop parts that overlap an earlier one or float, so most edits give a buildable candidate."""
    a = asm.copy()
    keep, occ = [], set()
    for b in a.bricks:
        cells = set(b.cells())
        if cells & occ:
            continue
        occ |= cells
        keep.append(b)
    a.bricks = keep
    g = a.grounded()
    a.bricks = [b for i, b in enumerate(a.bricks) if i in g]
    return a


@dataclass
class Candidate:
    asm: Assembly
    ev: Evaluation
    edit: str = "seed"
    generation: int = 0
    parent: int = -1                    # index in the history
    index: int = -1


@dataclass
class SearchResult:
    strategy: str
    goal: str
    seed: int
    budget: int
    history: list[Candidate]            # every simulated candidate, in order
    generations: list[list[int]]        # survivors (history indices) after each generation
    best_curve: np.ndarray              # best feasible score after each evaluation (nan before the first)
    wall_time_s: float
    meta: dict = field(default_factory=dict)

    @property
    def best(self) -> Candidate | None:
        feas = [c for c in self.history if c.ev.feasible]
        if not feas:
            return None
        return max(feas, key=lambda c: (c.ev.score, -c.ev.n_bricks))

    def best_of_generation(self, g: int) -> Candidate | None:
        cs = [self.history[i] for i in self.generations[g] if self.history[i].ev.feasible]
        return max(cs, key=lambda c: (c.ev.score, -c.ev.n_bricks)) if cs else None

    def failures_of_generation(self, g: int) -> list[Candidate]:
        return [c for c in self.history if c.generation == g and not c.ev.feasible and c.ev.status != "invalid"]


class _Evaluator:
    """Counts unique physics evaluations; duplicates (same signature) are looked up."""

    def __init__(self, goal: Goal, budget: int):
        self.goal, self.budget = goal, budget
        self.cache: dict[str, int] = {}
        self.history: list[Candidate] = []
        self.best: list[float] = []

    @property
    def left(self) -> int:
        return self.budget - len(self.history)

    def __call__(self, asm: Assembly, edit: str, generation: int, parent: int) -> int:
        sig = asm.signature()
        if sig in self.cache:
            return self.cache[sig]
        ev = self.goal.evaluate(asm)
        c = Candidate(asm, ev, edit, generation, parent, len(self.history))
        self.history.append(c)
        self.cache[sig] = c.index
        prev = self.best[-1] if self.best else np.nan
        val = ev.score if ev.feasible else np.nan
        self.best.append(np.nanmax([prev, val]) if not (np.isnan(prev) and np.isnan(val)) else np.nan)
        return c.index


def _select(goal: Goal, cands: list[Candidate], mu: int) -> list[int]:
    from pinneapple_design.design_optimizer.optimizer import EvolutionaryDesignOptimizer
    obj = np.array([goal.objectives(c.ev) for c in cands])
    pen = np.array([0.0 if c.ev.feasible else max(c.ev.violation, 1e-9) for c in cands])
    idx = EvolutionaryDesignOptimizer._nsga2_select(obj, pen, mu)
    return [cands[i].index for i in idx]


def _children(goal, ev_, parents: list[int], k: int, rng, gen: int, max_edits: int = 3):
    out = []
    tries = 0
    while len(out) < k and tries < 50 * k:
        tries += 1
        pi = parents[rng.integers(len(parents))]
        if len(parents) > 1:                    # binary tournament on the parents
            pj = parents[rng.integers(len(parents))]
            ci, cj = ev_.history[pi], ev_.history[pj]
            better = (ci.ev.feasible, ci.ev.score, -ci.ev.violation) >= (cj.ev.feasible, cj.ev.score, -cj.ev.violation)
            pi = pi if better else pj
        par = ev_.history[pi]
        asm, edits = par.asm, []
        for j in range(int(rng.integers(1, max_edits + 1))):
            asm, e = mutate(goal, asm, rng, par.ev if j == 0 else None)    # the evaluation is the parent's
            edits.append(e)
        asm = repair(goal, asm)
        if not len(asm) or not goal.in_bounds(asm):
            continue
        if asm.signature() in ev_.cache or any(a.signature() == asm.signature() for a, *_ in out):
            continue
        out.append((asm, "+".join(edits), pi))
    return out


def evolve(goal: Goal, budget: int = 200, *, seed: int = 0, mu: int = 10, lam: int = 10,
           progress: Callable[[str], None] | None = None) -> SearchResult:
    """(mu + lambda) evolution of builds with NSGA-II selection; ``budget`` physics evaluations."""
    return _run(goal, budget, seed, mu, lam, None, progress)


def surrogate_search(goal: Goal, budget: int = 200, *, seed: int = 0, mu: int = 10, lam: int = 10, pool: int = 8,
                     kappa: float = 1.0, warmup: int = 20, progress: Callable[[str], None] | None = None) -> SearchResult:
    """Same loop, but each generation proposes ``pool * lam`` children, a GP surrogate trained on all simulated
    builds ranks them by ``mean + kappa * std`` of a penalised score, and only the top ``lam`` are simulated."""
    return _run(goal, budget, seed, mu, lam, {"pool": pool, "kappa": kappa, "warmup": warmup}, progress)


def _fitness(ev: Evaluation, scale: float) -> float:
    """Scalar for the surrogate: the score when feasible, the score minus a violation penalty otherwise."""
    return ev.score - (0.0 if ev.feasible else scale * (1.0 + ev.violation))


def _run(goal, budget, seed, mu, lam, sur, progress) -> SearchResult:
    rng = np.random.default_rng(seed)
    ev_ = _Evaluator(goal, budget)
    t0 = time.time()
    start = goal.seed()
    population = [ev_(start, "seed", 0, -1)]
    while len(population) < mu and ev_.left > 0:                       # initial population: random edits of the seed
        kids = _children(goal, ev_, population, 1, rng, 0)
        if not kids:
            break
        asm, e, p = kids[0]
        population.append(ev_(asm, e, 0, p))
    generations = [list(population)]
    gen = 0
    sur_meta = {"fits": 0, "rank_corr": []}
    while ev_.left > 0:
        gen += 1
        k = min(lam, ev_.left)
        if sur is None or len(ev_.history) < sur["warmup"]:
            kids = _children(goal, ev_, population, k, rng, gen)
        else:
            pool = _children(goal, ev_, population, k * sur["pool"], rng, gen)
            if pool:
                kids = _rank_by_surrogate(goal, ev_, pool, k, sur, sur_meta)
            else:
                kids = []
        if not kids:
            break
        new = [ev_(asm, e, gen, p) for asm, e, p in kids]
        cands = [ev_.history[i] for i in dict.fromkeys(population + new)]
        population = _select(goal, cands, mu)
        generations.append(list(population))
        if progress:
            b = ev_.best[-1] if ev_.best else float("nan")
            progress(f"{goal.name} {'surrogate' if sur else 'evolution'} seed {seed} gen {gen}: "
                     f"{len(ev_.history)}/{budget} evals, best {b:.1f}")
    curve = np.array(ev_.best, float)
    res = SearchResult("surrogate" if sur else "evolution", goal.name, seed, budget, ev_.history, generations, curve,
                       time.time() - t0, {"mu": mu, "lam": lam, **({"surrogate": sur, **sur_meta} if sur else {})})
    return res


def _rank_by_surrogate(goal, ev_, pool, k, sur, meta):
    from pinneapple_neural.workflows.hybrid_surrogate_physics import GaussianProcessSurrogate
    hist = ev_.history[-300:]
    X = np.array([goal.features(c.asm) for c in hist])
    scores = np.array([c.ev.score for c in hist if c.ev.feasible] or [1.0])
    scale = max(float(np.ptp(scores)), float(np.abs(scores).max()), 1.0)
    y = np.array([_fitness(c.ev, scale) for c in hist])
    mx, sx = X.mean(0), X.std(0) + 1e-9
    gp = GaussianProcessSurrogate(use_sklearn=False).fit((X - mx) / sx, y)
    meta["fits"] += 1
    Xp = np.array([goal.features(a) for a, *_ in pool])
    pred = gp.predict((Xp - mx) / sx)
    ucb = pred.mean[:, 0] + sur["kappa"] * pred.std[:, 0]
    order = np.argsort(-ucb)[:k]
    return [pool[i] for i in order]


def compare(goal_factory: Callable[[], Goal], *, budget: int = 200, seeds=(0, 1, 2, 3, 4),
            strategies=("evolution", "surrogate"), progress=None, surrogate_kw: dict | None = None, **kw) -> dict:
    """Run every strategy with every seed at the same evaluation budget. ``kw`` (mu, lam) go to both strategies,
    ``surrogate_kw`` (pool, kappa, warmup) to the surrogate one. Returns the runs and a summary per strategy:
    ``best_median`` over all seeds with a seed that found no feasible build counted as the worst (nan when most
    seeds found none), min and max over the feasible seeds, mean and standard deviation over the feasible seeds,
    and the success rate."""
    runs = {s: [] for s in strategies}
    for s in strategies:
        for sd in seeds:
            if s == "evolution":
                runs[s].append(evolve(goal_factory(), budget, seed=sd, progress=progress, **kw))
            else:
                runs[s].append(surrogate_search(goal_factory(), budget, seed=sd, progress=progress,
                                                **kw, **(surrogate_kw or {})))
    summary = {}
    for s, rs in runs.items():
        best = np.array([r.best.ev.score if r.best else np.nan for r in rs], float)
        ok = np.isfinite(best)
        curves = np.array([np.pad(r.best_curve, (0, budget - len(r.best_curve)), mode="edge") if len(r.best_curve)
                           else np.full(budget, np.nan) for r in rs])
        med = float(np.median(np.where(ok, best, -np.inf)))       # a seed without a feasible build ranks worst
        summary[s] = {"best_median": med if np.isfinite(med) else float("nan"),
                      "best_mean": float(np.nanmean(best)) if ok.any() else float("nan"),
                      "best_std": float(np.nanstd(best)) if ok.any() else float("nan"),
                      "best_min": float(np.nanmin(best)) if ok.any() else float("nan"),
                      "best_max": float(np.nanmax(best)) if ok.any() else float("nan"),
                      "success_rate": float(ok.mean()), "seeds": list(seeds), "budget": budget,
                      "per_seed": best.tolist(), "curves": curves}
    return {"runs": runs, "summary": summary}
