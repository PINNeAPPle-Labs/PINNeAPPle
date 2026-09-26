"""Autoresearch mode: an automated experiment loop over a single training script.

Pattern from karpathy/autoresearch (MIT, ``ROADMAP.md`` §11): a fixed-time-budget training script,
one metric, and a proposer that edits the script between runs; changes that improve the metric are
kept, the others reverted, and every trial is logged. Here, adapted to physics AI:

- ``program.md`` holds the research instructions and the physics context for the proposer.
- ``train.py`` is the only file that changes. It must print ``METRIC <name>=<value>`` (lower is
  better, e.g. relative L2 against a closed form) and respect ``PINNEAPPLE_TIME_BUDGET`` seconds.
- Edits are limited to the ``# >>> TUNABLE`` ... ``# <<< TUNABLE`` block (``NAME = value`` lines),
  which keeps each proposal small, reviewable and safe to apply automatically.
- Proposers: :class:`RandomSearchProposer` (no LLM, reproducible) and :class:`LLMProposer`
  (any provider of ``pinneapple_llm``, or your own ``llm_call(prompt, system=...) -> str``).

Every trial goes to ``results.tsv``; the best script so far is kept as ``best_train.py``.

    from pinneapple_arena.autoresearch import AutoResearch, RandomSearchProposer, init_workdir
    init_workdir("runs/poisson")  # program.md + train.py (1D Poisson PINN, rel-L2 vs sin(pi x))
    AutoResearch("runs/poisson", RandomSearchProposer({"LR": [1e-3, 3e-3, 1e-2], "WIDTH": [16, 32, 64]}),
                 budget_s=60).run(max_trials=20)
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

_BLOCK = re.compile(r"(# >>> TUNABLE[^\n]*\n)(.*?)(# <<< TUNABLE)", re.S)
_ASSIGN = re.compile(r"^(\s*)([A-Z_][A-Z0-9_]*)\s*=\s*([^#\n]+?)(\s*#.*)?$", re.M)
_METRIC = re.compile(r"^METRIC\s+([\w.-]+)\s*=\s*([-+0-9.eEinfa]+)\s*$", re.M)


def read_tunables(code: str) -> Dict[str, Any]:
    m = _BLOCK.search(code)
    if not m:
        raise ValueError("train.py has no '# >>> TUNABLE' ... '# <<< TUNABLE' block")
    return {a.group(2): ast.literal_eval(a.group(3).strip()) for a in _ASSIGN.finditer(m.group(2))}


def apply_tunables(code: str, changes: Dict[str, Any]) -> str:
    """Rewrite values inside the TUNABLE block only; unknown names are rejected."""
    m = _BLOCK.search(code)
    known = read_tunables(code)
    unknown = set(changes) - set(known)
    if unknown:
        raise KeyError(f"not tunable: {sorted(unknown)} (tunable: {sorted(known)})")

    def sub(a):
        name = a.group(2)
        if name not in changes:
            return a.group(0)
        return f"{a.group(1)}{name} = {changes[name]!r}{a.group(4) or ''}"

    block = _ASSIGN.sub(sub, m.group(2))
    return code[: m.start(2)] + block + code[m.end(2):]


@dataclass
class Trial:
    index: int
    metric: float
    status: str  # ok | timeout | crash | no_metric
    kept: bool
    description: str
    seconds: float
    changes: Dict[str, Any]


def run_trial(workdir: str, budget_s: float, *, python: str = sys.executable, grace_s: float = 30.0):
    """Run train.py once. Returns (metric, status, stdout_tail, seconds)."""
    env = dict(os.environ, PINNEAPPLE_TIME_BUDGET=str(budget_s))
    t0 = time.time()
    try:
        p = subprocess.run([python, "train.py"], cwd=workdir, env=env, capture_output=True, text=True,
                           timeout=budget_s + grace_s)
        out = p.stdout + p.stderr
        status = "ok" if p.returncode == 0 else "crash"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        status = "timeout"
    found = _METRIC.findall(out)
    metric = float(found[-1][1]) if found else math.inf
    if status == "ok" and not found:
        status = "no_metric"
    if not math.isfinite(metric):
        metric = math.inf
    return metric, status, out[-2000:], round(time.time() - t0, 2)


class RandomSearchProposer:
    """Samples one or two tunables at a time from a discrete space (no LLM, reproducible)."""

    def __init__(self, space: Dict[str, Sequence[Any]], seed: int = 0, max_changes: int = 2):
        self.space = {k: list(v) for k, v in space.items()}
        self.rng = random.Random(seed)
        self.max_changes = max_changes

    def __call__(self, program: str, code: str, history: List[Trial]) -> Dict[str, Any]:
        cur = read_tunables(code)
        names = [k for k in self.space if k in cur]
        pick = self.rng.sample(names, k=min(len(names), self.rng.randint(1, self.max_changes)))
        changes = {}
        for k in pick:
            options = [v for v in self.space[k] if v != cur[k]] or self.space[k]
            changes[k] = self.rng.choice(options)
        return {"changes": changes, "description": "random: " + ", ".join(f"{k}={v}" for k, v in changes.items())}


class LLMProposer:
    """Asks an LLM for the next edit of the TUNABLE block, given program.md and the trial log."""

    SYSTEM = ("You run physics-AI experiments. Propose ONE small, well-motivated change to the "
              "tunable parameters to lower the metric. Answer with JSON only: "
              '{"changes": {"NAME": value, ...}, "description": "why, in one sentence"}.')

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None,
                 llm_call: Optional[Callable[..., str]] = None, history_len: int = 15, **llm_kwargs):
        self.provider, self.model, self.llm_call = provider, model, llm_call
        self.history_len, self.llm_kwargs = history_len, llm_kwargs

    def _call(self, prompt: str) -> str:
        if self.llm_call is not None:
            return self.llm_call(prompt, system=self.SYSTEM)
        from pinneapple_llm._dispatch import call_llm
        return call_llm(prompt, provider=self.provider, model=self.model, system=self.SYSTEM,
                        json_mode=True, module="arena.autoresearch", **self.llm_kwargs)

    def __call__(self, program: str, code: str, history: List[Trial]) -> Dict[str, Any]:
        log = "\n".join(f"#{t.index} metric={t.metric:.4g} {t.status} {'KEPT' if t.kept else 'reverted'} "
                        f"{t.description}" for t in history[-self.history_len:])
        prompt = (f"# Research program\n{program}\n\n# Current tunables (best script so far)\n"
                  f"{json.dumps(read_tunables(code))}\n\n# Trial log (lower metric is better)\n{log}\n")
        text = self._call(prompt)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"LLM answer has no JSON object: {text[:200]!r}")
        data = json.loads(m.group(0))
        return {"changes": dict(data.get("changes", {})), "description": str(data.get("description", "llm"))[:200]}


class AutoResearch:
    def __init__(self, workdir: str, proposer: Callable[[str, str, List[Trial]], Dict[str, Any]],
                 budget_s: float = 300.0, python: str = sys.executable):
        self.workdir, self.proposer, self.budget_s, self.python = workdir, proposer, budget_s, python
        self.train = os.path.join(workdir, "train.py")
        self.best_path = os.path.join(workdir, "best_train.py")
        self.log_path = os.path.join(workdir, "results.tsv")
        self.history: List[Trial] = []

    def _log(self, t: Trial) -> None:
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if new:
                f.write("trial\tmetric\tstatus\tkept\tseconds\tdescription\tchanges\n")
            f.write(f"{t.index}\t{t.metric:.6g}\t{t.status}\t{int(t.kept)}\t{t.seconds}\t"
                    f"{t.description}\t{json.dumps(t.changes)}\n")

    def run(self, max_trials: int = 10) -> Trial:
        program_path = os.path.join(self.workdir, "program.md")
        program = open(program_path).read() if os.path.exists(program_path) else ""
        best_code = open(self.train).read()
        metric, status, _, sec = run_trial(self.workdir, self.budget_s, python=self.python)
        best = Trial(0, metric, status, True, "baseline", sec, {})
        self.history.append(best)
        self._log(best)
        open(self.best_path, "w").write(best_code)
        for i in range(1, max_trials + 1):
            try:
                prop = self.proposer(program, best_code, self.history)
                code = apply_tunables(best_code, prop["changes"])
            except Exception as e:  # a bad proposal is logged, never applied
                t = Trial(i, math.inf, f"bad_proposal: {e}"[:120], False, "proposal rejected", 0.0, {})
                self.history.append(t)
                self._log(t)
                continue
            open(self.train, "w").write(code)
            metric, status, _, sec = run_trial(self.workdir, self.budget_s, python=self.python)
            kept = status == "ok" and metric < best.metric
            t = Trial(i, metric, status, kept, prop.get("description", ""), sec, prop["changes"])
            self.history.append(t)
            self._log(t)
            if kept:
                best, best_code = t, code
                open(self.best_path, "w").write(code)
            else:
                open(self.train, "w").write(best_code)  # revert
        return best


PROGRAM_TEMPLATE = """# Research program: 1D Poisson PINN

Goal: minimise the relative L2 error of a PINN for u''(x) = -pi^2 sin(pi x) on [-1, 1],
u(-1) = u(1) = 0, whose exact solution is u = sin(pi x). The metric is printed by train.py as
`METRIC rel_l2=<value>` and each run has a fixed time budget, so faster convergence matters.

Physics hints: the solution is smooth and single-scale; boundary conditions are imposed as a
penalty with weight BC_WEIGHT; second-order quasi-Newton refinement (SSBroyden) after Adam usually
lowers the error by orders of magnitude on problems like this one.
"""

TRAIN_TEMPLATE = '''"""Autoresearch target: 1D Poisson PINN. Only the TUNABLE block is edited by the proposer."""
import math
import os
import time

import torch

try:  # imported before the clock starts: a heavy import must not eat the training budget
    from pinneapple_neural.trainer.self_scaled_qn import SelfScaledQuasiNewton
except ImportError:
    SelfScaledQuasiNewton = None

# >>> TUNABLE (edited by pinneapple_arena.autoresearch)
LR = 1e-3
WIDTH = 32
DEPTH = 2
N_COLLOCATION = 128
BC_WEIGHT = 10.0
ADAM_FRACTION = 1.0  # share of the time budget spent in Adam; the rest in SSBroyden
# <<< TUNABLE

BUDGET = float(os.environ.get("PINNEAPPLE_TIME_BUDGET", "60"))
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
layers, d = [], 1
for _ in range(DEPTH):
    layers += [torch.nn.Linear(d, WIDTH), torch.nn.Tanh()]
    d = WIDTH
net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
x = torch.linspace(-1, 1, N_COLLOCATION).reshape(-1, 1).requires_grad_(True)
xb = torch.tensor([[-1.0], [1.0]])
f = (-(math.pi ** 2) * torch.sin(math.pi * x)).detach()


def loss_fn():
    u = net(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    return ((d2u - f) ** 2).mean() + BC_WEIGHT * (net(xb) ** 2).mean()


t0 = time.time()
opt = torch.optim.Adam(net.parameters(), lr=LR)
while time.time() - t0 < ADAM_FRACTION * BUDGET * 0.9:
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()
    opt.step()
if ADAM_FRACTION < 1.0 and SelfScaledQuasiNewton is not None:
    qn = SelfScaledQuasiNewton(net.parameters(), variant="ssbroyden", max_iter=10)

    def closure():
        qn.zero_grad()
        l = loss_fn()
        l.backward()
        return l

    while time.time() - t0 < BUDGET * 0.85:  # one step is several iterations: stop with margin
        qn.step(closure)
xt = torch.linspace(-1, 1, 1001).reshape(-1, 1)
with torch.no_grad():
    ref = torch.sin(math.pi * xt)
    err = float(torch.linalg.norm(net(xt) - ref) / torch.linalg.norm(ref))
print(f"METRIC rel_l2={err:.6e}")
'''


def init_workdir(path: str, *, overwrite: bool = False) -> str:
    """Create ``program.md`` + ``train.py`` (1D Poisson PINN template) in ``path``."""
    os.makedirs(path, exist_ok=True)
    for name, text in (("program.md", PROGRAM_TEMPLATE), ("train.py", TRAIN_TEMPLATE)):
        p = os.path.join(path, name)
        if overwrite or not os.path.exists(p):
            open(p, "w").write(text)
    return path
