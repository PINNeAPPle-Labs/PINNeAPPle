"""What an item lacks for the next letter, and the PINNeAPPle action that would close each gap.

The curation (``pinneapple_lab.curation``) computes, for every run, the exact gaps between its letter and the next one
up from the same evidence that gave the letter: the failing checks (with the measured value against its tolerance or
bound) for E and F, the trust-card stage nobody answered for D, the missing independent check or the reproducibility
score for C, the flagship gates for B, and for A what would strengthen it further (a second independent reference, a
quantified uncertainty, a benchmark comparison, a human review). Each gap carries the library actions that address it,
from a curated registry (:data:`ACTIONS`), with the item's own experiment and parameters filled in. A gap with no
registered action says so (``no PINNeAPPle action yet``); nothing outside the library is ever suggested, and a test
resolves every action's import path or command.

    from pinneapple_lab.curation import next_steps
    for gap in next_steps("repo_results/benchmark_arena"):        # from lab/curation.json
        print(gap.to, gap.why, [a.command for a in gap.actions])

    python -m pinneapple_lab seal --explain repo_results/benchmark_arena [--lang pt]
    python -m pinneapple_lab improve repo_results/benchmark_arena [--run]

Stable API (the Flow Studio reads it): :class:`Gap` (``type``, ``to``, ``stage``, ``check``, ``why`` / ``why_pt``,
``short`` / ``short_pt``, ``measured``, ``actions``) and :class:`Action` (``id``, ``kind`` "python" or "cli",
``target``, ``command``, ``params``, ``text`` / ``text_pt``, ``runnable``, ``argv``). Both serialise with
``to_dict`` and load with ``from_dict``; the curation record stores them under ``next_steps``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

NO_ACTION = ("no PINNeAPPle action yet", "ainda sem ação PINNeAPPle")
STRENGTHEN = "A+"                                   # ``Gap.to`` of what would strengthen an A item further
_RESOLUTION = r"^(n|nx|ny|nz|n_?cells|cells|elements|n_?elements|resolution|grid|n_?grid|refine|level|epochs|" \
              r"n_?collocation|iterations|steps)$"


@dataclass(frozen=True)
class Action:
    id: str
    kind: str                     # "python" (a call to make in the experiment) or "cli" (a command to run)
    target: str                   # dotted import path ("pinneapple.compare") or "cli:pinneapple_lab run"
    command: str                  # the call or command, with the item's experiment and parameters filled in
    text: str
    text_pt: str
    params: dict = field(default_factory=dict)
    runnable: bool = False        # ``improve --run`` may execute it (registered, side-effect-free lab commands only)
    argv: tuple = ()              # for runnable commands: the argument list of the in-process call

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["argv"] = list(self.argv)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Action:
        return cls(**{**d, "argv": tuple(d.get("argv", ()))})


@dataclass(frozen=True)
class Gap:
    type: str                     # a key of GAP_TYPES
    to: str                       # the letter this gap stands between the item and ("A+" = strengthen an A)
    stage: str                    # trust-card stage it belongs to
    why: str
    why_pt: str
    short: str                    # one line for the seal: "B: pass an independent check"
    short_pt: str
    check: str = ""               # the check concerned, when there is one
    measured: dict | None = None  # {"value", "threshold", "op"} when the curation measured something
    actions: tuple = ()           # Action, possibly empty: then NO_ACTION applies

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["actions"] = [a.to_dict() if isinstance(a, Action) else a for a in self.actions]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gap:
        return cls(**{**d, "actions": tuple(Action.from_dict(a) for a in d.get("actions", []))})

    def line(self, lang: str = "en") -> str:
        return f"{self.to}: {self.short_pt if lang == 'pt' else self.short}"


# ---------------------------------------------------------------------------------------------- the registry
# action id -> kind, target (checked by the tests), command template, (EN, PT) text, condition, runnable
ACTIONS: dict[str, dict[str, Any]] = {
    "lab_run": dict(kind="cli", target="cli:pinneapple_lab run", runnable=True,
                    command="python -m pinneapple_lab run {experiment}{params} --force",
                    text=("rerun the item from a committed tree (after the fix); the curation grades it again",
                          "reexecutar o item a partir de um commit (depois da correção); a curadoria reavalia")),
    "lab_sweep_refine": dict(kind="cli", target="cli:pinneapple_lab sweep", runnable=True, needs="resolution",
                             command="python -m pinneapple_lab sweep {experiment} -g {res}={res_values}{fixed}",
                             text=("rerun with {res} doubled and see whether the result converges",
                                   "reexecutar com {res} dobrado e ver se o resultado converge")),
    "lab_sweep_seeds": dict(kind="cli", target="cli:pinneapple_lab sweep", runnable=True, needs="seed",
                            command="python -m pinneapple_lab sweep {experiment} -g seed=0,1,2{fixed}",
                            text=("repeat with three seeds: the curation measures how much the key metric varies",
                                  "repetir com três sementes: a curadoria mede quanto a métrica principal varia")),
    "physics_check_cli": dict(kind="cli", target="cli:pinneapple check", runnable=True,
                              command="pinneapple check {run_dir}",
                              text=("lint the run's fields for conservation, boundary conditions, residual and units",
                                    "verificar conservação, condições de contorno, resíduo e unidades nos campos")),
    "ctx_check_physics": dict(kind="python", target="pinneapple_lab.context.RunContext.check",
                              command='ctx.check("<law>_balance", passed, detail="...", kind="physics")',
                              text=("record a law the result was not fitted to (mass, energy, equilibrium) as a check",
                                    "registrar uma lei que o resultado não ajustou (massa, energia, equilíbrio)")),
    "ctx_check_reference": dict(kind="python", target="pinneapple_lab.context.RunContext.check",
                                command='ctx.check("<quantity>_vs_<source>", value=v, reference=ref, rtol=tol, '
                                        'kind="reference", label="<quantity> vs <source>")',
                                text=("compare with an exact solution, a measurement or a published value, with its "
                                      "source", "comparar com uma solução exata, medição ou valor publicado, com a "
                                      "fonte")),
    "ctx_check_generalization": dict(kind="python", target="pinneapple_lab.context.RunContext.check",
                                     command='ctx.check("error_on_unseen_<cases>", value=err, max=tol, '
                                             'kind="generalization")',
                                     text=("score held-out or out-of-distribution inputs as a check",
                                           "avaliar entradas fora do treino como verificação")),
    "compare_reference": dict(kind="python", target="pinneapple.compare",
                              command='pp.compare(problem, [result, "fem"], reference="exact")',
                              text=("compare on the same points with a reference or a baseline solver at equal "
                                    "budget", "comparar nos mesmos pontos com uma referência ou um solver de base, "
                                    "com o mesmo orçamento")),
    "trust_card": dict(kind="python", target="pinneapple.trust_card",
                       command="pp.trust_card(grid, values, pde=..., dirichlet=..., reference=...)",
                       text=("check the field's PDE residual, boundary conditions and conservation",
                             "verificar resíduo da EDP, condições de contorno e conservação do campo")),
    "validate_model": dict(kind="python", target="pinneapple.validate",
                           command='pp.validate(model, problem, level="standard")',
                           text=("validate the model per factor: accuracy, physics, applicability range",
                                 "validar o modelo por fator: exatidão, física, faixa de aplicação")),
    "validate_pinn": dict(kind="python", target="pinneapple_analysis.validation.pinn.validate_pinn", needs="pinn",
                          command="validate_pinn(net, problem)",
                          text=("diagnose the PINN on fresh points (residual, failure modes such as a collapsed or "
                                "spectrally biased solution) before retraining",
                                "diagnosticar a PINN em pontos novos (resíduo, modos de falha) antes de retreinar")),
    "trust_gate": dict(kind="python", target="pinneapple_analysis.trust.TrustGate",
                       command="TrustGate(train_bounds)",
                       text=("narrow the declared validity range to the inputs where it passes, and gate queries "
                             "outside it", "restringir a faixa de validade declarada às entradas onde passa e "
                             "bloquear consultas fora dela")),
    "gci": dict(kind="python", target="pinneapple_lab.uq.gci_check", needs="solver",
                command='gci_check(ctx, "<quantity>", [coarse, fine], [h_coarse, h_fine], order=2, max_band=0.05)',
                text=("grid convergence index from two or three meshes: the discretisation error band",
                      "índice de convergência de malha com duas ou três malhas: a banda de erro de discretização")),
    "coverage": dict(kind="python", target="pinneapple_lab.uq.coverage_check", needs="ml",
                     command='coverage_check(ctx, "<quantity>", y_true, mean, std, level=0.9)',
                     text=("check that the predictive interval holds its nominal fraction of unseen values",
                           "verificar que o intervalo preditivo contém a fração nominal dos valores não vistos")),
    "ensemble": dict(kind="python", target="pinneapple_veriphysics.robustness.ensemble_study",
                     command="ensemble_study(...)",
                     text=("ensemble spread over seeds or members", "dispersão de um ensemble de sementes ou membros")),
    "benchmark_run": dict(kind="python", target="pinneapple.benchmark.run",
                          command='pp.benchmark.run(config, out_dir="benchmark_out")',
                          text=("an equal-budget benchmark row against other methods",
                                "uma linha de benchmark com o mesmo orçamento contra outros métodos")),
    "ctx_figure": dict(kind="python", target="pinneapple_lab.context.RunContext.figure",
                       command='ctx.figure("<name>", fig); ctx.gif("<name>", frames)',
                       text=("save the key plots, a movie or a render with the run",
                             "salvar os gráficos principais, um vídeo ou um render com a execução")),
    "field_view": dict(kind="python", target="pinneapple.viz.field_view",
                       command="pp.viz.field_view(mesh, fields)",
                       text=("a 3-D view of the field on its mesh", "uma vista 3-D do campo na malha")),
    "experiment_references": dict(kind="python", target="pinneapple_lab.spec.Experiment.references",
                                  command="references = [\"<paper or data source>\"]  # in the Experiment class",
                                  text=("cite the papers or data the experiment reproduces, and cite `{experiment}` "
                                        "in a docs page", "citar os artigos ou dados que o experimento reproduz e "
                                        "citar `{experiment}` numa página da documentação")),
    "lab_review": dict(kind="cli", target="cli:pinneapple_lab review",
                       command='python -m pinneapple_lab review {experiment} --story "..." --novelty N --approve paper',
                       text=("a human review: novelty, clarity, the one-line story",
                             "uma revisão humana: novidade, clareza, a história numa linha")),
}

# gap type -> stage, (EN, PT) short line, action ids (in order of preference), F-causing
GAP_TYPES: dict[str, dict[str, Any]] = {
    "fix_crashed": dict(stage="decision", short=("make the run finish without an error",
                                                 "fazer a execução terminar sem erro"),
                        actions=["lab_run"]),
    "fix_diverged": dict(stage="data_geometry", short=("make the result finite and converged",
                                                       "tornar o resultado finito e convergido"),
                         actions=["lab_sweep_refine", "validate_pinn", "lab_run"]),
    "fix_physics": dict(stage="physics", short=("make the physics check pass", "fazer a verificação física passar"),
                        actions=["physics_check_cli", "trust_card", "lab_sweep_refine", "validate_pinn"]),
    "fix_contradicts": dict(stage="benchmark", short=("agree with the reference", "concordar com a referência"),
                            actions=["validate_pinn", "lab_sweep_refine", "compare_reference"]),
    "fix_reference": dict(stage="benchmark", short=("bring the error vs the reference within its tolerance",
                                                    "trazer o erro face à referência para a tolerância"),
                          actions=["lab_sweep_refine", "validate_pinn", "compare_reference"]),
    "fix_generalization": dict(stage="model", short=("pass on the unseen inputs, or narrow the validity range",
                                                     "passar nos dados não vistos ou restringir a faixa"),
                               actions=["trust_gate", "validate_model"]),
    "fix_baseline": dict(stage="benchmark", short=("beat the baseline", "superar o baseline"),
                         actions=["compare_reference", "benchmark_run"]),
    "fix_check": dict(stage="decision", short=("fix the failing check", "corrigir a verificação que falha"),
                      actions=["lab_run", "physics_check_cli"]),
    "no_evidence": dict(stage="physics", short=("record a physics or reference check",
                                                "registrar uma verificação física ou de referência"),
                        actions=["ctx_check_physics", "ctx_check_reference", "trust_card", "physics_check_cli"]),
    "no_independent": dict(stage="benchmark", short=("pass an independent check (reference, baseline or physics)",
                                                     "passar numa verificação independente"),
                           actions=["ctx_check_reference", "compare_reference", "ctx_check_physics"]),
    "reproducibility": dict(stage="decision", short=("rerun from a committed tree with a code snapshot",
                                                     "reexecutar a partir de um commit com o código"),
                            actions=["lab_run", "lab_sweep_seeds"]),
    "no_reference": dict(stage="benchmark", short=("compare with an independent reference",
                                                   "comparar com uma referência independente"),
                         why=("no passing reference check (exact solution, measurement, published value)",
                              "nenhuma verificação de referência passa (solução exata, medição, valor publicado)"),
                         actions=["ctx_check_reference", "compare_reference"]),
    "no_baseline_or_physics": dict(stage="physics", short=("pass a baseline or physics check",
                                                           "passar numa verificação de baseline ou física"),
                                   why=("no passing baseline or physics check", "nenhuma verificação de baseline ou "
                                        "física passa"),
                                   actions=["ctx_check_physics", "compare_reference", "physics_check_cli"]),
    "generalization": dict(stage="model", short=("test on unseen inputs", "testar em entradas não vistas"),
                           why=("a learned model with no passing check on unseen inputs",
                                "modelo aprendido sem verificação que passe em entradas não vistas"),
                           actions=["ctx_check_generalization", "validate_model"]),
    "assets": dict(stage="decision", short=("add figures: three plots, a movie, a render or a 3-D view",
                                            "adicionar figuras: três gráficos, vídeo, render ou vista 3-D"),
                   actions=["ctx_figure", "field_view"]),
    "documentation": dict(stage="decision", short=("document it: references and a docs page",
                                                   "documentar: referências e uma página"),
                          actions=["experiment_references"]),
    "consistency": dict(stage="decision", short=("make most of its runs pass", "fazer a maioria das execuções passar"),
                        actions=["lab_run"]),
    "second_reference": dict(stage="benchmark", short=("add a second independent reference",
                                                       "adicionar uma segunda referência independente"),
                             actions=["ctx_check_reference", "compare_reference"]),
    "uncertainty": dict(stage="uncertainty", short=("quantify the uncertainty", "quantificar a incerteza"),
                        actions=["gci", "coverage", "ensemble"]),
    "benchmark_row": dict(stage="benchmark", short=("add an equal-budget benchmark comparison",
                                                    "adicionar uma comparação de benchmark"),
                          why=("no passing baseline check: not compared with another method",
                               "nenhuma verificação de baseline: não comparado com outro método"),
                          actions=["benchmark_run", "compare_reference"]),
    "review": dict(stage="decision", short=("get a human review", "obter uma revisão humana"),
                   why=("no human review recorded (python -m pinneapple_lab review)",
                        "nenhuma revisão humana registrada"),
                   actions=["lab_review"]),
}
F_CAUSING = ("fix_crashed", "fix_diverged", "fix_physics", "fix_contradicts")


# ---------------------------------------------------------------------------------------------- filling in
def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def measured(check: dict[str, Any]) -> dict[str, Any] | None:
    """The measured value of a check against its threshold, from what the check recorded (None if no number)."""
    v = check.get("value")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    ref = check.get("reference")
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        tol = (check.get("atol") or 0.0) + (check.get("rtol") or 0.0) * abs(float(ref))
        return {"value": v, "threshold": ref, "op": "±", "tolerance": tol}
    if isinstance(check.get("max"), (int, float)):
        return {"value": v, "threshold": check["max"], "op": "<="}
    if isinstance(check.get("min"), (int, float)):
        return {"value": v, "threshold": check["min"], "op": ">="}
    return {"value": v, "threshold": None, "op": ""}


def measured_text(m: dict[str, Any] | None, lang: str = "en") -> str:
    if not m:
        return ""
    pt = lang == "pt"
    val = ("valor " if pt else "value ") + _fmt(m["value"])
    if m.get("op") == "±":
        return f"{val}, {'referência' if pt else 'reference'} {_fmt(m['threshold'])} ± {_fmt(m['tolerance'])}"
    if m.get("op") == "<=":
        return f"{val}, {'máx.' if pt else 'max'} {_fmt(m['threshold'])}"
    if m.get("op") == ">=":
        return f"{val}, {'mín.' if pt else 'min'} {_fmt(m['threshold'])}"
    if m.get("op") == "score":
        return f"{'pontuação' if pt else 'score'} {_fmt(m['value'])} < {_fmt(m['threshold'])}"
    return val


def context_for(experiment: str, params: dict[str, Any], run_dir: str, tags: set[str], is_ml: bool,
             cls_params: dict[str, Any] | None) -> dict[str, Any]:
    known = cls_params if cls_params is not None else params
    res = next((k for k, v in known.items() if re.match(_RESOLUTION, k) and isinstance(params.get(k, v), int)
                and not isinstance(params.get(k, v), bool) and params.get(k, v) > 0), None)
    return {"experiment": experiment, "params": {k: v for k, v in params.items() if k in known}, "run_dir": run_dir,
            "tags": set(tags), "is_ml": is_ml, "res": res, "seed": "seed" in known,
            "pinn": bool(set(tags) & {"pinn", "xtfc", "physics-ai"}) or "pinn" in experiment}


def _p(params: dict[str, Any], skip: tuple = ()) -> list[str]:
    out = []
    for k, v in params.items():
        if k not in skip:
            out += ["-p", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    return out


def make_action(aid: str, ctx: dict[str, Any]) -> Action | None:
    """The registered action ``aid`` filled in for the item, or None when its condition does not hold."""
    spec = ACTIONS[aid]
    need = spec.get("needs")
    if need == "resolution" and not ctx["res"]:
        return None
    if need == "seed" and not ctx["seed"]:
        return None
    if need == "pinn" and not ctx["pinn"]:
        return None
    if need == "ml" and not ctx["is_ml"]:
        return None
    if need == "solver" and ctx["is_ml"]:
        return None
    exp, params = ctx["experiment"], ctx["params"]
    argv: list[str] = []
    fill = {"experiment": exp, "run_dir": ctx["run_dir"], "params": "", "fixed": "", "res": ctx["res"] or "",
            "res_values": ""}
    if aid == "lab_run":
        argv = ["run", exp, *_p(params), "--force"]
        fill["params"] = "".join(" " + x for x in _p(params))
    elif aid == "lab_sweep_refine":
        v = int(params.get(ctx["res"]))
        fill["res_values"] = f"{v},{2 * v}"
        argv = ["sweep", exp, "-g", f"{ctx['res']}={v},{2 * v}", *_p(params, (ctx["res"],))]
        fill["fixed"] = "".join(" " + x for x in _p(params, (ctx["res"],)))
    elif aid == "lab_sweep_seeds":
        argv = ["sweep", exp, "-g", "seed=0,1,2", *_p(params, ("seed",))]
        fill["fixed"] = "".join(" " + x for x in _p(params, ("seed",)))
    elif aid == "physics_check_cli":
        argv = ["check", ctx["run_dir"]]
    en, pt = spec["text"]
    return Action(id=aid, kind=spec["kind"], target=spec["target"], command=spec["command"].format(**fill),
                  text=en.format(**fill), text_pt=pt.format(**fill),
                  params={"experiment": exp, **({"run_dir": ctx["run_dir"]} if aid == "physics_check_cli" else {}),
                          **({"params": params} if argv and aid != "physics_check_cli" else {})},
                  runnable=bool(spec.get("runnable")), argv=tuple(argv))


def make_gap(gtype: str, to: str, ctx: dict[str, Any], *, check: dict[str, Any] | None = None,
             why: tuple[str, str] | None = None, meas: dict[str, Any] | None = None) -> Gap:
    """A gap of a registered type, its actions filled in for the item (``KeyError`` for an unregistered type)."""
    spec = GAP_TYPES[gtype]
    acts = tuple(a for a in (make_action(i, ctx) for i in spec["actions"]) if a is not None)
    if check is not None:
        from .curation import check_line
        meas = measured(check) if meas is None else meas
        line = check_line(check, 72)
        en = f"failing check {line}" + (f" ({measured_text(meas)})" if meas else "")
        pt = f"verificação que falha: {line}" + (f" ({measured_text(meas, 'pt')})" if meas else "")
        why = why or (en, pt)
    short_en, short_pt = spec["short"]
    why = why or spec.get("why") or (short_en, short_pt)
    return Gap(type=gtype, to=to, stage=spec["stage"], why=why[0], why_pt=why[1], short=short_en, short_pt=short_pt,
               check=(check or {}).get("name", ""), measured=meas, actions=acts)


def group(gaps) -> list[dict[str, Any]]:
    """Gaps that share the same actions (same commands, same order) merged into one entry, in first-seen order:
    ``[{"gaps": [gap, ...], "actions": [action, ...]}]``. Works on :class:`Gap` objects or their dicts, so three
    failing checks fixed by the same commands list the commands once."""
    def cmds(g):
        acts = g.actions if isinstance(g, Gap) else g.get("actions", [])
        return tuple(a.command if isinstance(a, Action) else a.get("command") for a in acts)
    out: list[dict[str, Any]] = []
    index: dict[tuple, int] = {}
    for g in gaps:
        k = cmds(g)
        if k in index:
            out[index[k]]["gaps"].append(g)
        else:
            index[k] = len(out)
            out.append({"gaps": [g], "actions": list(g.actions if isinstance(g, Gap) else g.get("actions", []))})
    return out


def explain(gaps: list[Gap], lang: str = "en", title: str = "") -> str:
    """The gaps as text: gap, why (with the measured value), and the suggested PINNeAPPle actions, listed once for
    all the gaps they address (:func:`group`)."""
    pt = lang == "pt"
    out = [title] if title else []
    if not gaps:
        out.append("Nada em falta." if pt else "Nothing missing.")
    i = 0
    for grp in group(gaps):
        for g in grp["gaps"]:
            i += 1
            out.append(f"{i}. [{g.to}] {g.short_pt if pt else g.short}")
            out.append(f"   {'porquê' if pt else 'why'}: {g.why_pt if pt else g.why}")
        if len(grp["gaps"]) > 1:
            out.append(f"   {'para os' if pt else 'for the'} {len(grp['gaps'])} "
                       f"{'pontos acima' if pt else 'gaps above'}:")
        if not grp["actions"]:
            out.append(f"   -> {NO_ACTION[1] if pt else NO_ACTION[0]}")
        for a in grp["actions"]:
            out.append(f"   -> {a.text_pt if pt else a.text}")
            out.append(f"      {a.command}")
    return "\n".join(out)
