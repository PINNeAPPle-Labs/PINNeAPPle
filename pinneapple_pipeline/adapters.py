"""The adapters: the only place where a configuration meets a module that computes something.

Every function here calls existing code — ``pinneapple_physics.pde_environment`` presets and condition sampling,
``pinneapple_physics.symbolic_pde``, ``pinneapple_physics.solving`` (reference and PINN solves, ``compare``),
``pinneapple_data.doe`` designs, ``pinneapple_analysis.surrogate`` studies, ``pinneapple_tools.model_export`` —
and converts the result into the few objects the stages pass between themselves. No physics, no training loop and
no metric is implemented here: if a number is needed that the library does not already compute, it is reported as
missing instead of being invented.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ProblemBundle", "build_problem", "problem_description", "design_cases", "probe_points", "solve_reference",
           "solution_row", "train_physics_model", "train_pinn_with_early_stopping", "reload_physics_model",
           "run_surrogate_study", "residual_batch", "physics_terms", "export_model", "predict_frame"]


# ------------------------------------------------------------------------------------------------------- problem
@dataclass
class ProblemBundle:
    """The problem a run is about, whatever it was declared as.

    ``backend`` says how the rest of the pipeline must treat it: ``"solve"`` (a ``PhysicalProblem`` that
    ``pinneapple_physics.solving`` can solve and compare) or ``"symbolic"`` (an ``EquationSystem`` with a
    ``PINNProblem``, trained and validated by ``pinneapple_analysis.validation.pinn``).
    """

    backend: str
    problem: Any = None
    spec: Any = None
    system: Any = None
    pinn_problem: Any = None
    pinn_model: Any = None                      # the network pinn_problem() built, reused by the training stage
    notes: list[str] = field(default_factory=list)

    @property
    def coords(self) -> tuple[str, ...]:
        if self.problem is not None:
            return tuple(self.problem.coords)
        return tuple(self.system.coords)

    @property
    def fields(self) -> tuple[str, ...]:
        if self.problem is not None:
            return tuple(self.problem.fields)
        return tuple(self.system.fields)

    @property
    def domains(self) -> dict[str, tuple[float, float]]:
        if self.problem is not None:
            return {c: tuple(map(float, b)) for c, b in self.problem.domain_bounds.items()}
        return {c: tuple(map(float, self.pinn_problem.bounds[i])) for i, c in enumerate(self.coords)}

    @property
    def name(self) -> str:
        return getattr(self.problem, "name", None) or getattr(self.pinn_problem, "name", "symbolic_problem")


def build_problem(cfg) -> ProblemBundle:
    """The problem of :class:`~pinneapple_pipeline.configs.ProblemConfig`, with any extra conditions applied."""
    pc = cfg.problem
    if pc.source == "preset":
        import pinneapple as pp
        from pinneapple_physics.solving import as_problem
        spec = pp.get_preset(pc.preset, **dict(pc.preset_kwargs))
        bundle = ProblemBundle("solve", problem=as_problem(spec), spec=spec)
    elif pc.source == "spec":
        from pinneapple_physics.solving import as_problem
        problem = as_problem(pc.problem)
        bundle = ProblemBundle("solve", problem=problem, spec=problem.to_pde_spec()
                               if hasattr(problem, "to_pde_spec") else None)
    elif pc.source == "symbolic":
        from pinneapple_physics.symbolic_pde.pinn import pinn_problem
        from pinneapple_physics.symbolic_pde.text import parse_system
        constants = dict(pc.constants)
        material_notes: list[str] = []
        materials = getattr(cfg, "materials", None)
        if materials is not None and materials.materials:
            for name, v in materials.targets(set(constants)).items():
                constants[name] = float(v["value"])
                material_notes.append(f"material {v['body']}: {name} = {v['value']:.6g} ({v['property']}; "
                                      f"{v['source']})")
        system = parse_system(list(pc.equations), list(pc.conditions), coords=list(pc.coords),
                              fields=list(pc.fields), coefficients=constants)
        issues = system.issues()
        if issues:
            raise ValueError("the symbolic problem is incomplete:\n  - " + "\n  - ".join(issues))
        exact = None
        if pc.exact is not None:
            if not callable(pc.exact):
                raise ValueError("problem.exact must be a callable X -> (n, n_fields)")
            exact = pc.exact
        elif pc.exact_expression:
            from .configs import _expression_fn
            exact = _expression_fn(pc.exact_expression, list(pc.coords), constants)
        pinn, net = pinn_problem(system, {c: tuple(pc.domain_bounds[c]) for c in pc.coords}, exact=exact)
        if exact is not None:
            bundle_notes = "an exact solution is declared: the model gate measures the error against it"
        else:
            bundle_notes = ""
        bundle = ProblemBundle("symbolic", system=system, pinn_problem=pinn, pinn_model=net)
        if bundle_notes:
            bundle.notes.append(bundle_notes)
        bundle.notes.append(f"symbolic system: {len(system.pdes)} PDE(s), {len(system.conditions)} condition(s), "
                            f"coefficients {dict(system.coefficients)}")
        bundle.notes.extend(material_notes)
    else:                                                          # pragma: no cover - validated in the config
        raise ValueError(f"unknown problem source {pc.source!r}")

    if cfg.boundary_conditions:
        if bundle.backend != "solve":
            raise ValueError("boundary_conditions can only be added to a problem built by pinneapple_physics "
                             "(sources preset and spec); a symbolic problem declares its conditions in "
                             "problem.conditions")
        extra = tuple(c.to_condition(bundle.coords, bundle.domains) for c in cfg.boundary_conditions)
        bundle.problem = dataclasses.replace(bundle.problem, conditions=tuple(bundle.problem.conditions) + extra)
        bundle.notes.append(f"added {len(extra)} condition(s) from the configuration: "
                            f"{[c.name for c in extra]}")
    _apply_materials(cfg, bundle)
    return bundle


def _apply_materials(cfg, bundle: ProblemBundle) -> None:
    """Set the problem's own parameters from the material catalogue, and record where each number came from."""
    materials = getattr(cfg, "materials", None)
    if materials is None or not materials.materials:
        return
    if bundle.backend != "solve":                       # the symbolic coefficients are applied before parsing
        return
    names = set(bundle.problem.parameters)
    values = materials.targets(names)
    if not values:
        bundle.notes.append(f"materials declared, but the problem has none of their properties among {sorted(names)}")
        return
    bundle.problem = bundle.problem.with_parameters(**{k: float(v["value"]) for k, v in values.items()})
    for name, v in values.items():
        bundle.notes.append(f"material {v['body']}: {name} = {v['value']:.6g} ({v['property']}; {v['source']})")


def problem_description(bundle: ProblemBundle) -> dict[str, Any]:
    """What the problem is, as the run records it (and the ``problem`` gate checks the configuration against)."""
    out = {"name": bundle.name, "backend": bundle.backend, "coords": list(bundle.coords),
           "fields": list(bundle.fields), "domain_bounds": {k: list(v) for k, v in bundle.domains.items()}}
    if bundle.problem is not None:
        out["pde"] = getattr(bundle.problem, "pde", None)
        out["conditions"] = [{"name": c.name, "kind": c.kind, "fields": list(c.fields)}
                             for c in bundle.problem.conditions]
        out["reference_solver"] = dict(getattr(bundle.problem, "reference_solver", {}) or {})
        out["parameters"] = {k: {"value": p.value, "role": p.role} for k, p in
                             getattr(bundle.problem, "parameters", {}).items()}
    else:
        out["pde"] = {"kind": "symbolic",
                      "equations": [e.text for e in bundle.system.pdes],
                      "conditions": [e.text for e in bundle.system.conditions],
                      "params": dict(bundle.system.coefficients)}
    if hasattr(out.get("pde"), "to_dict") or dataclasses.is_dataclass(out.get("pde")):
        pde = out["pde"]
        out["pde"] = {"kind": pde.kind, "fields": list(pde.fields), "coords": list(pde.coords),
                      "params": dict(pde.params)}
    return out


# ------------------------------------------------------------------------------------------------------- design
def design_cases(cfg) -> Any:
    """The cases of the parameter space (a ``Design``), or ``None`` for a physics-only run."""
    space = cfg.parameters.to_space(f"{cfg.name}_parameters")
    if space is None:
        return None
    from pinneapple_data.doe import design
    method = str(cfg.sampling.design_sampling.get("method", "lhs"))
    n = int(cfg.sampling.design_sampling.get("n_samples", 16))
    if method == "full_factorial":
        return design(space, n, method="full_factorial",
                      levels=int(cfg.sampling.design_sampling.get("levels", 3)), seed=cfg.sampling.seed)
    return design(space, n, method=method, seed=cfg.sampling.seed)


def probe_points(cfg, bundle: ProblemBundle) -> np.ndarray:
    """The points where a field solution becomes the surrogate's outputs.

    ``data.probes``: how many values per coordinate (space of the field); ``data.probe_at``: coordinates held fixed
    (a time, a plane). Coordinates that are neither get the default count, and a time coordinate with no fixed value
    is taken at the end of its interval: a surrogate of the final state, unless the config asks for the whole time
    history.
    """
    coords = list(bundle.coords)
    domains = bundle.domains
    probes = dict(cfg.data.probes)
    at = dict(cfg.data.probe_at)
    unknown = sorted((set(probes) | set(at)) - set(coords))
    if unknown:
        raise ValueError(f"data.probes/probe_at name {unknown}, which the problem (coords {tuple(coords)}) does "
                         f"not have")
    axes = []
    for c in coords:
        lo, hi = domains[c]
        if c in at:
            axes.append(np.array([float(at[c])]))
        elif c in probes:
            axes.append(np.linspace(lo, hi, int(probes[c])))
        elif c == "t":
            axes.append(np.array([hi]))
        else:
            axes.append(np.linspace(lo, hi, 5))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


# ----------------------------------------------------------------------------------------------- reference data
def solve_reference(cfg, bundle: ProblemBundle, case: Mapping[str, Any]) -> Any:
    """One reference solution for one case (parameter values -> ``Solution``)."""
    import pinneapple as pp
    problem = bundle.problem
    values = {k: v for k, v in dict(case).items() if k in getattr(problem, "parameters", {})}
    if values:
        problem = problem.with_parameters(**{k: float(v) if not isinstance(v, str) else v for k, v in values.items()})
    return pp.solve(problem, method=cfg.solver.method, **dict(cfg.solver.options))


def solution_row(solution, points: np.ndarray) -> np.ndarray:
    """The field values at ``points`` as one row of the dataset."""
    return np.asarray(solution.predict(points), dtype=float).ravel()


# ------------------------------------------------------------------------------------------------------- models
def train_physics_model(cfg, bundle: ProblemBundle):
    """Train a physics-only model (a PINN) on the problem; returns a ``Solution``-like object.

    ``solve`` backend: ``pp.solve(problem, method="pinn")`` with the architecture and training options of the
    configuration — or, when ``training.early_stopping`` is on, the chunked loop of
    :func:`train_pinn_with_early_stopping`. ``symbolic`` backend: ``train_pinn`` on the ``PINNProblem``, wrapped in
    a small object with the same ``predict`` surface.
    """
    if bundle.backend == "symbolic":
        return _train_symbolic(cfg, bundle)

    import pinneapple as pp
    arch = cfg.architecture
    family = arch.candidates[0] if arch.is_auto else arch.family
    options = {"architecture": family, "epochs": int(cfg.training.epochs), "lr": float(cfg.training.lr),
               "n_collocation": int(cfg.training.n_collocation), "seed": int(cfg.training.seed),
               "device": cfg.training.resolved_device(), "weights": cfg.loss.to_weights()}
    if cfg.training.hidden_dim:
        options["hidden_dim"] = int(cfg.training.hidden_dim)
    if cfg.training.n_layers:
        options["n_layers"] = int(cfg.training.n_layers)
    options.update({k: v for k, v in arch.parameters.items() if k in ("hidden_dim", "n_layers")})
    if cfg.training.early_stopping:
        return train_pinn_with_early_stopping(cfg, bundle, family, options)
    return pp.solve(bundle.problem, method="pinn", **options)


def _train_symbolic(cfg, bundle: ProblemBundle):
    """``train_pinn`` on the symbolic problem, stopping early when the physics validation says to.

    ``training.early_stopping`` here is ``PINNMonitor``: it watches the checks of
    ``pinneapple_analysis.validation.pinn`` (boundary, initial, residual, causality, trivial solution) while the
    optimizer runs and stops at the first epoch where a physical failure has persisted for ``patience`` checks —
    "the loss is going down and the condition is still wrong" is exactly what the monitor exists for. Without it,
    the run trains for the whole budget and the validation gate says what it found afterwards.
    """
    from pinneapple_analysis.validation.pinn import PINNMonitor, train_pinn

    tc = cfg.training
    monitor = None
    if tc.early_stopping:
        monitor = PINNMonitor(bundle.pinn_problem, every=int(tc.check_every), warmup=int(tc.check_every),
                              patience=int(tc.patience))
    model, info = train_pinn(bundle.pinn_problem, bundle.pinn_model, n_collocation=int(tc.n_collocation),
                             adam_steps=int(tc.epochs), lbfgs_steps=int(tc.lbfgs_steps),
                             lr=float(tc.lr), seed=int(tc.seed), monitor=monitor,
                             weights={"boundary": cfg.loss.weights.get("w_bc", 10.0),
                                      "initial": cfg.loss.weights.get("w_ic", 10.0)})
    return _SymbolicModel(bundle, model, info)


def train_pinn_with_early_stopping(cfg, bundle: ProblemBundle, family: str, options: Mapping[str, Any]):
    """Train a PINN in chunks, keeping the weights with the lowest validation loss and stopping when it stops falling.

    Why a loop and not a single call: ``pp.solve(..., method="pinn")`` trains for the whole budget in one shot, and
    there is no way to ask it to stop. ``pinneapple_physics.solve_pde`` trains a model *in place* for ``epochs``,
    so calling it repeatedly on the same module continues the same training. Between chunks the pipeline measures
    the held-out physics loss — ``compile_problem``'s total on a fixed batch drawn once, at points the optimizer
    never sees again — keeps the best ``state_dict``, and stops after ``patience`` chunks without an improvement of
    at least ``min_delta``.

    The objective is the physics loss, not a test-set error: reducing the held-out residual is what a PINN can
    honestly do without an answer to look at. When the run also has a reference solution, that error is reported
    separately by the ``model`` gate. A spec whose conditions are geometry tags needs the fixtures ``_pinn``
    resolves; this loop does not resolve them, so it falls back to one ``pp.solve`` and says so in the notes.
    """
    import copy
    import time

    import torch

    from pinneapple_physics import compile_problem, solve_pde

    tc = cfg.training
    spec = bundle.spec if bundle.spec is not None else bundle.problem.to_pde_spec()
    try:
        from pinneapple_neural.architectures.registry import ModelRegistry
        net, report = ModelRegistry.build_with_report(
            family, in_dim=len(bundle.problem.coords), out_dim=len(bundle.problem.fields),
            hidden_dim=int(tc.hidden_dim or options.get("hidden_dim") or 64),
            n_layers=int(tc.n_layers or options.get("n_layers") or 4))
    except Exception as exc:                                     # an architecture the loop cannot build
        bundle.notes.append(f"early stopping fell back to one solve: {type(exc).__name__}: {exc}")
        import pinneapple as pp
        return pp.solve(bundle.problem, method="pinn", **dict(options))
    weights = cfg.loss.to_weights()
    loss_fn = compile_problem(spec, weights=weights)
    val_batch, missing = residual_batch(spec, n_interior=int(tc.n_collocation), seed=int(tc.seed) + 101)
    chunk = int(tc.check_every)
    total = int(tc.epochs)
    best, best_loss, bad, done = None, float("inf"), 0, 0
    history, checks, t0 = [], [], time.perf_counter()
    stopped_early = False
    while done < total:
        n = min(chunk, total - done)
        try:
            out = solve_pde(spec, net, epochs=n, lr=float(tc.lr), n_collocation=int(tc.n_collocation),
                            seed=int(tc.seed) + done, device=tc.resolved_device(), weights=weights)
        except Exception as exc:
            if "tag" in f"{type(exc).__name__}{exc}".lower() and done == 0:
                bundle.notes.append(f"early stopping could not run on this problem ({type(exc).__name__}: {exc}): "
                                    "trained once for the whole budget instead")
                import pinneapple as pp
                return pp.solve(bundle.problem, method="pinn", **dict(options))
            raise
        done += n
        history.extend(float(v) for v in (out.get("history") or {}).get("loss") or [])
        with torch.enable_grad():
            value = float(loss_fn(net, None, dict(val_batch))["total"].detach())
        checks.append({"epochs": done, "validation_loss": value})
        if value < best_loss - float(tc.min_delta):
            best_loss, bad = value, 0
            best = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= int(tc.patience):
                stopped_early = True
                break
    if best is not None:
        net.load_state_dict(best)
    seconds = time.perf_counter() - t0
    if stopped_early:
        bundle.notes.append(f"early stopping: no improvement of more than {tc.min_delta:g} in {tc.patience} check(s) "
                            f"of {chunk} epoch(s); stopped at {done} of {total} epochs, keeping the best "
                            f"(validation loss {best_loss:.4g})")
    if missing:
        bundle.notes.extend(missing)
    return _chunked_pinn_solution(cfg, bundle, net, {"history": {"loss": history}, "seconds": seconds,
                                                     "adam_steps": done, "budget": total,
                                                     "stopped_early": stopped_early, "checks": checks,
                                                     "best_validation_loss": best_loss})


def _chunked_pinn_solution(cfg, bundle: ProblemBundle, net, info: Mapping[str, Any]):
    """The early-stopped network as a real ``Solution``, so ``pp.compare`` accepts it like any other method."""
    import torch

    from pinneapple_physics.solving import Solution

    def predict_fn(x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            out = net(torch.tensor(x))
        out = out[0] if isinstance(out, (tuple, list)) else getattr(out, "y", out)
        return np.asarray(out.detach().cpu().numpy(), dtype=float).reshape(len(x), -1)

    detail = {k: v for k, v in info.items() if k != "history"}
    return Solution(problem=bundle.problem, method="pinn", predict_fn=predict_fn,
                    wall_time_s=float(info.get("seconds", 0.0)), history=dict(info.get("history") or {}),
                    info={"architecture": cfg.architecture.family, "early_stopping": True, **detail},
                    model=net, kind="neural")


class _SymbolicModel:
    """A trained ``PINNProblem`` network behind the ``predict``/``model`` surface the pipeline expects."""

    def __init__(self, bundle: ProblemBundle, model, info: Mapping[str, Any]):
        self.problem, self.model, self.info = bundle.pinn_problem, model, dict(info)
        self.history = {"loss": list(dict(info).get("history") or [])}
        self.method, self.kind = "pinn", "neural"
        self.wall_time_s = float(info.get("seconds", 0.0))
        self.grid = None

    def predict(self, x) -> np.ndarray:
        import torch
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            out = self.model(torch.tensor(x))
        out = out[0] if isinstance(out, (tuple, list)) else getattr(out, "y", out)
        return np.asarray(out.detach().cpu().numpy(), dtype=float).reshape(len(x), -1)

    def metadata(self) -> dict[str, Any]:
        return {"problem": self.problem.name, "method": "pinn", "kind": "neural",
                "wall_time_s": self.wall_time_s, "fields": list(self.problem.outputs),
                "coords": [f"x{i}" for i in range(self.problem.dim)], "on_grid": False, "info": self.info}


def run_surrogate_study(cfg, data, *, methods: Sequence[str] | str = "auto", seeds: int = 1):
    """The surrogate study of ``pinneapple_analysis.surrogate`` on a dataset (fit, split, compare, gate)."""
    from pinneapple_analysis import surrogate

    v = cfg.sampling.validation_sampling
    split_options = {"test_frac": float(v.get("test_fraction", 0.2)), "val_frac": float(v.get("val_fraction", 0.15)),
                     "shift": v.get("shift", "auto")}
    budget = float(cfg.training.budget_seconds or 10.0)
    return surrogate.run_study(data, methods=methods, budget=budget, seeds=int(seeds),
                               split_options=split_options, threads=int(cfg.execution.threads))


# ------------------------------------------------------------------------------------------- physics evidence
def residual_batch(spec, *, n_interior: int = 2048, n_condition: int | None = None, seed: int = 0,
                   ctx: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """A batch of fresh points for the compiled physics loss of ``spec``, and what could not be sampled.

    Interior points fill the box; every condition whose selector is self-describing (``all`` or ``callable``) draws
    its own points on the box and its faces, exactly as the training loop does. A condition that selects by geometry
    tag needs a mesh this pipeline never had, so it is reported in the second return value instead of being
    silently dropped.
    """
    import torch

    from pinneapple_physics.pde_environment.condition_sampling import (
        box_face_normals,
        sample_condition_points,
    )

    bounds, coords = spec.domain_bounds, list(spec.coords)
    rng = np.random.default_rng(seed)
    n_condition = int(n_condition or max(64, n_interior // 4))
    ctx = dict(ctx or {})
    x_col = np.stack([rng.uniform(*bounds[c], size=n_interior) for c in coords], axis=1).astype(np.float32)
    notes, buckets = [], {"bc": ([], [], []), "ic": ([], []), "data": ([], [])}
    for cond in spec.conditions:
        suffix = {"dirichlet": "bc", "neumann": "bc", "robin": "bc", "initial": "ic"}.get(cond.kind, "data")
        if cond.selector_type not in ("all", "callable"):
            notes.append(f"condition {cond.name!r} selects by geometry tag {cond.selector!r}: no mesh in this run, "
                         "its error is not measured")
            continue
        x_sel = sample_condition_points(lambda X, _c=cond: _c.mask(X, ctx), bounds, coords, n_condition, rng)
        if x_sel.shape[0] == 0:
            notes.append(f"condition {cond.name!r} selects no point on the box or its faces")
            continue
        normals = None
        if suffix == "bc":
            normals, on_face = box_face_normals(x_sel, bounds, coords)
            if cond.kind in ("neumann", "robin") and not on_face.all():
                x_sel, normals = x_sel[on_face], normals[on_face]
                if x_sel.shape[0] == 0:
                    continue
        y = cond.values(x_sel, ctx)
        buckets[suffix][0].append(x_sel)
        buckets[suffix][1].append(y)
        if suffix == "bc":
            buckets["bc"][2].append(normals)

    def _cat(parts, width, dtype=np.float32):
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, width), dtype=dtype)

    batch = {"x_col": torch.as_tensor(x_col), "ctx": ctx}
    for suffix, n_fields in (("bc", len(spec.fields)), ("ic", len(spec.fields)), ("data", len(spec.fields))):
        xs, ys = buckets[suffix][0], buckets[suffix][1]
        batch[f"x_{suffix}"] = torch.as_tensor(_cat(xs, len(coords)))
        batch[f"y_{suffix}"] = torch.as_tensor(_cat(ys, n_fields))
        if suffix == "bc":
            batch["n_bc"] = torch.as_tensor(_cat(buckets["bc"][2], len(coords)))
    return batch, notes


def physics_terms(cfg, bundle: ProblemBundle, model) -> tuple[dict[str, float], list[str]]:
    """The terms of the compiled physics loss of a trained model on fresh points (``{"pde": ..., "bc_...": ...}``)."""
    if bundle.backend == "symbolic":
        return {}, ["the symbolic backend is validated by validate_pinn, not by the compiled loss"]
    from pinneapple_physics import compile_problem

    spec = bundle.spec if bundle.spec is not None else bundle.problem.to_pde_spec()
    batch, notes = residual_batch(spec, n_interior=int(cfg.training.n_collocation), seed=int(cfg.training.seed) + 1)
    net = getattr(model, "model", None) or getattr(model, "net", None)
    if net is None:
        return {}, ["the trained object exposes no torch module to evaluate the loss on"] + notes
    loss_fn = compile_problem(spec, weights=cfg.loss.to_weights())
    net.eval()
    with _grad_enabled():
        out = loss_fn(net, None, batch)
    terms = {k: float(v.detach().cpu()) for k, v in out.items() if hasattr(v, "detach")}
    return terms, notes


class _grad_enabled:
    """``torch.enable_grad()`` when torch is loaded (a physics residual needs autograd inside no_grad contexts)."""

    def __enter__(self):
        import torch
        self._ctx = torch.enable_grad()
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# ------------------------------------------------------------------------------------------------------- export
def export_model(cfg, model, out_dir, *, example_input: np.ndarray | None = None) -> dict[str, Any]:
    """Write the checkpoint and the requested portable formats; returns what was written and what failed."""
    from pathlib import Path

    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = getattr(model, "model", None)
    written: dict[str, Any] = {}
    if net is None:
        return {"error": "the trained object exposes no torch module to export", "written": written}
    net.eval()
    exportable = _TensorOutput(net)
    ckpt = out / "model.pt"
    torch.save({"state_dict": net.state_dict(), "class_name": type(net).__name__}, ckpt)
    written["pytorch"] = str(ckpt)
    if example_input is None:
        n_in = int(getattr(net, "in_dim", None) or _infer_in_dim(net))
        example_input = np.zeros((4, n_in), dtype=np.float32)
    x = torch.as_tensor(np.asarray(example_input, dtype=np.float32))
    for fmt in cfg.export.formats:
        if fmt == "pytorch":
            continue
        try:
            if fmt == "torchscript":
                from pinneapple_tools.model_export import export_torchscript
                written["torchscript"] = export_torchscript(exportable, str(out / "model.ts"), x)
            elif fmt == "onnx":
                from pinneapple_tools.model_export import export_onnx
                written["onnx"] = export_onnx(exportable, str(out / "model.onnx"), x)
            elif fmt == "npz":
                with torch.no_grad():
                    y = exportable(x)
                np.savez(out / "model.npz", example_input=x.numpy(), example_output=np.asarray(y.detach()))
                written["npz"] = str(out / "model.npz")
        except Exception as exc:                                  # a format the architecture cannot export
            written[f"{fmt}_error"] = f"{type(exc).__name__}: {exc}"
    return {"written": written, "net": exportable, "example_input": x}


class _TensorOutput:
    """Wrap a model whose forward returns anything tensor-like so an exporter sees a plain tensor."""

    def __new__(cls, net):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                y = self.inner(x)
                if torch.is_tensor(y):
                    return y
                if isinstance(y, (tuple, list)):
                    return y[0]
                return getattr(y, "y", y)

        wrapped = _Wrapped(net)
        wrapped.eval()
        return wrapped


def _infer_in_dim(net) -> int:
    for m in net.modules():
        if hasattr(m, "in_features"):
            return int(m.in_features)
    raise ValueError("cannot infer the model's input dimension; pass example_input")


def predict_frame(model, points: np.ndarray) -> np.ndarray:
    """``model.predict`` with the output always ``(n, n_fields)``."""
    y = np.asarray(model.predict(points), dtype=float)
    return y.reshape(len(points), -1) if y.ndim != 2 else y


# ------------------------------------------------------------------------------------------------ reloading
def reload_physics_model(cfg, bundle: ProblemBundle, run_dir):
    """The trained physics model of a finished run, rebuilt from ``model/physics/`` (checkpoint + recipe).

    ``resume`` in another process, ``from_run`` and the deployment path all need the network back without the
    object that trained it. The weights are loaded with ``torch.load(weights_only=True)`` — a state dict of tensors,
    which cannot execute code — and the module is rebuilt from the recipe the training stage wrote: the network
    the symbolic backend already builds, or a registry architecture with the same size.
    """
    import json
    from pathlib import Path

    folder = Path(run_dir) / "model" / "physics"
    weights, card = folder / "model.pt", folder / "model.json"
    if not weights.is_file() or not card.is_file():
        return None
    import torch

    payload = torch.load(weights, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not state:
        return None
    recipe = json.loads(card.read_text(encoding="utf-8"))
    if bundle.backend == "symbolic":
        net = bundle.pinn_model
        try:
            net.load_state_dict(state)
        except Exception:                                       # a different network: the report says so
            return None
        net.eval()
        return _SymbolicModel(bundle, net, {"history": recipe.get("history") or [], "seconds": 0.0,
                                            "reloaded_from": str(folder)})
    from pinneapple_neural.architectures.registry import ModelRegistry

    family = recipe.get("architecture") or cfg.architecture.family
    try:
        net, _ = ModelRegistry.build_with_report(
            family if family != "auto" else "modified_mlp", in_dim=len(bundle.problem.coords),
            out_dim=len(bundle.problem.fields), hidden_dim=int(recipe.get("hidden_dim") or 64),
            n_layers=int(recipe.get("n_layers") or 4))
        net.load_state_dict(state)
    except Exception:
        return None
    net.eval()
    info = {"history": {"loss": recipe.get("history") or []}, "seconds": 0.0, "reloaded_from": str(folder),
            "architecture": family}
    return _chunked_pinn_solution(cfg, bundle, net, info)
