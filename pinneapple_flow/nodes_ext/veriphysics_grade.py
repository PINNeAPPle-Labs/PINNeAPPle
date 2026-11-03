"""VeriPhysics grade of a model trained elsewhere (:mod:`pinneapple_veriphysics.model_grade`, #1187): the
equations, conditions and reference data the model claims to satisfy, its residuals by finite differences of its
outputs (any framework: TorchScript, ONNX, a library callable, or its predictions on points), applicability and
robustness, and a decision APPROVED / REVIEW / REJECT with the evidence."""
from __future__ import annotations

from ..registry import Param, Port, node

HEAT = {"pdes": ["u_t = alpha*lap(u)"],
        "conditions": [{"kind": "initial", "on": "t_min", "equation": "u = sin(pi*x)"},
                       {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
        "coords": ["x", "t"], "fields": ["u"], "coefficients": {"alpha": 0.1},
        "bounds": {"x": [0.0, 1.0], "t": [0.0, 1.0]}, "exact": "exp(-alpha*pi**2*t)*sin(pi*x)"}


def _demo_pinn(problem, steps: int):
    """A small PINN trained here on the problem (stands in for a model from another framework)."""
    import numpy as np
    import torch

    from pinneapple_analysis.validation.pinn import train_pinn
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    from pinneapple_physics.symbolic_pde.text import parse_system
    s = parse_system(problem["pdes"], problem["conditions"], coords=problem["coords"], fields=problem["fields"],
                     coefficients=problem["coefficients"])
    P, net = pinn_problem(s, {k: tuple(v) for k, v in problem["bounds"].items()})
    net, _ = train_pinn(P, net, adam_steps=steps, lbfgs_steps=100)

    def model(X):
        with torch.no_grad():
            return net(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy().astype(float)
    return model


@node("veriphysics.grade", "VeriPhysics grade of a model trained elsewhere", category="Validation",
      outputs=[Port("metrics", "metrics")],
      params=[Param("model", "enum", "demo_pinn", choices=["demo_pinn", "torchscript", "onnx", "tool",
                                                           "predictions_csv"],
                    help="demo_pinn trains a small PINN here; the others grade your model"),
              Param("path", "str", "", help="TorchScript / ONNX file, or a CSV of predictions (coords + fields)"),
              Param("tool", "str", "", help="library tool id model(X) -> (N, fields)"),
              Param("problem", "dict", HEAT, help="pdes, conditions, coords, fields, coefficients, bounds, exact"),
              Param("validity", "dict", {}, help="the model's declared validity box"),
              Param("policy", "dict", {}, help="the approval rules: GradePolicy fields over your saved default "
                                               "(empty: your default, else the library's); 'path' names a JSON"),
              Param("steps", "int", 1500, minimum=10, advanced=True)],
      calls="pinneapple_veriphysics.model_grade.grade_model / grade_predictions")
def veriphysics_grade(ctx, inputs, params):
    """Grade the model and report the evidence."""
    import numpy as np

    from pinneapple_physics.symbolic_pde.text import parse_system
    from pinneapple_veriphysics.model_grade import grade_model, grade_predictions
    pb = dict(params["problem"])
    s = parse_system(pb["pdes"], pb.get("conditions", []), coords=pb["coords"], fields=pb["fields"],
                     coefficients=pb.get("coefficients") or {})
    bounds = {k: tuple(v) for k, v in pb["bounds"].items()}
    ref = None
    if pb.get("exact"):
        rng = np.random.default_rng(1)
        lo = np.array([bounds[c][0] for c in s.coords])
        hi = np.array([bounds[c][1] for c in s.coords])
        Xr = lo + rng.random((400, len(lo))) * (hi - lo)
        import sympy as sp
        names = list(s.coords) + list(pb.get("coefficients") or {})
        ex = sp.lambdify(names, sp.sympify(pb["exact"], locals={n: sp.Symbol(n) for n in names}), "numpy")
        val = ex(*[Xr[:, i] for i in range(len(s.coords))], *(pb.get("coefficients") or {}).values())
        ref = (Xr, np.asarray(val, float).reshape(len(Xr), -1))
    from pinneapple_veriphysics.policy import load_policy
    pol = dict(params.get("policy") or {})
    policy = load_policy(pol.pop("path")) if "path" in pol else load_policy(pol or None)
    kw = dict(reference=ref, validity={k: tuple(v) for k, v in (params.get("validity") or {}).items()} or None,
              policy=policy)
    kind = str(params["model"])
    if kind == "predictions_csv":
        import csv
        with open(params["path"], newline="") as f:
            rows = list(csv.DictReader(f))
        P = np.array([[float(r[c]) for c in s.coords] for r in rows])
        V = np.array([[float(r[f]) for f in s.fields] for r in rows])
        g = grade_predictions(P, V, s, bounds, **kw)
    else:
        if kind == "demo_pinn":
            model = _demo_pinn(pb, int(params["steps"]))
            kw["validity"] = kw["validity"] or bounds          # trained on exactly this box
        elif kind == "tool":
            from pinneapple_autolab.study.verify import resolve_model
            model = resolve_model(str(params["tool"]))
        elif kind == "torchscript":
            import torch
            mod = torch.jit.load(str(params["path"])).eval()

            def model(X):
                with torch.no_grad():
                    return mod(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy().astype(float)
        else:
            from pinneapple_analysis.surrogate.model_source import onnx_function
            try:
                model = onnx_function(str(params["path"]), list(s.coords), list(s.fields))
            except RuntimeError as e:                        # onnxruntime is not installed
                from pinneapple_core.workflows import StepFailed
                raise StepFailed(str(e)) from e
        g = grade_model(model, s, bounds, **kw)
    ctx.metric("decision", g["decision"], "; ".join(g["reasons"]) or "every check passes")
    ctx.metric("policy", g["policy"]["name"], "the approval rules used (all fields in the report)")
    ctx.table("policy", ["field", "value"], [[k, str(v)] for k, v in g["policy"].items()])
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in g["checks"]])
    for c in g["checks"]:
        if isinstance(c["value"], (int, float)):
            ctx.metric(c["code"], c["value"])
        elif isinstance(c["value"], list):
            for row in c["value"]:
                ctx.metric(f"{c['code']}.{row.get('equation') or row.get('condition')}", row["rel_rms"])
    ctx.summary(f"{kind}: {g['decision']}")
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "VeriPhysics grade of a model trained elsewhere"
    fb = FlowBuilder(title, "The equations, conditions and reference data a model claims to satisfy; its residuals "
                            "by finite differences of its outputs (any framework), applicability and robustness; "
                            "APPROVED / REVIEW / REJECT with the evidence (pinneapple_veriphysics.model_grade).")
    fb.add("grade", "veriphysics.grade", dict(params), label="VeriPhysics grade")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("grade", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="veriphysics.grade", title="Grade a model trained elsewhere (VeriPhysics)", domain="verification",
        keywords={"veriphysics": 3, "verify my model": 3, "verificar meu modelo": 3, "validar modelo": 2,
                  "external model": 3, "modelo externo": 3, "physicsnemo": 2, "onnx": 1.5, "grade": 1,
                  "physics residual": 2, "resíduo físico": 2, "trust": 1, "confiança": 1},
        min_score=3, nodes=("veriphysics.grade",), build=lambda text, det: build_flow(),
        inputs=("the model (file, tool or predictions)", "the equations and conditions it claims to satisfy",
                "reference data, when any"),
        uncited="the grade follows the PINN validation checks of pinneapple_analysis.validation.pinn; no external "
                "reference",
        description="pinneapple_veriphysics.model_grade.grade_model"))


_register_capabilities()
