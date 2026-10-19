"""Run an external model (for instance the FNO or MeshGraphNet of a PhysicsNeMo experiment) through the hardened
defence of the validation arena (#707), and compare each of its outputs with the reference solver.

    python examples/validation_arena/external_model.py --problem darcy_2d --predict my_models.py:fno_predict

``my_models.py:fno_predict`` is ``predict(inputs, params) -> field`` on the problem's grid (Darcy: ``inputs["a"]``
of shape (32, 32) -> u (32, 32); Burgers: ``inputs["u0"]`` (128,) -> u (21, 128)). Without ``--predict`` the
reference solver itself is audited (no escape, no false alarm expected). Everything is in
``pinneapple_analysis.validation.arena.audit_model``; this script only calls it.
"""
import argparse
import importlib.util

from pinneapple_analysis.validation.arena import audit_model
from pinneapple_analysis.validation.fault_injection import get_problem

ap = argparse.ArgumentParser()
ap.add_argument("--problem", default="darcy_2d", choices=["darcy_2d", "burgers_1d"])
ap.add_argument("--predict", help="file.py:function")
ap.add_argument("--n-cases", type=int, default=20)
a = ap.parse_args()

if a.predict:
    file, fn = a.predict.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("external_model", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict = getattr(mod, fn)
else:
    p = get_problem(a.problem)
    predict = p.predict_reference

audit = audit_model(a.problem, predict, n_cases=a.n_cases)
print(audit.to_markdown())
print("escapes (case seeds):", audit.escapes)
print("accurate outputs rejected (case seeds):", audit.false_alarms)
