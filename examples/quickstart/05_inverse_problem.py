"""05 — Inverse problem: find an unknown physical parameter from measurements.

The library ships inverse-problem presets (the "Inverse Heat Lab"): a real sensor
layout, a forward model, and a known true answer to check the recovered value.

Run from the repository root:

    python examples/quickstart/05_inverse_problem.py
"""
from __future__ import annotations

import pinneapple_analysis.inverse_problems as pinv

# --- 1. Load a preset and turn it into an InverseProblem ----------------
preset = pinv.get_preset("fin_h")
built = preset.build()
problem = pinv.InverseProblem(
    model=built["model"], observations=built["observations"],
    parameters=built["parameters"], validation=built["validation"], name="fin_h",
)
print("problem :", preset.title)
print("unknown :", [p.name for p in built["parameters"]])
print("methods :", {k: v["available"] for k, v in pinv.available_methods(problem).items()})

# --- 2. Solve it ---------------------------------------------------------
result = pinv.solve_inverse(problem, "least_squares")
for p in result["parameters"]:
    lo, hi = p["interval95"]
    print(f"\n{p['name']} = {p['estimate']:.2f} {p['unit']}  (true {p['truth']}, "
          f"error {p['error_pct']:+.2f} %)  95% CI [{lo:.2f}, {hi:.2f}]")
print("truth inside the CI   :", result["truth_recovered"])
print("held-out sensors RMSE :", f"{result['validation']['rmse']:.3f} degC")

# --- 3. Compare every applicable method on the same problem -------------
print()
comparison = pinv.compare_methods(problem, methods=["least_squares", "bayesian"])
print(f"{'method':<20}{'fit RMSE':>10}{'val RMSE':>10}{'seconds':>10}  truth")
for row in comparison["table"]:
    print(f"{row['method']:<20}{row['fit_rmse']:>10.4f}{row['validation_rmse']:>10.4f}"
          f"{row['seconds']:>10.4f}  {row['truth_recovered']}")
print("best:", comparison["best"])
