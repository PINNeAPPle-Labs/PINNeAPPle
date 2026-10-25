"""Exceptional-point tuning in PT-symmetric dimers: the no-go theorem (Zenodo 23287622) checked symbolically and
numerically, the three routes around it, the hypothesis map as a physics audit, and the figure.

    python examples/non_hermitian/ep_tuning.py      # writes docs/assets/non_hermitian/ep_tuning.png
"""
import os

from pinneapple_physics.non_hermitian import (
    PTChain,
    PTDimer,
    analyze_ep_tuning,
    coupling_classes,
    design_features,
    feedback_coupling,
    route_a_sqrt_coupling,
    route_c_asymmetric,
    theorem_profile,
)
from pinneapple_physics.non_hermitian.viz import ep_tuning_figure
from pinneapple_physics.physics_audit import audit_assumptions

for _key, title, b, b_ba in coupling_classes():
    print(f"\n== {title}")
    print(analyze_ep_tuning(b, b_ba))

print("\n== numerical check (J = 1, G = 0.5)")
d_a = PTDimer(route_a_sqrt_coupling(1.0, 0.5))
ab, ba = route_c_asymmetric(1.0, 0.5)
d_c = PTDimer(ab, ba)
d_l = PTDimer(lambda x: 1.0 + 0.25 * x)
for x in (-1.0, 0.0, 1.0):
    print(f"lam = {x:+.1f}: target {1 + 0.5 * x:.4f} | linear coupling {d_l.ep_gamma_numeric(x) ** 2:.4f} | "
          f"route A {d_a.ep_gamma_numeric(x) ** 2:.4f} | route C {d_c.ep_gamma(x) ** 2:.4f}")
print("route B feedback fixed point at lam = 0.8:", feedback_coupling(1.0, 0.5, 0.8))

print("\n== hypothesis map: does the theorem constrain these designs?")
for name, H, poly in [("PT dimer, polynomial control", PTDimer(1 + 0.3j).hamiltonian(0.5), True),
                      ("route C, non-reciprocal", d_c.hamiltonian(0.5, 1.0), True),
                      ("PT trimer", PTChain(3, 1.0).hamiltonian(0.5), True)]:
    a = audit_assumptions(theorem_profile(), design_features(H, poly))
    print(f"{name}: {a.status}; violated: {[c.name for c in a.violated] or 'none'}")

out = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "non_hermitian", "ep_tuning.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
print("\nwrote", ep_tuning_figure(out))
