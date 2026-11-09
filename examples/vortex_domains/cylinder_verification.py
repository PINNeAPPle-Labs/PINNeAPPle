"""What the viscous vortex domains solver reproduces: four exact or near-exact properties, then the cylinder.

The first four checks are properties of the machinery -- a panel integral against a quadrature, the potential flow
past a cylinder, the attached sheet of a spinning cylinder, the diffusion drift -- and take a second. The cylinder
run afterwards is the physics: its drag at Re = 40 is compared with Dennis & Chang (1970), ``Cd = 1.3375``.

    python cylinder_verification.py [--full]
"""
from __future__ import annotations

import argparse


def run(smoke: bool = True, out_dir=None) -> dict:
    """Run the checks (and, unless ``smoke``, the cylinder) and return the metrics."""
    from pinneapple_simulation.vortex_domains import verification
    from pinneapple_simulation.vortex_domains.cases import cylinder_case
    from pinneapple_simulation.vortex_domains.study import run_study

    checks = verification.run(quick=True)
    print("\n".join(f"{c['name']:<18} {'OK  ' if c['ok'] else 'FAIL'} {c['detail']}" for c in checks))
    metrics = {f"exact_{c['name']}": float(c["ok"]) for c in checks}

    result = run_study(cylinder_case(re=40.0, panels=32 if smoke else 96, t_end=0.6 if smoke else 10.0))
    print(result.summary())
    for check in result.checks:
        print(f"  {check['name']:<16} {'OK  ' if check['ok'] else 'FAIL'} {check['detail']}")
    metrics["cylinder_cd"] = result.coefficients["cd"]
    metrics["cylinder_cl"] = result.coefficients["cl"]
    metrics["cylinder_panels"] = 32 if smoke else 96
    if out_dir is not None:
        result.save(out_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="96 panels and t_end = 10 instead of a short run")
    parser.add_argument("--out", default=None, help="folder for study.json and the time series")
    args = parser.parse_args()
    print(run(smoke=not args.full, out_dir=args.out))
