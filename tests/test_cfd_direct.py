"""The campaign's cases and its pre-registered verdict rule (docs/research/cfd_direct_from_stl.md), on literal run
records."""
import numpy as np
import pytest

from pinneapple_simulation.cfd_direct.cases import (
    SCHAFER_TUREK,
    BackwardStep,
    Sphere,
    clift_grace_weber,
    cylinder_polygon,
    reattachment,
    schiller_naumann,
)
from pinneapple_simulation.cfd_direct.verdicts import (
    GATES,
    operating_points,
    profile_errors,
    verdict,
)


def test_published_values_and_geometry():
    assert SCHAFER_TUREK.Re == pytest.approx(20.0)
    assert clift_grace_weber(20) == pytest.approx(2.735, abs=1e-3)
    assert clift_grace_weber(100) == pytest.approx(1.087, abs=1e-3)
    assert schiller_naumann(100) == pytest.approx(1.092, abs=1e-3)
    with pytest.raises(ValueError):
        clift_grace_weber(5)
    V = cylinder_polygon(512)
    assert np.allclose(np.hypot(V[:, 0] - 0.2, V[:, 1] - 0.2), 0.05)
    s = BackwardStep(100)
    assert s.H / s.h == pytest.approx(1.942, abs=1e-3) and s.x1_ref == 2.96
    y = np.linspace(s.S, s.H, 2001)
    assert np.trapezoid(s.inflow(y), y) / s.h == pytest.approx(s.U, rel=1e-6)
    assert Sphere(50).Cd_ref == pytest.approx(clift_grace_weber(50))


def test_reattachment_skips_the_corner_eddy():
    x = np.linspace(0, 10, 2001)
    u = np.where(x < 0.05, 1e-4, np.where(x < 3.0, -np.sin(np.pi * x / 3), 0.1))
    assert reattachment(x, u) == pytest.approx(3.0, abs=0.01)
    assert np.isnan(reattachment(x, np.ones_like(x)))


def _run(method, n, total, **err):
    return {"method": method, "n": n, "ok": True, "times": {"total_s": total, "solve_s": total / 2},
            "profiles": {"a": [1.0, 2.0]}, **{f"err_{k}": v for k, v in err.items()}}


def test_verdict_rule():
    runs = [_run("reference", 8, 2.0, Cd=0.02), _run("reference", 16, 5.0, Cd=0.005),
            _run("reference", 32, 20.0, Cd=0.001),
            _run("fast", 4, 1.0, Cd=0.035), _run("fast", 8, 3.0, Cd=0.01), _run("fast", 2, 0.5, Cd=0.08),
            _run("slow", 4, 4.0, Cd=0.01), _run("never", 4, 0.1, Cd=0.5)]
    profile_errors(runs)
    assert all(r["err_profile"] == 0.0 for r in runs)
    op = operating_points(runs, "C2", 20)
    assert op["gate"] == GATES("C2") and op["gate_passed"] and op["reference"]["n"] == 8
    v = verdict(op, "fast")              # R*: n = 8 (2 % <= 4 %, cheapest); X*: n = 4 (3.5 % <= 2 % + 2 pp), 2x faster
    assert v["verdict"] == "validated" and v["speedup"] == pytest.approx(2.0) and v["n"] == 4
    assert verdict(op, "slow")["verdict"] == "refuted"                         # S = 0.5
    assert verdict(op, "never")["verdict"] == "refuted"
    assert verdict(op, "absent")["verdict"] == "not tested"
