"""Surrogates of the axial-flux machine (``machines3d.afpm_surrogate``): the design space, the physics checks and the
certification, on an analytic stand-in for the 3-D solve (the real campaign is ``examples/electrodynamics/25``)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pinneapple_autolab import FunctionSimulator  # noqa: E402
from pinneapple_autolab.study import benchmark as B  # noqa: E402
from pinneapple_autolab.study import dataset as D  # noqa: E402
from pinneapple_autolab.study.campaign import plan_doe, run_campaign  # noqa: E402
from pinneapple_simulation.electromagnetics.machines3d import afpm_surrogate as S  # noqa: E402


def _toy(p):
    """Flux linkage rising with the magnet and falling with the gap; positive cogging; ratios near the measured."""
    lm, g, a, s = p["magnet_thickness"], p["gap"], p["magnet_arc"], p["slot_fraction"]
    link = 1.3e-3 * a * lm / (lm + 1.05 * (g + 0.004))
    cog = 10.0 * (1.0 + 0.5 * (s - 0.5)) * lm / 0.004
    return {"cogging_pkpk_3d": cog, "linkage_3d": link, "cogging_pkpk_q3d": 1.12 * cog, "linkage_q3d": 1.02 * link,
            "ratio_cogging": 1.12 + 0.02 * (g - 0.001) / 0.0008, "ratio_linkage": 1.02 + 0.005 * a}


@pytest.fixture(scope="module")
def board_ds(tmp_path_factory):
    plan = plan_doe(S.space(), 36, "maximin", seed=0)
    recs = run_campaign(FunctionSimulator(_toy, S.OUTPUTS, name="toy"), plan["cases"],
                        str(tmp_path_factory.mktemp("camp")), log=lambda s: None)
    names = list(S.VARIABLES)
    ds = D.build_dataset(recs, names, list(S.OUTPUTS), bounds={k: v[:2] for k, v in S.VARIABLES.items()})
    j = names.index("magnet_thickness")
    ds = ds.with_split(D.region_split(ds, holdout=lambda X: X[:, j] > 0.0055, seed=0))
    measured = {t: (float(ds.y(t).min()), float(ds.y(t).max())) for t in ("ratio_cogging", "ratio_linkage")}
    board = B.benchmark(ds, ["linkage_3d", "ratio_cogging"], None, scalar_families=("gp", "quadratic"), k=4,
                        physics=S.physics_checks(measured), seed=0, log=lambda s: None)
    return board, ds


def test_space_matches_the_variables():
    sp = S.space()
    assert [p.name for p in sp.parameters] == list(S.VARIABLES)
    assert [(p.low, p.high) for p in sp.parameters] == [v[:2] for v in S.VARIABLES.values()]


def test_simulator_refuses_designs_outside_the_bounds():
    sim = S.simulator()
    assert sim.outputs == list(S.OUTPUTS) or tuple(sim.outputs) == S.OUTPUTS
    bad = sim.checker({"magnet_arc": 0.95, "slot_fraction": 0.5, "gap": 0.001, "magnet_thickness": 0.004})
    assert len(bad) == 1 and "magnet_arc" in bad[0]
    assert sim.checker({"magnet_arc": 0.75, "slot_fraction": 0.5, "gap": 0.001, "magnet_thickness": 0.004}) == []


def test_physics_checks_pass_a_physical_predictor_and_fail_a_wrong_one():
    checks = S.physics_checks(measured_ratios={"ratio_cogging": (1.05, 1.2)}, n=64, seed=0)
    assert set(checks) >= {"linkage_3d", "cogging_pkpk_3d", "ratio_cogging"}
    names = list(S.VARIABLES)

    def physical(U):                       # rises with the magnet (index 3), falls with the gap (index 2)
        return 1.0 + U[:, names.index("magnet_thickness")] - 0.5 * U[:, names.index("gap")], np.zeros(len(U))

    def wrong(U):
        return 1.0 - U[:, names.index("magnet_thickness")], np.zeros(len(U))
    assert all(c(physical).status == "supports" for c in checks["linkage_3d"])
    assert [c(wrong).status for c in checks["linkage_3d"]][1] == "contradicts"
    assert checks["ratio_cogging"][0](lambda U: (np.full(len(U), 1.1), 0 * U[:, 0])).status == "supports"
    assert checks["ratio_cogging"][0](lambda U: (np.full(len(U), 1.4), 0 * U[:, 0])).status == "contradicts"


def test_certify_approves_a_good_surrogate_and_records_the_evidence(board_ds):
    from pinneapple_veriphysics.ledger import EvidenceLedger
    board, ds = board_ds
    led = EvidenceLedger()
    c = S.certify(board, ds, "linkage_3d", ledger=led)
    assert c["decision"] == "APPROVED", c["reasons"]
    names = {x["name"] for x in c["checks"]}
    assert {"test_r2", "test_coverage90", "linkage_3d_rises_with_magnet_thickness"} <= names
    assert all(v == "passed" for k, v in c["ledger_status"].items() if not k.startswith("extrapolation")), c
    assert len(led.to_dict()["entries"]) >= len(c["checks"])


def test_certify_rejects_when_a_physics_check_fails(board_ds):
    board, ds = board_ds
    fam = board.select()["choice"]["linkage_3d"]
    key = f"linkage_3d/{fam}"
    saved = board.physics[key]
    board.physics[key] = [*saved, {"name": "made_up", "passed": False, "value": 0.0, "limit": 1.0, "reason": "test"}]
    try:
        c = S.certify(board, ds, "linkage_3d", family=fam)
        assert c["decision"] == "REJECT" and c["ledger_status"]["made_up"] == "failed"
    finally:
        board.physics[key] = saved
