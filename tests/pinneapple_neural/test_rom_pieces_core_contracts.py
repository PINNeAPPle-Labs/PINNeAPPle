"""Core coverage (#1129, toward 1.0) for the ROM building blocks: steppers, study helpers, contracts, views, the
toy systems, the deep UQ ROM, the Koopman operator, the base class and the registry.

Exact references:
* two travelling waves are an exact rank-4 linear system. The steppers centre the POD and fit a linear map with no
  constant term, so they are exact when the training mean is invariant: over one full common period of the waves,
  a POD-DMD or POD-OpInf step reproduces the next snapshot and the DMD eigenvalues lie on the unit circle (over a
  partial period the centred dynamics are affine and a ~4 % error remains);
* field scales are the RMS of each field; the error of a prediction scaled by 1.01 is exactly 1 %;
* a "stable" Koopman operator is non-expansive; a "block" one is block-diagonal.
"""
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_data.snapshots import snapshots_from_arrays  # noqa: E402
from pinneapple_neural.architectures.rom import base as RB  # noqa: E402
from pinneapple_neural.architectures.rom import contracts as RC  # noqa: E402
from pinneapple_neural.architectures.rom import deep_uq_rom as DU  # noqa: E402
from pinneapple_neural.architectures.rom import koopman as KO  # noqa: E402
from pinneapple_neural.architectures.rom import registry as RR  # noqa: E402
from pinneapple_neural.architectures.rom import steppers as ST  # noqa: E402
from pinneapple_neural.architectures.rom import study as SY  # noqa: E402
from pinneapple_neural.architectures.rom import toy_systems as TS  # noqa: E402
from pinneapple_neural.architectures.rom import views as VW  # noqa: E402

X1 = np.linspace(0, 2 * np.pi, 120)


def _waves(n=90, dt=2 * np.pi / 60):                        # 60 steps: one common period of both waves
    t = np.arange(n) * dt
    return np.array([1 + np.sin(X1 - 2 * s) + 0.3 * np.sin(2 * X1 + 3 * s) for s in t])


# ------------------------------------------------------------------------------------------------ steppers
def test_pod_dmd_step_reproduces_the_next_snapshot_of_a_linear_system():
    U = _waves()
    step, eig = ST.pod_dmd_stepper(U[:60], r=4)
    assert abs(eig - 1.0) < 1e-9                                                 # neutral: travelling waves
    assert np.allclose(step(U[70]), U[71], atol=1e-9)                            # exact, beyond the window


def test_pod_opinf_step_reproduces_the_next_snapshot():
    U = _waves()
    step, rank = ST.pod_opinf_stepper(U[None, :60], energy=1 - 1e-12)
    assert rank == 4 and np.allclose(step(U[70]), U[71], atol=1e-7)                # exact up to the ridge


# ------------------------------------------------------------------------------------------------ study helpers
def test_default_ranks():
    assert SY.default_ranks(10, 300) == [1, 2, 4, 8, 9]                          # powers of two up to n - 1
    sv = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
    r = SY.default_ranks(50, 5, sv)
    assert r == sorted(set(r)) and max(r) == 5 and 2 in r                       # 99 % of the energy: rank 2


def test_field_scales_are_the_rms_of_each_field():
    a, b = np.random.default_rng(0).normal(size=(6, 4)), 1e3 * np.random.default_rng(1).normal(size=(6, 3))
    s = snapshots_from_arrays({"d": a, "s": b}, params=np.arange(6.0)[:, None])
    sc = SY.field_scales(s, [0, 1, 2])
    assert np.allclose(sc[:4], np.sqrt(np.mean(a[:3] ** 2))) and np.allclose(sc[4:], np.sqrt(np.mean(b[:3] ** 2)))


# ------------------------------------------------------------------------------------------------ contracts
@pytest.fixture(scope="module")
def bump_study():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 1, (40, 2))
    X = np.array([np.exp(-((X1 - 3 - p[0]) ** 2) / (0.3 + p[1])) for p in P])
    s = snapshots_from_arrays({"u": X}, params=P, param_names=["shift", "width"])
    return s, SY.rom_study(s, ranks=[4, 8], methods=["nearest", "pod_rbf"], timing_repeats=1)


def test_study_contracts(bump_study):
    s, res = bump_study
    checks = RC.rom_checks(res, tolerance=0.05)
    beat = next(c for c in checks if c.name.startswith("best ROM beats"))
    assert beat.status == "PASS" and beat.value > 1                              # ratio baseline / best
    sr = RC.study_result(res, study_id="s1", title="bump", tolerance=0.05)
    assert sr.id == "s1" and len(sr.table.columns["method"]) == len(res.rows)
    rm = SY.fit_reduced_model(s, res.split.train, "pod_rbf", rank=8)
    ref = RC.model_ref(rm, name="bump-rom", uri="rom.npz", payload=b"weights")
    assert ref.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert [v.name for v in ref.inputs] == ["shift", "width"]


# ------------------------------------------------------------------------------------------------ views
def test_zones_and_prediction_view_on_a_small_mesh():
    xs = np.linspace(0, 1, 6)
    P = np.array([[x, y, 0.0] for x in xs for y in xs])
    tris = [t for i in range(5) for j in range(5) for t in ([i * 6 + j, i * 6 + j + 6, i * 6 + j + 7],
                                                          [i * 6 + j, i * 6 + j + 7, i * 6 + j + 1])]
    U = np.array([np.sin(np.pi * P[:, 0] * k) for k in range(1, 5)])
    s = snapshots_from_arrays({"u": U}, locations={"point": P}, cells=[("triangle", np.array(tris))],
                              params=np.arange(4.0)[:, None], meta={"node_sets": {"clamp": {"nodes": [0, 1, 2],
                                                                                            "kind": "bc"}}})
    z = VW.zones(s, P, 0.1)
    assert set(z) == {"clamp zone", "rest of the body"} and len(z["clamp zone"]) + len(z["rest of the body"]) == 36
    fv, info = VW.prediction_view(s, 1, U[1] * 1.01)
    assert info["errors"]["u"]["rel_l2"] == pytest.approx(0.01, rel=1e-9)        # exact: 1 % scaling
    assert info["regions"] == {"clamp zone": 3, "rest of the body": 33}


# ------------------------------------------------------------------------------------------------ models
def test_toy_linear_system_is_contractive_and_sequences_follow_it():
    A = TS.make_linear_system(6, seed=1)
    assert float(torch.linalg.svdvals(A).max()) == pytest.approx(0.95, rel=1e-5)
    seq = TS.generate_sequences(3, 5, 6, A, noise=0.0)
    assert seq.shape == (3, 5, 6) and torch.allclose(seq[:, 1], seq[:, 0] @ A.T, atol=1e-6)
    assert torch.equal(TS.generate_sequences(3, 5, 6, A, seed=7), TS.generate_sequences(3, 5, 6, A, seed=7))


def test_deep_uq_rom_predicts_the_next_step_with_a_variance():
    m = DU.DeepUQROM(r=3, hidden=16, layers=1)
    a = torch.randn(2, 5, 3)
    out = m(a, a_true=a, return_loss=True)
    assert isinstance(out, RB.ROMOutput) and isinstance(m, RB.ROMBase)
    assert out.y.shape == (2, 4, 3) and "total" in out.losses and torch.isfinite(out.losses["total"])
    assert RB.ROMBase.mse(torch.ones(3), torch.zeros(3)).item() == 1.0


def test_koopman_operator_structures():
    torch.manual_seed(0)
    st = KO.KoopmanOperator(6, structure="stable")
    z = torch.randn(50, 6)
    assert torch.all(st(z).norm(dim=1) <= z.norm(dim=1) + 1e-5)                 # non-expansive
    bk = KO.KoopmanOperator(6, structure="block", num_blocks=3)
    e = torch.zeros(1, 6)
    e[0, 0] = 1.0
    assert torch.all(bk(e)[0, 2:] == 0)                                          # block 1 maps into block 1
    assert KO.KoopmanOperator(4)(torch.randn(2, 4)).shape == (2, 4)


def test_rom_registry(monkeypatch):
    from pinneapple_neural.architectures.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "_REGISTRY", {})
    RR.register_into_global()
    assert {"pod"} <= set(ModelRegistry._REGISTRY) and all(s.family == "rom" for s in ModelRegistry._REGISTRY.values())
