"""Dataset physics audit (#700). References: the finite-volume Darcy solver (the dataset's own discretisation), an
independent finite-difference solver of the expanded equation ``-a lap u - grad a . grad u = f``, and the Gaussian
random field generator of the Li et al. (2021) Darcy benchmark."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pinneapple_data.audit import (
    DARCY,
    audit_dataset,
    darcy_consistency,
    effective_sample_count,
    leakage,
    near_duplicates,
)
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

N = 32
GRF = {"alpha": 2.0, "tau": 3.0}                                   # Li et al. (2021), section 5.2


def generator(n: int, seed: int) -> np.ndarray:                     # the stated generator of the permeability
    return ob.threshold_coefficient(ob.periodic_grf(n, (N, N), seed=seed, **GRF))


def collapsed_generator(n: int, seed: int) -> np.ndarray:
    """The NVIDIA/physicsnemo#2035 defect: every Fourier coefficient of a sample shares one random value."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    amp = (4 * np.pi ** 2 * (KX ** 2 + KY ** 2) + GRF["tau"] ** 2) ** (-GRF["alpha"] / 2)
    amp[0, 0] = 0.0
    xi = rng.standard_normal(n) + 1j * rng.standard_normal(n)       # one value per sample, not per mode
    g = np.fft.ifft2(xi[:, None, None] * amp[None]).real
    return ob.threshold_coefficient(g)


def solve(a: np.ndarray) -> np.ndarray:
    return np.stack([ob.darcy_2d(ai, 1.0) for ai in a])


def smooth_coefficient(rng, n=N, ghost=False):
    m = n + 2 if ghost else n
    x = (np.arange(m) + 0.5 - (1 if ghost else 0)) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = rng.standard_normal((4, 4))
    return np.exp(0.25 * sum(c[i, j] * np.cos(np.pi * i * X) * np.cos(np.pi * j * Y) for i in range(4) for j in range(4)))


def expanded_fd(a_ghost: np.ndarray, cross: float = 1.0) -> np.ndarray:
    """Finite differences of ``-a lap u - cross * grad a . grad u = 1`` (cell centres, ghost cells ``-u`` for the
    zero boundary). ``cross = 1`` is the documented equation; the #2036 defect scales the cross term wrongly."""
    n = a_ghost.shape[0] - 2
    h = 1.0 / n
    a = a_ghost[1:-1, 1:-1]
    ax = (a_ghost[2:, 1:-1] - a_ghost[:-2, 1:-1]) / (2 * h)
    ay = (a_ghost[1:-1, 2:] - a_ghost[1:-1, :-2]) / (2 * h)
    idx = np.arange(n * n).reshape(n, n)
    R, C, V = [], [], []
    diag = np.zeros((n, n))
    for di, dj, g in ((1, 0, ax), (-1, 0, -ax), (0, 1, ay), (0, -1, -ay)):
        w = -a / h ** 2 - cross * g / (2 * h)
        diag += a / h ** 2
        i0, i1, j0, j1 = max(0, -di), n - max(0, di), max(0, -dj), n - max(0, dj)
        R.append(idx[i0:i1, j0:j1].ravel())
        C.append(idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel())
        V.append(w[i0:i1, j0:j1].ravel())
        bi = slice(n - 1, n) if di == 1 else slice(0, 1) if di == -1 else slice(0, n)
        bj = slice(n - 1, n) if dj == 1 else slice(0, 1) if dj == -1 else slice(0, n)
        diag[bi, bj] -= w[bi, bj]                                  # ghost value -u of the boundary cell
    R.append(idx.ravel())
    C.append(idx.ravel())
    V.append(diag.ravel())
    A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n * n, n * n))
    return spla.spsolve(A, np.ones(n * n)).reshape(n, n)


@pytest.fixture(scope="module")
def darcy_set():
    a_tr, a_te = generator(60, 1), generator(20, 2)
    return {"train": {"a": a_tr, "u": solve(a_tr)}, "test": {"a": a_te, "u": solve(a_te)}}


def test_clean_darcy_set_passes(darcy_set):
    card = audit_dataset(darcy_set, name="darcy-clean", physics="darcy", generator=generator,
                         units={"a": "-", "u": "-"}, constraints=["u > 0", "u < 1 / 24"])
    assert card.decision == "pass", str(card)
    assert card["physics.solve"].value < 1e-10 and card["physics.flux"].value < 1e-10
    assert card["diversity.effective"].evidence["ratio"] > 0.8
    json.dumps(card.to_dict())
    assert "| leakage | pass |" in card.to_markdown()


def test_cross_term_scaled_wrongly_fails_the_residual_check():          # NVIDIA/physicsnemo#2036
    rng = [np.random.default_rng(s) for s in range(12)]
    ag = [smooth_coefficient(r, ghost=True) for r in rng]
    a = np.stack([g[1:-1, 1:-1] for g in ag])
    good = {"train": {"a": a, "u": np.stack([expanded_fd(g, 1.0) for g in ag])}}
    bad = {"train": {"a": a, "u": np.stack([expanded_fd(g, 0.5) for g in ag])}}
    cg = audit_dataset(good, physics=DARCY, max_physics_samples=None)
    cb = audit_dataset(bad, physics=DARCY, max_physics_samples=None)
    assert cg["physics.solve"].status == "pass" and cg["physics.solve"].value < 0.01   # another consistent scheme
    assert cb["physics.solve"].status == "fail" and cb["physics.solve"].value > 0.1
    assert cb["physics.flux"].status == "fail"
    assert cb["physics.solve"].evidence["n_failing"] == 12
    assert cb["physics.solve"].evidence["failing_samples"][0]["split"] == "train"


def test_collapsed_fourier_coefficients_fail_the_diversity_check():     # NVIDIA/physicsnemo#2035
    a = collapsed_generator(60, 1)
    card = audit_dataset({"train": {"a": a, "u": solve(a)}}, physics="darcy", generator=generator)
    assert card["diversity.effective"].status == "fail"
    assert card["diversity.effective"].value < 0.1                     # far below the stated generator's
    assert card["diversity.duplicates"].status == "fail"                # only two distinct fields (sign of xi)
    assert card["physics.solve"].status == "pass"                       # the data still solve the PDE


def test_five_percent_leakage_fails(darcy_set):
    d = {"train": dict(darcy_set["train"]), "test": {k: v.copy() for k, v in darcy_set["test"].items()}}
    d["test"]["a"][3] = d["train"]["a"][17]                             # 1 of 20 test samples = 5 %
    d["test"]["u"][3] = d["train"]["u"][17]
    card = audit_dataset(d, physics="darcy", generator=generator)
    lk = card["leakage"]
    assert lk.status == "fail" and lk.value == pytest.approx(0.05)
    assert lk.evidence["pairs"][0][:2] == (3, 17)


def test_building_blocks_against_closed_forms():
    rng = np.random.default_rng(0)
    pattern = rng.standard_normal(50)
    assert effective_sample_count(np.outer(rng.standard_normal(30), pattern)) == pytest.approx(1.0)
    iso = rng.standard_normal((2000, 5))
    assert effective_sample_count(iso) == pytest.approx(5.0, rel=0.05)  # isotropic: every direction counts
    x = rng.standard_normal((40, 8))
    x[7] = x[3]
    assert near_duplicates(x)["exact"] == 2
    lk = leakage(x, np.vstack([x[:2] + 1e-9, rng.standard_normal((8, 8))]))
    assert lk["n_flagged"] == 2 and lk["fraction"] == pytest.approx(0.2)
    a = generator(1, 5)[0]
    r = darcy_consistency(a, ob.darcy_2d(a, 1.0))
    assert r["solve"] == 0.0 and r["flux"] < 1e-12 and r["bc"] < 0.02


def test_finite_constraints_and_units():
    d = {"train": {"x": np.array([1.0, 2.0, np.nan]), "y": np.array([2.0, 4.0, 6.0])},
         "test": {"x": np.array([np.inf]), "y": np.array([1.0])}}
    card = audit_dataset(d, inputs=["x", "y"], units={"x": "m", "y": "furlongs per fortnight"},
                         expected_units={"x": "s"}, constraints=["y == 2 * x", "y > 0"])
    assert card["finite"].status == "fail" and "test/x" in card["finite"].summary
    assert card["constraint: y == 2 * x"].status == "fail"              # the NaN row and the test row
    assert card["units"].status == "fail" and "expected s" in card["units"].summary
    assert card["physics"].status == "not-applicable"
    nan_only = audit_dataset({"train": {"x": np.array([1.0, np.nan])}}, missing_documented=["x"])
    assert nan_only["finite"].status == "warn" and "documented" in nan_only["finite"].summary


LOCAL_ZOO = ["airfoil_rans_cst_re4e6", "heatsink_forced_convection", "heatsink_natural_convection",
             "rover_contact_grc1_viper", "single_wheel_tests"]


@pytest.mark.parametrize("name", LOCAL_ZOO + ["burgers_1d_li2021", "darcy_2d_li2021"])
def test_zoo_datasets_pass_the_audit(name):
    from pinneapple_data.audit import audit_zoo_dataset
    from pinneapple_hub.zoo._artifacts import ArtifactUnavailableError
    try:
        card = audit_zoo_dataset(name)
    except ArtifactUnavailableError as e:                               # Hub-only sets need the network
        pytest.skip(f"{name} not available offline: {e}")
    assert card.passed, str(card)
    assert not [c for c in card.checks if c.status == "fail"]


@pytest.mark.parametrize("name", LOCAL_ZOO)
def test_stored_audit_cards_are_current(name):
    from pinneapple_hub.zoo import datasets as zd
    stored = zd.audit_card(name)
    assert stored is not None and stored["schema"] == "dataset-audit/v1"
    fresh = zd.audit(name, offline=True).to_dict()
    assert stored["decision"] == fresh["decision"]
    assert [(c["id"], c["status"]) for c in stored["checks"]] == [(c["id"], c["status"]) for c in fresh["checks"]]
    assert "## Physics audit" in zd.info(name).to_markdown()


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6])
def test_exact_copies_are_found_whatever_the_rounding(offset):
    """|a|^2 + |b|^2 - 2 a.b put exact copies of 1000-wide rows up to 1.4 apart (offset 1e6) and, on some CI CPUs,
    ~1e-7 apart even for small rows, so near_duplicates and leakage missed them when copies were half the set."""
    rng = np.random.default_rng(1)
    x = offset + rng.normal(size=(6, 1000))
    x = np.vstack([x, x[:2], x[:1]])                                  # rows 0 and 1 copied: 5 rows with a copy
    r = near_duplicates(x)
    assert r["exact"] == 5 and r["near"] == 5
    assert all(d == 0.0 for *_, d in r["pairs"])
    lk = leakage(x[:6], x[:2])
    assert lk["n_flagged"] == 2 and lk["fraction"] == 1.0
