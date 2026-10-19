"""Metamorphic relations (#698). References: the finite-volume Darcy solver (linear in the source, invariant under
the symmetries of the square, scale-free in (a, f)), and the physics the zoo models were trained on."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.metamorphic import (
    Invariance,
    Monotonicity,
    ResolutionConsistency,
    Similarity,
    Superposition,
    Symmetry,
    box_sampler,
    check_relations,
    field_sampler,
    reflect_field,
    relations_from_spec,
    rotate_field,
)
from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d

N = 16


def smooth(rng: np.random.Generator, n: int = N, k: int = 4) -> np.ndarray:
    """Smooth random field: a few low Fourier modes."""
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = rng.standard_normal((k, k))
    return sum(c[i, j] * np.cos(np.pi * i * X) * np.cos(np.pi * j * Y) for i in range(k) for j in range(k))


A = np.exp(0.5 * smooth(np.random.default_rng(42)))                  # fixed positive coefficient


def darcy_source(f_batch: np.ndarray) -> np.ndarray:                 # f -> u with the coefficient A
    return np.stack([darcy_2d(A, f) for f in f_batch])


def darcy_pair(x: np.ndarray) -> np.ndarray:                         # (a, f) channels -> u
    return np.stack([darcy_2d(s[0], s[1]) for s in x])


pairs = field_sampler(lambda rng: np.stack([np.exp(0.5 * smooth(rng)), smooth(rng)]))
sources = field_sampler(smooth)


def test_reference_solver_obeys_superposition_to_round_off():
    rep = check_relations(darcy_source, [Superposition("darcy-source", tol=1e-10)], sources, n=8)
    assert rep.passed and rep["darcy-source"].worst < 1e-12


def test_reference_solver_symmetries_similarity_and_resolution():
    rels = [
        Symmetry("reflect-x", tol=1e-10, transform_in=lambda x: np.flip(x, axis=2),
                 transform_out=lambda x, y: np.flip(y, axis=1)),
        Symmetry("rotate-90", tol=1e-10, transform_in=lambda x: np.rot90(x, 1, axes=(2, 3)),
                 transform_out=lambda x, y: rotate_field(1)(y)),
        Similarity("scale-a-and-f", tol=1e-10, transform_in=lambda x: 3.7 * x),     # u(c a, c f) = u(a, f)
    ]
    rep = check_relations(darcy_pair, rels, pairs, n=6)
    assert rep.passed, rep.summary()

    def coarse(x):                                                   # cell averages on a 2x coarser grid
        n = x.shape[-1]
        return x.reshape(len(x), 2, n // 2, 2, n // 2, 2).mean(axis=(3, 5))
    fine = field_sampler(lambda rng: np.stack([np.exp(0.3 * smooth(rng, 32)), 1.0 + 0.3 * smooth(rng, 32)]))
    res = ResolutionConsistency("mean-pressure", tol=0.02, restrict=coarse)
    rep = check_relations(darcy_pair, [res], fine, n=4)
    assert rep.passed and rep["mean-pressure"].worst > 1e-6        # second-order difference, not zero


def test_broken_scaling_and_injected_asymmetry_fail_with_a_counterexample():
    def broken(x):                                                   # a hard-coded unit source: ignores f
        x = np.asarray(x)
        return darcy_pair(np.stack([x[:, 0], np.ones_like(x[:, 1])], axis=1))
    rep = check_relations(broken, [Similarity("scale-a-and-f", tol=1e-3, transform_in=lambda x: 2.0 * x)], pairs, n=4)
    r = rep["scale-a-and-f"]
    assert not r.passed and r.worst > 0.4                           # u halves when a doubles and f does not
    assert "input" in r.counterexample and r.counterexample["violation"] == pytest.approx(r.worst)
    json.dumps(rep.to_dict())

    def tilted(x):
        u = darcy_pair(x)
        return u + 0.05 * np.abs(u).max() * np.linspace(0, 1, N)[None, :, None]
    rel = Symmetry("reflect-x", tol=1e-3, transform_in=lambda x: np.flip(x, axis=2),
                   transform_out=lambda x, y: reflect_field(0)(y))
    r = check_relations(tilted, [rel], pairs, n=4)["reflect-x"]
    assert not r.passed and r.worst > 0.01


def _train_fno(steps: int = 300):
    torch = pytest.importorskip("torch")
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    F = np.stack([smooth(rng) for _ in range(96)])
    U = darcy_source(F)
    s = float(np.abs(U).max())
    net = FNO2d(1, 1, width=16, modes1=6, modes2=6, layers=3)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt = torch.tensor(F[:, None], dtype=torch.float32)
    Yt = torch.tensor(U[:, None] / s, dtype=torch.float32)
    for _ in range(steps):
        idx = torch.randint(0, len(Xt), (16,))
        loss = ((net(Xt[idx]).y - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()

    def model(f):
        with torch.no_grad():
            return net(torch.tensor(np.asarray(f)[:, None], dtype=torch.float32)).y[:, 0].numpy() * s
    test = np.stack([smooth(np.random.default_rng(100 + i)) for i in range(8)])
    rel = np.linalg.norm(model(test) - darcy_source(test)) / np.linalg.norm(darcy_source(test))
    return model, rel


def test_fno_violates_superposition_measurably():
    model, rel_err = _train_fno()
    assert rel_err < 0.3                                             # it did learn the map
    rep = check_relations(model, [Superposition("darcy-source", tol=1e-3)], sources, n=16)
    r = rep["darcy-source"]
    assert not r.passed and r.worst > 1e-2                           # a GELU network is not linear in f
    ref = check_relations(darcy_source, [Superposition("darcy-source", tol=1e-3)], sources, n=16)
    assert r.worst > 1e8 * ref["darcy-source"].worst


def test_monotonicity_and_invariance_on_closed_forms_and_hypothesis_search():
    def peak(x):                                                     # peak temperature of a heated slab
        return {"T": x["T0"] + x["q"] * x["L"] ** 2 / (8 * x["k"])}
    box = box_sampler({"q": (1e3, 1e5), "L": (0.01, 0.1), "k": (1.0, 400.0), "T0": (250.0, 350.0)})
    rels = relations_from_spec([
        {"kind": "monotonic", "name": "more-heat-hotter", "input": "q", "output": "T", "direction": "increasing"},
        {"kind": "monotonic", "name": "better-conductor-cooler", "input": "k", "output": "T", "direction": "decreasing"},
        {"kind": "invariant", "name": "q-L2-group", "inputs": {"q": "4 * q", "L": "L / 2"}, "output": "T",
         "tol": 1e-12},
    ])
    rep = check_relations(peak, rels, box, n=64, search="hypothesis")
    assert rep.passed, rep.summary()

    def wrong(x):                                                    # sign error: more conductivity, hotter
        return {"T": x["T0"] + x["q"] * x["L"] ** 2 * x["k"] / 8e4}
    rep = check_relations(wrong, rels[1:2], box, n=32, search="hypothesis")
    r = rep["better-conductor-cooler"]
    assert not r.passed and r.counterexample["after"][0] > r.counterexample["before"][0]
    assert r.search == "random+hypothesis"


def test_monotonicity_needs_a_direction():
    with pytest.raises(ValueError):
        Monotonicity("bad", input="q", direction=0)
    with pytest.raises(ValueError, match="unknown kind"):
        relations_from_spec([{"kind": "telepathy"}])
    assert Invariance("ok", transform_in=lambda x: x).kind == "invariance"


@pytest.fixture(scope="module")
def zoo():
    import pinneapple as pp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {n: pp.zoo.load(n) for n in ("airfoil_cst_rans_mlp", "heatsink_forced_mlp")}


@pytest.mark.parametrize("name", ["airfoil_cst_rans_mlp", "heatsink_forced_mlp"])
def test_zoo_models_pass_the_relations_their_physics_implies(zoo, name):
    from pinneapple_hub.zoo.relations import check_zoo_relations
    rep = check_zoo_relations(zoo[name], n=128)
    assert rep.passed, rep.summary()
    assert len(rep.results) >= 4


def test_zoo_model_with_injected_asymmetry_fails(zoo):
    from pinneapple_hub.zoo.relations import check_zoo_relations
    m = zoo["airfoil_cst_rans_mlp"]

    class Biased:                                                    # a lift offset breaks the camber mirror
        name = m.name

        def predict(self, x):
            y = dict(m.predict(x))
            y["Cl"] = np.asarray(y["Cl"]) + 0.03
            return y
    rep = check_zoo_relations(Biased(), n=64)
    r = rep["camber-mirror-lift"]
    assert not r.passed and r.worst == pytest.approx(6.0, rel=0.25)  # |-(Cl+0.03) - (-Cl+0.03)| / 0.01 = 6
    assert set(r.counterexample["input"]) >= {"M0", "alpha_deg"}


def test_relation_expressions_cannot_run_code():
    from pinneapple_analysis.validation.metamorphic import _expr_map
    f = _expr_map({"q": "4 * q", "a": "-sin(a) + pi"})
    out = f({"q": np.array([1.0, 2.0]), "a": np.array([0.0, 0.0])})
    assert np.allclose(out["q"], [4, 8]) and np.allclose(out["a"], np.pi)
    for bad in ("().__class__.__base__.__subclasses__()", "q[0]", "__import__('os')", "open('x')",
                "sin(q, out=q)"):
        with pytest.raises(ValueError):
            _expr_map({"q": bad})({"q": np.array([1.0])})
