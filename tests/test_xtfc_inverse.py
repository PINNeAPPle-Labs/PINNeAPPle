"""X-TFC parameter discovery (pinneapple_simulation.numerical_solvers.xtfc_inverse) and the epidemiological models."""
import dataclasses

import numpy as np
import pytest
import torch

from pinneapple_physics.paper_problems_epi import EPI_MODELS, epi_model, synthetic_data
from pinneapple_simulation.numerical_solvers.xtfc_general import (
    Constraint,
    XTFCProblem,
    activation_derivative,
)
from pinneapple_simulation.numerical_solvers.xtfc_inverse import solve_xtfc_inverse


def test_arctan_derivatives_match_finite_differences():
    z = np.linspace(-3, 3, 13)
    h = 1e-5
    for n in range(1, 4):
        fd = (activation_derivative("arctan", z + h, n - 1) - activation_derivative("arctan", z - h, n - 1)) / (2 * h)
        assert np.allclose(activation_derivative("arctan", z, n), fd, atol=1e-6)
    with pytest.raises(ValueError):
        activation_derivative("arctan", z, 4)


@pytest.mark.parametrize("name", sorted(EPI_MODELS))
def test_models_conserve_the_population_with_their_stated_rates(name):
    m = epi_model(name)
    _, clean, _ = synthetic_data(m, 0.0)
    total = clean.sum(axis=1)
    assert np.allclose(total, total[0], rtol=1e-9)


def test_noise_is_bounded_by_delta_and_seeded():
    m = epi_model("sir")
    _, c, y1 = synthetic_data(m, 2.0, seed=3)
    _, _, y2 = synthetic_data(m, 2.0, seed=3)
    assert np.array_equal(y1, y2) and np.max(np.abs(y1 - c)) <= 2.0 and np.max(np.abs(y1 - c)) > 1.0


def test_inverse_recovers_parameters_of_a_logistic_ode():
    # u' = a u (1 - u / K), u(0) = 1; true a = 0.8, K = 10 (not a paper target)
    t = np.linspace(0, 8, 60)
    a, K = 0.8, 10.0
    u = K / (1 + (K - 1) * np.exp(-a * t))
    prob = XTFCProblem([(0.0, 8.0)], {"u": [Constraint(0, 0.0)]},
                       {"u": lambda X: torch.ones(X.shape[0], dtype=X.dtype)}, lambda X, D: [],
                       [("u", (0,)), ("u", (1,))])

    def res(X, D, P):
        return D[("u", (1,))] - P["a"] * D[("u", (0,))] * (1 - D[("u", (0,))] / P["K"])

    r = solve_xtfc_inverse(prob, t[:, None], {"u": (t[:, None], u)}, res, ["a", "K"], n_neurons=40,
                           activation="arctan", weight_range=(-10, 10), map_to_unit=True, params0={"a": 0.5, "K": 8})
    assert abs(r.params["a"] - a) < 1e-6 and abs(r.params["K"] - K) < 1e-5
    assert r.predict(np.array([[0.0]]), "u")[0] == pytest.approx(1.0, abs=1e-12)


def test_sir_with_other_parameters_is_recovered_from_clean_data():
    """SIR at beta = 0.7, gamma = 0.2 (not the paper's values)."""
    m = dataclasses.replace(epi_model("sir"), true_params={"beta": 0.7, "gamma": 0.2})
    t, _, y = synthetic_data(m, 0.0)
    data = {c: (t[:, None], y[:, i]) for i, c in enumerate(m.compartments)}
    r = solve_xtfc_inverse(m.problem(), t[:, None], data, m.residual, m.unknowns, n_neurons=50, activation="arctan",
                           weight_range=(-10, 10), seed=1, map_to_unit=True)
    assert abs(r.params["beta"] - 0.7) < 1e-4 and abs(r.params["gamma"] - 0.2) < 1e-4
