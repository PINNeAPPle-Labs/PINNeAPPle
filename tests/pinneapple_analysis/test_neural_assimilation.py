"""Data assimilation with a model trained here (#346): a neural forward model and a neural observation operator
inside the ensemble Kalman filter and inside 4D-Var, measured on the Lorenz-96 twin experiment.

What this file holds the claim to:

* the surrogate is **trained**, and the test proves it by its held-out error, not by its type: a randomly
  initialised network of the same shape would have a one-step RMSE near the spread of the state (about 3.6), the
  trained one is below 0.1;
* both analysis kernels on that surrogate recover the truth far better than no assimilation at all, and no worse
  than the same kernel on the true model by more than the factor asserted here;
* an **undertrained** network makes the analysis worse, and the test asserts the number rather than tuning it away;
* the observation operator can be a network too: autograd goes through it, so 4D-Var differentiates a learned
  instrument;
* both kernels run on the one :class:`Assimilation` object, and the object refuses a method or an option it does
  not have instead of silently ignoring it.

The experiment costs about a minute (the training dominates); the metrics it asserts are quoted in the module
docstring of ``pinneapple_analysis.data_assimilation.neural``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pinneapple_analysis.data_assimilation.assimilation import (  # noqa: E402
    METHODS,
    AnalysisReport,
    Assimilation,
    cyclic_localisation,
)
from pinneapple_analysis.data_assimilation.models import lorenz96_step  # noqa: E402
from pinneapple_analysis.data_assimilation.neural import (  # noqa: E402
    PeriodicTendency,
    lorenz96_trajectory,
    neural_twin_experiment,
    surrogate_step,
)
from pinneapple_analysis.data_assimilation.var4d import Observation, Var4D, gradient_check  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def twin():
    """One trained surrogate, one twin experiment: every test below reads the same measurement."""
    return neural_twin_experiment(seed=0, train_steps=1500)


@pytest.fixture(scope="module")
def undertrained():
    """The same experiment with a network given a fifth of the training: the negative result."""
    return neural_twin_experiment(seed=0, train_steps=300)


# -- the model is trained, not just different ----------------------------------
def test_surrogate_learns_the_closure_it_was_never_shown(twin):
    m = twin["surrogate"]
    assert m["n_train"] == 4800 and m["n_holdout"] == 1200        # the metrics come from a held-out tail
    assert m["step_rmse"] < 0.1, m                                # one step of Lorenz-96; a random net is ~3.6
    assert m["tendency_rmse"] < 3.0, m
    assert m["free_run_steps"] == 20 and m["free_run_rmse"] < 2.0, m   # 20 steps of free run, the window's length
    # and it keeps some skill over a ten-step rollout from the truth of the twin experiment, which is what an
    # assimilation window needs
    start = twin["truth"][0]
    short = start.clone()
    step = surrogate_step(twin["model"], 0.05)
    with torch.no_grad():
        for _ in range(10):
            short = step(short)
    free = float(torch.sqrt(torch.mean((short - twin["truth"][10]) ** 2)))
    assert free < 1.0, free


def test_a_randomly_initialised_network_is_not_a_model():
    # the control for the test above: same shape, same data, no training -- its ten-step rollout carries no skill
    torch.manual_seed(11)
    raw = PeriodicTendency().double()
    data = lorenz96_trajectory(steps=400, seed=7)
    step = surrogate_step(raw, 0.05)
    x = data[0].clone()
    with torch.no_grad():
        for _ in range(10):
            x = step(x)
    assert float(torch.sqrt(torch.mean((x - data[10]) ** 2))) > 1.5


# -- the measured claim: neural < no assimilation, and close to the true model ---
# Every bound below was checked on three seeds (0, 1, 2); the ones that held on one seed and not the others are in the
# docstring as a spread, not here as an assertion. The twin experiment costs a training, so the suite runs one seed.
def test_the_neural_analysis_beats_no_assimilation_in_both_kernels(twin):
    rmse = twin["rmse"]
    for kernel in ("4dvar", "enkf"):
        neural, free = rmse[f"{kernel}_neural"]["end"], rmse["none_neural"]["end"]
        assert neural < 0.7 * free, (kernel, neural, free)     # measured 0.15-0.59 over three seeds
    # and against the true model's own free run, the harder reference
    assert rmse["enkf_neural"]["end"] < 0.5 * rmse["none_true"]["end"]
    assert rmse["4dvar_neural"]["end"] < 0.5 * rmse["none_true"]["end"]


def test_the_ensemble_analysis_on_the_surrogate_matches_the_true_model(twin):
    # the headline of the measurement: once the one-step error is well below the observation error, the filter built on
    # the surrogate is the filter built on the truth. Three seeds measured 0.84, 1.03 and 1.14 times the true model.
    rmse = twin["rmse"]
    assert rmse["enkf_neural"]["end"] < 1.5 * rmse["enkf_true"]["end"], rmse
    # 4D-Var is not asserted at this factor: over three seeds it measured 0.88, 1.44 and 4.10 times the true-model
    # 4D-Var, the spread the docstring reports. What is asserted is that it still beats no assimilation (above).
    assert rmse["4dvar_neural"]["end"] < rmse["none_true"]["end"]


def test_the_two_kernels_disagree_where_the_theory_says_they_should(twin):
    # 4D-Var analyses the start of the window with all of it; the filter's start is the background (the mean of the
    # initial ensemble, so the background plus sampling noise) and it analyses the end. That asymmetry is the thing
    # the unified object documents, so it is asserted rather than assumed.
    rmse = twin["rmse"]
    assert rmse["4dvar_neural"]["start"] < 0.85 * rmse["enkf_neural"]["start"]
    assert 0.9 * rmse["none_neural"]["start"] < rmse["enkf_neural"]["start"] < 1.2 * rmse["none_neural"]["start"]


def test_an_undertrained_surrogate_makes_the_analysis_worse(undertrained, twin):
    # the negative result: a fifth of the training (one-step RMSE about 0.39 instead of 0.07) is enough to lose most
    # of the gain, though not all of it -- it still beats no assimilation
    assert undertrained["surrogate"]["step_rmse"] > 3 * twin["surrogate"]["step_rmse"]
    assert undertrained["surrogate"]["free_run_rmse"] > 2 * twin["surrogate"]["free_run_rmse"]
    assert undertrained["rmse"]["4dvar_neural"]["end"] > 2 * twin["rmse"]["4dvar_true"]["end"]
    assert undertrained["rmse"]["enkf_neural"]["end"] > 2 * twin["rmse"]["enkf_true"]["end"]
    assert undertrained["rmse"]["enkf_neural"]["end"] < undertrained["rmse"]["none_true"]["end"]


# -- a neural observation operator --------------------------------------------
#: the eight variables the instrument looks at, evenly spaced around the ring
SENSORS = torch.arange(0, 40, 5)


class Sensor(torch.nn.Module):
    """A learned array of eight sensors: each reads the squared anomaly ``((x_i - F) / 4)**2`` at its own variable,
    which is a calibration curve, not an index selection. Fitted from state/reading pairs -- the observation
    operator of an assimilation may be a network as easily as a matrix."""

    def __init__(self, n: int = 40, m: int = len(SENSORS), hidden: int = 64):
        super().__init__()
        self.core = torch.nn.Sequential(torch.nn.Linear(n, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, m))

    def forward(self, x):
        return self.core(x)


def _train_sensor(data, steps: int = 1200, seed: int = 0):
    """The true reading, the network fitted to it, its held-out error and the spread of the reading."""
    with torch.no_grad():
        reads = ((data[..., SENSORS] - 8.0) / 4.0) ** 2
    scale = float(reads.std())
    torch.manual_seed(seed)
    net = Sensor().double()
    x = torch.as_tensor(data, dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(int(steps)):
        idx = torch.randint(0, x.shape[0], (128,), generator=g)
        loss = ((net(x[idx]) - reads[idx] / scale) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        fit = float(torch.sqrt(((net(x) * scale - reads) ** 2).mean()))
    return lambda s: net(s) * scale, fit, scale


def test_a_trained_observation_operator_is_differentiable_and_assimilates():
    data = lorenz96_trajectory(steps=1200, seed=5)
    h, fit_rmse, reading_std = _train_sensor(data)
    assert fit_rmse < 0.25 * reading_std, (fit_rmse, reading_std)     # the instrument is learned, not guessed

    window = 10
    truth = data[: window + 1]
    g = torch.Generator().manual_seed(0)
    obs = [Observation(k, h(truth[k]) + 0.5 * torch.randn(len(SENSORS), generator=g, dtype=torch.float64), h, 0.25)
           for k in range(2, window + 1, 2)]
    xb = truth[0] + 1.0 * torch.randn(40, generator=g, dtype=torch.float64)
    model = lambda s: lorenz96_step(s, 0.05, 8.0)  # noqa: E731

    # autograd goes through the network as it would through any H: the adjoint test of the variational path
    chk = gradient_check(Var4D(model, 1.0), xb + 0.1, xb, obs, window)
    assert chk["relative_error"] < 1e-6, chk

    background = float(torch.sqrt(torch.mean((xb - truth[0]) ** 2)))
    report = Assimilation(model, xb, 1.0, obs, window).analyse("4dvar", truth=truth)
    assert report.rmse["start"] < 0.85 * background, (report.rmse, background)
    assert all(torch.isfinite(report.states).flatten())


# -- one object over both kernels ---------------------------------------------
def test_one_object_runs_both_kernels_on_the_same_problem(twin):
    idx = twin["observed_index"]
    problem = Assimilation(surrogate_step(twin["model"], 0.05), twin["background"], 1.0, twin["observations"], 20)
    reports = {}
    for method in METHODS:
        options = {"max_iter": 50} if method == "4dvar" else {"n_ens": 20, "inflation": 1.05,
                                                              "localisation": cyclic_localisation(40, idx)}
        reports[method] = problem.analyse(method, truth=twin["truth"][:21], **options)
    for method, report in reports.items():
        assert isinstance(report, AnalysisReport) and report.method == method
        assert report.states.shape == (21, 40)
        assert set(report.rmse) == {"start", "end", "window"}
        assert report.n_observations == len(twin["observations"])
        report.to_dict()
    assert reports["4dvar"].spread is None and reports["enkf"].spread is not None
    assert reports["4dvar"].cost_final is not None and reports["enkf"].cost_final is None


def test_the_object_refuses_a_method_or_an_option_it_does_not_have(twin):
    problem = Assimilation(lambda x: lorenz96_step(x, 0.05, 8.0), twin["background"], 1.0, twin["observations"], 20)
    with pytest.raises(ValueError, match="unknown analysis"):
        problem.analyse("letkf")
    with pytest.raises(TypeError, match="n_ens"):
        problem.analyse("4dvar", n_ens=10)        # an ensemble option is not silently ignored by the variational path
    with pytest.raises(TypeError, match="max_iter"):
        problem.analyse("enkf", max_iter=10)
    with pytest.raises(ValueError, match="n_ens >= 2"):
        problem.analyse("enkf", n_ens=1)
    # a filter runs one analysis per step, so two records at the same step are refused there; 4D-Var sums them, which
    # is a legitimate way of writing two instruments reading at once
    twice = Assimilation(lambda x: x, torch.zeros(4), 1.0,
                         [Observation(1, torch.zeros(1)), Observation(1, torch.zeros(1))])
    with pytest.raises(ValueError, match="two observations at step"):
        twice.analyse("enkf", n_ens=4)


def test_localisation_weights_peak_at_the_observation_and_vanish_far_from_it():
    loc_xy, loc_yy = cyclic_localisation(40, [0, 10, 20], half_width=4.0)
    assert loc_xy.shape == (40, 3) and loc_yy.shape == (3, 3)
    assert loc_xy[0, 0] == pytest.approx(1.0) and loc_xy[10, 1] == pytest.approx(1.0)
    assert loc_xy[0, 0] > loc_xy[2, 0] > loc_xy[5, 0]              # decays with distance
    assert loc_xy[8, 0] == pytest.approx(0.0, abs=1e-12)           # zero beyond two half-widths (distance > 8)
    assert loc_xy[39, 0] > 0.0                                     # the ring has no edge: 39 is next to 0
    assert loc_yy[0, 1] == pytest.approx(0.0, abs=1e-12)           # 10 variables apart, beyond the half-width
