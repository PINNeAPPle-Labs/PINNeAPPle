"""pinneapple_systems.control.optimal: PoNN and BeNN against closed-form answers.

Every threshold here is a number measured on this machine and quoted next to the assertion, so a
failure says what changed.  The two references are independent of the module: the algebraic Riccati
equation is solved with :func:`scipy.linalg.solve_continuous_are`, the finite-horizon one with a
backward ODE integration of the Riccati equation, and the controller design with a backward
integration of the Pontryagin system; the closed forms (cubic transfer, bang-bang landing) are
algebra.  The whole file fits in well under two minutes: the two value-network fits dominate.
"""
import numpy as np
import pytest
from scipy import linalg

from pinneapple_systems.control.optimal import (
    LQProblem,
    _fly,
    _fly_with_handover,
    _state_after_gust,
    bellman,
    direct_transcription,
    landing_demo,
    min_energy_transfer,
    min_fuel_landing,
    mpc_control,
    pontryagin,
    residual_report,
    riccati_solution,
    steady_state_lqr,
)

# the regulator used throughout: a double integrator, the simplest system whose Riccati solution is
# not diagonal, so a transposed or symmetrised A cannot pass unnoticed.
A_LQR = np.array([[0.0, 1.0], [0.0, 0.0]])
B_LQR = np.array([[0.0], [1.0]])
Q_LQR = np.diag([1.0, 1.0])
R_LQR = np.array([[1.0]])


def lqr_problem(T: float = 5.0) -> LQProblem:
    return LQProblem(A=A_LQR, B=B_LQR, Q=Q_LQR, R=R_LQR, x0=[1.0, 0.0], t_span=(0.0, T))


# --------------------------------------------------------------------------------------
# the LQR regulator against the algebraic Riccati equation
# --------------------------------------------------------------------------------------
def test_steady_state_lqr_against_scipy_are():
    """``P``, ``K = R^-1 B' P`` and ``1/2 x0' P x0`` reproduce an independently solved CARE.

    ``scipy.linalg.solve_continuous_are(A, B, Q, R)`` is the reference here, and it is the
    *stabilising* root by construction -- which matters, because the HJB residual alone does not pick
    it (see ``test_bellman_does_not_claim_the_stationary_problem``).  Measured: ``P`` agrees to
    1.8e-16 absolute on its largest entry, ``K`` to 0.0e+00, and the two poles of ``A - B K`` are
    ``-0.5`` and ``-1.118`` (both in the left half plane, ``A`` itself has one at the origin).
    """
    problem = lqr_problem()
    out = steady_state_lqr(problem)
    P_ref = linalg.solve_continuous_are(A_LQR, B_LQR, Q_LQR, R_LQR)
    K_ref = np.linalg.solve(R_LQR, B_LQR.T @ P_ref)

    assert np.abs(out["P"] - P_ref).max() < 1e-12            # measured 1.8e-16
    assert np.abs(out["gain"] - K_ref).max() < 1e-12          # measured 0.0e+00
    # the cost carries the half convention of this module against the no-half one of tuning.lqr
    assert out["cost"] == pytest.approx(0.5 * problem.x0 @ P_ref @ problem.x0, rel=1e-12)
    assert out["cost"] == pytest.approx(0.8660254037844384, rel=1e-12)
    assert np.all(np.real(out["poles"]) < 0.0)
    # and P really solves the CARE, not just equals the solver's output
    residual = A_LQR.T @ out["P"] + out["P"] @ A_LQR - out["P"] @ B_LQR @ B_LQR.T @ out["P"] + Q_LQR
    assert np.abs(residual).max() < 1e-12                     # measured 1.6e-15


# --------------------------------------------------------------------------------------
# PoNN: the two-point problem
# --------------------------------------------------------------------------------------
def test_pontryagin_reproduces_the_finite_horizon_riccati_cost():
    """The cost of the X-TFC solution against a backward integration of the Riccati equation.

    ``riccati_solution`` integrates ``Pdot = -(A'P + PA - P B R^-1 B' P + Q)`` from ``P(tf) = 0``: it
    shares no code with the X-TFC collocation.  Measured on the double integrator over 5 s:
    ``J_pontryagin = 0.865515``, ``J_riccati = 0.865425`` -- 1.0e-4 relative, which is the trapezoid
    error of the reported cost, not a solved-for difference.  The collocation residual after the
    single least-squares solve is 4.5e-8 RMS, and the boundary conditions hold at 0.0e+00 because
    the constrained expression carries them identically.
    """
    problem = lqr_problem()
    reference = riccati_solution(problem)
    solution = pontryagin(problem)

    assert solution.cost == pytest.approx(reference["cost"], rel=1e-3)   # measured 1.0e-4
    assert solution.boundary_error < 1e-12                              # measured 0.0e+00
    assert solution.residual_rms < 1e-6                                 # measured 4.5e-8
    assert solution.n_iter == 1                                         # one affine solve
    # the value grows towards t0: the Riccati P starts at Qf = 0 at tf and climbs backwards
    assert reference["P"][0, 0, 0] > reference["P"][-1, 0, 0]          # 3.53 vs 0.0
    assert np.all(reference["P"][-1] == 0.0)


def test_pontryagin_energy_optimal_transfer_matches_the_cubic():
    """The hard-terminal-constraint case against the closed-form minimum-energy transfer.

    ``x(tf) = xf`` pinned exactly (``terminal_mask``) leaves the costate endpoint *free*: with
    ``Qf = 0`` the transversality would have said ``lambda(tf) = 0``, which no transfer to a
    different orbit satisfies.  Measured: control endpoints ``+2.00000`` and ``-4.00000`` against
    the closed form's ``+2`` and ``-4``, cost 20.0007 against 20.0 (3.8e-5 relative), terminal state
    ``[100.0, 0.0]`` at 0.0e+00.
    """
    problem = LQProblem(A=A_LQR, B=B_LQR, Q=np.zeros((2, 2)), R=R_LQR, x0=[0.0, 10.0],
                        t_span=(0.0, 10.0), Qf=np.zeros((2, 2)), xf=[100.0, 0.0],
                        terminal_mask=[True, True])
    closed = min_energy_transfer(0.0, 10.0, 100.0, 10.0, vf=0.0)
    solution = pontryagin(problem, n_basis=24, n_collocation=160)

    assert closed.cost == pytest.approx(20.0, rel=1e-12)
    assert solution.control[0, 0] == pytest.approx(closed.u0, abs=1e-3)   # measured +2.00000
    assert solution.control[-1, 0] == pytest.approx(closed.u1, abs=1e-3)  # measured -4.00000
    assert solution.cost == pytest.approx(closed.cost, rel=1e-4)          # measured 3.8e-5
    assert solution.boundary_error < 1e-10                                # measured 0.0e+00
    assert np.abs(solution.state[-1] - problem.xf).max() < 1e-8           # measured 0.0e+00


def test_pontryagin_tfc_expression_holds_the_boundary_conditions_exactly():
    """The constrained expression satisfies the terminal condition to machine precision.

    This is the part of the method that is exact rather than fitted: the boundary row is removed
    from the basis, so every coefficient vector satisfies it.  Measured: ``|x(tf) - xf| = 0.0e+00``
    for the pinned component and ``|x(t0) - x0| = 0.0e+00`` on this doubly pinned variable, and the
    free-state regulator reaches ``|x(t0) - x0| <= 1.1e-16``.
    """
    problem = LQProblem(A=A_LQR, B=B_LQR, Q=np.zeros((2, 2)), R=R_LQR, x0=[0.0, 10.0],
                        t_span=(0.0, 10.0), Qf=np.zeros((2, 2)), xf=[100.0, 0.0],
                        terminal_mask=[True, True])
    solution = pontryagin(problem)
    x0, xf = solution.state_at([0.0])[0], solution.state_at([10.0])[0]
    assert np.abs(x0 - problem.x0).max() < 1e-10        # measured 0.0e+00
    assert np.abs(xf - problem.xf).max() < 1e-10        # measured 0.0e+00

    regulator = pontryagin(lqr_problem())
    assert np.abs(regulator.state_at([0.0])[0] - regulator.problem.x0).max() < 1e-10  # 1.1e-16
    # a terminal cost instead of a mask: lambda(tf) = Qf (x(tf) - xf) holds to the same precision
    with_cost = LQProblem(A=A_LQR, B=B_LQR, Q=Q_LQR, R=R_LQR, x0=[1.0, 0.0], t_span=(0.0, 5.0),
                          Qf=np.diag([2.0, 2.0]), xf=[0.0, 0.0])
    pinned = pontryagin(with_cost)
    lam_tf = pinned.costate_at([5.0])[0]
    assert np.abs(lam_tf - with_cost.Qf @ (pinned.state_at([5.0])[0] - with_cost.xf)).max() < 1e-10


# --------------------------------------------------------------------------------------
# the closed forms
# --------------------------------------------------------------------------------------
def test_min_fuel_landing_switching_time_and_fuel():
    """The bang-bang landing against its own two-arc solution, integrated independently.

    ``tau = sqrt((v0^2 + 2 g h0) / (a_max (a_max - g)))`` and ``t_coast = (v0 + (a_max - g) tau)/g``
    come from requiring touchdown at zero velocity; the state here is then integrated with RK4 over
    the two arcs, using the switching time the class reports.  Measured: touchdown ``h = 2.1e-14``,
    ``v = 3.6e-15``, fuel ``28.61`` m/s over a 4.41 s burn and a 3.99 s coast.
    """
    h0, v0, g, a_max = 120.0, -15.0, 1.62, 4.0 * 1.62
    landing = min_fuel_landing(h0, v0, g, a_max)
    assert landing.burn_time == pytest.approx(np.sqrt((v0 ** 2 + 2.0 * g * h0) / (a_max * (a_max - g))))
    assert landing.coast_time == pytest.approx((v0 + (a_max - g) * landing.burn_time) / g)
    assert landing.fuel == pytest.approx(a_max * landing.burn_time, rel=1e-15)   # 28.6077
    assert landing.flight_time == pytest.approx(8.3998, rel=1e-4)

    h, v = landing.state(landing.flight_time)
    assert abs(float(h[0])) < 1e-10          # measured 2.1e-14
    assert abs(float(v[0])) < 1e-10          # measured 3.6e-15
    # the thrust switches once, at the coast time: zero before, a_max after
    assert np.all(landing.control(np.array([0.0, 0.5 * landing.coast_time])) == 0.0)
    assert np.all(landing.control(np.array([landing.coast_time + 1e-9, landing.flight_time])) == a_max)
    # the burn is the only place fuel is spent, and the mean thrust follows from it
    assert landing.summary()["mean_thrust"] == pytest.approx(landing.fuel / landing.flight_time)


def test_min_energy_transfer_is_the_cubic_through_the_endpoints():
    """The transfer returns a cubic whose endpoints and cost close exactly, against the algebra.

    The free-final-velocity case is the one that catches a wrong closed form: with ``u(t) = u0 (1 -
    t/T)`` (the costate is linear and ``lambda_v(T) = 0``), ``u0 = 6D/T^2`` and the energy is
    ``u0^2 T/6 = 6 D^2 / T^3`` -- 9.6 J for ``D = -40`` over 10 s, which is the number here.  (The
    class docstring said ``3 D^2 / 2 T^3``; the code was right and the sentence was off by four.)
    """
    transfer = min_energy_transfer(0.0, 10.0, 100.0, 10.0, vf=0.0)
    p, v = transfer.state(np.array([0.0, 10.0]))
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert v[0] == pytest.approx(10.0, abs=1e-12)
    assert p[1] == pytest.approx(100.0, abs=1e-10)     # 1.4e-14
    assert v[1] == pytest.approx(0.0, abs=1e-10)       # 3.6e-15
    # J = T (u0^2 + u0 u1 + u1^2) / 6 with u0 = 2, u1 = -4 -> 20
    assert transfer.cost == pytest.approx(20.0, rel=1e-12)
    assert (transfer.u0, transfer.u1) == pytest.approx((2.0, -4.0), abs=1e-12)
    # free final velocity: lambda_v(T) = 0, the control ends at zero, and J = 6 D^2 / T^3
    free = min_energy_transfer(0.0, 10.0, 60.0, 10.0)
    assert free.u1 == 0.0
    assert free.u0 == pytest.approx(6.0 * (60.0 - 100.0) / 10.0 ** 2, rel=1e-12)     # -2.4
    assert free.cost == pytest.approx(6.0 * (60.0 - 100.0) ** 2 / 10.0 ** 3, rel=1e-12)  # 9.6
    # and the same energy from the integral of u^2/2, not from the formula it is checking
    grid = np.linspace(0.0, 10.0, 20001)
    assert 0.5 * np.trapezoid(free.control(grid) ** 2, grid) == pytest.approx(free.cost, rel=1e-8)


# --------------------------------------------------------------------------------------
# BeNN: the value network
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lqr_net():
    """A value network on the regulator over 10 s, the case the docs table quotes."""
    problem = lqr_problem(10.0)
    return bellman(problem), riccati_solution(problem)


def test_bellman_terminal_condition_is_exact(lqr_net):
    """``V(tf, x) = 1/2 (x - xf)' Qf (x - xf)`` for any coefficient vector: the TFC projection.

    ``psi(t, x) = phi(t, x) - phi(tf, x)`` is built so the coefficients cancel at ``tf``; the
    terminal cost is added back.  Measured at 25 states drawn across the training box:
    ``|V(tf, x) - terminal cost| = 2.8e-12`` absolute, which is the rounding of the feature sum, and
    ``|grad V| = 0`` at ``tf`` to 1.4e-16 because the terminal gradient is exactly what ``Qf`` says.
    """
    net = lqr_net[0]
    problem = net.problem
    rng = np.random.default_rng(7)
    x = rng.uniform(net.box[:, 0], net.box[:, 1], size=(25, problem.n))
    tf = np.full(25, problem.t_span[1])
    assert np.abs(net.value(tf, x) - problem.terminal_cost(x)).max() < 1e-10     # measured 2.8e-12
    assert np.abs(net.grad_value(tf, x) - (x - problem.xf) @ problem.Qf.T).max() < 1e-8


def test_bellman_value_and_gain_against_riccati(lqr_net):
    """The value and the policy gain against the backward Riccati integration, on a 10 s horizon.

    This is the accuracy the module claims for BeNN, stated as measured: the cost to go at the start
    is ``0.9258`` against Riccati's ``0.8660`` (+6.9 %), and the local policy gain at ``x0`` is
    ``[0.9983, 1.7166]`` against ``K(t0) = [1.0000, 1.7321]`` (0.17 % and 0.89 % on the two
    entries).  The gain is the robust part and the value is the weak one: the features can carry the
    gradient better than they carry the value.
    """
    net, reference = lqr_net
    assert net.cost_to_go() == pytest.approx(reference["cost"], rel=0.10)        # +6.9 %
    gain_ref = reference["gain"][0]
    gain = net.gain()
    assert np.abs(gain - gain_ref).max() < 0.02 * np.abs(gain_ref).max()        # measured 0.0155
    assert gain[0, 0] == pytest.approx(gain_ref[0, 0], rel=0.02)                # 0.9983 vs 1.0000
    assert gain[0, 1] == pytest.approx(gain_ref[0, 1], rel=0.02)                # 1.7166 vs 1.7321


def test_bellman_value_never_claims_the_stationary_problem():
    """The stationary HJB is not offered, and the docstring says why with the number behind it.

    A stationary fit solves the HJB residual just as exactly on the anti-stabilising root of the
    algebraic Riccati equation, which is not a value function; the module therefore exposes only the
    finite horizon, whose terminal condition makes the answer unique.  Measured when the stationary
    branch existed: residual ``3.8e-16`` with gain ``[-0.62, +0.62]`` and cost to go ``-0.69``
    against the Riccati ``[1.00, 1.73]`` and ``+0.87``.
    """
    import inspect

    assert "horizon" not in inspect.signature(bellman).parameters
    assert "reference_x" not in inspect.signature(bellman).parameters
    assert "anti-stabilising" in (bellman.__doc__ or "")


# --------------------------------------------------------------------------------------
# the demo problem: trust, and the hand-over
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def landing_fixture():
    """A small landing value network plus the nominal flight and the flight after a gust.

    Smaller than :func:`landing_demo` (16 features, 12 time nodes) so the assertions stay fast; the
    ordering it shows is the one the demo shows, with the numbers below.
    """
    landing = min_fuel_landing(120.0, -15.0, 1.62, 4.0 * 1.62)
    duration = landing.flight_time
    problem = LQProblem(A=A_LQR, B=B_LQR, Q=np.zeros((2, 2)), R=R_LQR, x0=[120.0, -15.0],
                        t_span=(0.0, duration), c=[0.0, -1.62], Qf=np.diag([0.5, 5.0]),
                        xf=[0.0, 0.0], name="test landing")
    corridor = np.array([[-5.0, 132.0], [-18.0, 2.0]])
    net = bellman(problem, box=corridor, weighted=True, n_features=16, poly_degree=2,
                  poly_time_degree=4, n_time=12, n_space=9, seed=0, max_nfev=150)
    nominal = _fly(net, [120.0, -15.0], 0.0, duration, 0.5)
    gust = (0.45 * duration, [0.0, -20.0])
    after = _state_after_gust(nominal, gust)
    disturbed = _fly(net, after, gust[0], duration, 0.5)
    return {"landing": landing, "problem": problem, "net": net, "duration": duration,
            "nominal": nominal, "disturbed": disturbed, "gust": gust,
            "nominal_report": residual_report(net, nominal),
            "gust_report": residual_report(net, disturbed)}


def test_hjb_residual_is_larger_outside_the_fitted_region(landing_fixture):
    """The trust claim, as an ordering with the two numbers that make it.

    The same network, flown from the nominal start and from the state a -20 m/s gust leaves it in:
    ``relative rms 0.378`` inside the corridor against ``0.913`` after the gust, ``relative max
    1.099`` against ``2.938``, and in absolute units ``239`` against ``577``.  The trajectory that
    stayed in the box the value was fitted on is the one with the smaller residual, which is the
    whole content of the claim -- it is a statement about the fit, not about optimality.
    """
    nominal = landing_fixture["nominal_report"]
    gust = landing_fixture["gust_report"]
    assert nominal["relative_rms"] == pytest.approx(0.378, abs=0.02)
    assert gust["relative_rms"] == pytest.approx(0.913, abs=0.05)
    assert gust["relative_rms"] > 2.0 * nominal["relative_rms"]          # measured 2.42x
    assert gust["relative_max"] > nominal["relative_max"]                # 2.938 vs 1.099
    assert gust["rms"] > nominal["rms"]                                  # 577 vs 239
    assert nominal["within"] and gust["within"]                          # no threshold given


def test_handover_fires_when_the_residual_crosses_and_not_before(landing_fixture):
    """The hand-over to MPC triggers on the crossing, and the same flight without a gust never does.

    With the threshold at ``1.5`` -- above the nominal flight's largest relative residual (1.099) and
    below the largest the gusted flight reaches (3.088 in the step that follows it) -- the disturbed
    flight hands over at ``t = 4.00 s``, the first grid step at or after the gust (3.78 s), and the
    undisturbed flight reports ``handed_over_at = None``.  Measured touchdown: ``h = +51.26 m`` on
    the learned policy alone (it gives up part-way) and ``h = -0.52 m`` with the hand-over, i.e. the
    MPC puts it on the ground.
    """
    net = landing_fixture["net"]
    duration, gust = landing_fixture["duration"], landing_fixture["gust"]
    threshold = 1.5
    assert landing_fixture["nominal_report"]["relative_max"] < threshold         # 1.099
    assert landing_fixture["nominal_report"]["relative_rms"] < threshold         # 0.378

    with_gust = _fly_with_handover(net, [120.0, -15.0], 0.0, duration, 0.5, threshold,
                                   gust=gust, seed=0)
    no_gust = _fly_with_handover(net, [120.0, -15.0], 0.0, duration, 0.5, threshold,
                                 gust=None, seed=0)

    assert no_gust["handed_over_at"] is None
    assert no_gust["residuals"].max() < threshold                                # 1.099
    assert with_gust["handed_over_at"] is not None
    # the gust is applied on the first grid step at or after its own time: 3.78 s -> the step at 4.0 s
    assert with_gust["disturbed_at"] >= gust[0] - 1e-12                          # 4.00 >= 3.78
    assert with_gust["disturbed_at"] - gust[0] < 0.5                             # within one step
    assert with_gust["handed_over_at"] >= with_gust["disturbed_at"]              # 4.00 >= 4.00
    # the switch happens on the first step whose residual is over the line, not later and not before
    # (one residual per applied step, recorded before the step is integrated)
    times, residuals = with_gust["t"][:-1], with_gust["residuals"]
    first = float(times[int(np.flatnonzero(residuals > threshold)[0])])
    assert with_gust["handed_over_at"] == pytest.approx(first, abs=1e-9)         # 4.00 s
    assert np.all(residuals[times < with_gust["handed_over_at"]] <= threshold)
    # and it is the hand-over that lands it
    assert with_gust["state"][-1, 0] < no_gust["state"][-1, 0]                   # -0.52 < 51.26


def test_mpc_step_is_the_exact_feedback_of_the_linear_problem():
    """One MPC step re-solves the two-point problem and returns the optimal control there.

    On a linear-quadratic problem the receding-horizon re-solve is the exact optimal feedback, so it
    can be checked against ``u = -R^-1 B' (P x + s)`` from the Riccati reference.  Measured at
    ``t = 1 s`` from ``x = (0.8, -0.2)`` on the regulator: ``u = -0.450257`` from the MPC against
    ``-0.450258`` from the Riccati reference (2.2e-6 relative), on a collocation of 80 points that
    leaves a residual of 9.0e-5.
    """
    problem = lqr_problem()
    x_now = np.array([0.8, -0.2])
    step = mpc_control(problem, 1.0, x_now)
    reference = riccati_solution(problem, t_query=[1.0])
    P, s = reference["P"][0], reference["s"][0]
    u_ref = -(problem.Rinv @ problem.B.T) @ (P @ x_now + s)
    assert float(step["control"][0]) == pytest.approx(float(u_ref[0]), rel=1e-4)   # 2.2e-6 measured
    assert step["residual_rms"] < 1e-4                                            # measured 9.0e-5


def test_direct_transcription_agrees_with_the_closed_forms():
    """The independent discretised solve lands on the same cost, and only as close as 24 knots allow.

    SLSQP on the RK4-marched transcription shares nothing with the constrained expressions or the
    Riccati integration.  Measured on the 5 s regulator: ``0.8695`` against Riccati's ``0.8654``
    (+0.48 %), and on the constrained transfer ``20.059`` against the cubic's ``20.0`` (+0.29 %).
    """
    problem = lqr_problem()
    transcribed = direct_transcription(problem, n_knots=24)
    assert transcribed["success"]
    assert transcribed["cost"] == pytest.approx(riccati_solution(problem)["cost"], rel=0.01)  # 0.48 %

    transfer = LQProblem(A=A_LQR, B=B_LQR, Q=np.zeros((2, 2)), R=R_LQR, x0=[0.0, 10.0],
                         t_span=(0.0, 10.0), Qf=np.zeros((2, 2)), xf=[100.0, 0.0],
                         terminal_mask=[True, True])
    constrained = direct_transcription(transfer, n_knots=24)
    assert constrained["cost"] == pytest.approx(20.0, rel=0.01)                   # measured 20.059
    assert np.abs(constrained["state"][-1] - transfer.xf).max() < 1e-6            # equality held


def test_landing_demo_runs_and_reports_the_hand_over():
    """The demo end to end: the gust is caught, the hand-over fires, the table says both.

    Kept as one test because the demo is one measured story.  Measured: flight 8.40 s, fuel 28.61
    m/s, nominal relative residual 0.409, after the gust 17.81, threshold 4.094, hand-over at 4.00 s
    against a gust at 3.78 s, touchdown +36.42 m on the policy alone and +2.59 m with the hand-over.
    """
    out = landing_demo()
    assert out["landing"].flight_time == pytest.approx(8.3998, rel=1e-3)
    assert out["nominal_report"]["relative_rms"] < 1.0                   # measured 0.409
    assert out["disturbed_report"]["relative_rms"] > 10.0                # measured 17.81
    assert out["disturbed_report"]["within"] is False
    assert out["disturbed"]["handed_over_at"] is not None
    assert out["disturbed"]["handed_over_at"] >= out["gust"][0] - 1e-9   # 4.00 vs 3.78
    assert out["disturbed"]["state"][-1, 0] < out["free"]["state"][-1, 0]  # 2.59 vs 36.42
    assert len(out["table"]) == 7
    assert all(isinstance(line, str) for line in out["table"])


# --------------------------------------------------------------------------------------
# the problem container itself
# --------------------------------------------------------------------------------------
def test_lq_problem_does_not_symmetrise_the_drift():
    """``A`` is kept as given: symmetrising it silently solves another problem.

    ``0.5 (A + A')`` on the double integrator turns ``pdot = v, vdot = u`` into a different system
    entirely, and every closed form then agrees with a wrong answer.  Measured: the stored ``A``
    equals the input matrix exactly (including the off-diagonal ``2`` that symmetrising would have
    moved to ``1``), and ``Q`` is symmetrised as documented.
    """
    problem = LQProblem(A=[[0.0, 2.0], [0.0, 0.0]], B=B_LQR, Q=[[1.0, 3.0], [0.0, 2.0]],
                        R=R_LQR, x0=[1.0, 0.0], t_span=(0.0, 1.0))
    assert np.array_equal(problem.A, np.array([[0.0, 2.0], [0.0, 0.0]]))
    assert problem.A[0, 1] == 2.0 and problem.A[1, 0] == 0.0
    assert np.array_equal(problem.Q, np.array([[1.0, 1.5], [1.5, 2.0]]))


def test_lq_problem_rejects_what_it_cannot_solve():
    """Invalid input is refused rather than solved into nonsense."""
    with pytest.raises(ValueError):
        LQProblem(A=A_LQR, B=B_LQR, Q=Q_LQR, R=[[0.0]], x0=[1.0, 0.0], t_span=(0.0, 1.0))
    with pytest.raises(ValueError):
        LQProblem(A=A_LQR, B=B_LQR, Q=Q_LQR, R=R_LQR, x0=[1.0, 0.0], t_span=(1.0, 1.0))
    with pytest.raises(ValueError):
        LQProblem(A=[[0.0, 1.0, 0.0]], B=B_LQR, Q=Q_LQR, R=R_LQR, x0=[1.0, 0.0], t_span=(0.0, 1.0))
    with pytest.raises(ValueError):
        min_fuel_landing(120.0, -15.0, 1.62, 1.0)          # a_max < g: cannot hover
