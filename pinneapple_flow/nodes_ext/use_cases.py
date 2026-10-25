"""Use-case nodes (``examples/use_cases``): the pin-fin inverse problem of ``fin_convection_inverse`` -- the convection
coefficient h from five noisy thermocouples, by a PINN with h as a trainable parameter
(:class:`pinneapple_physics.pinn_solver.factory.pinn_factory.PINNFactory`) and by a least-squares fit of the closed
form (:func:`pinneapple_physics.closed_form.heat.pin_fin_theta`), over independent noise draws."""
from __future__ import annotations

import math
import time

import numpy as np

from ..registry import Param, Port, node


@node("inverse.pin_fin", "Convection coefficient of a pin fin from thermocouples (PINN)", category="Inverse problems",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("k", "float", 16.0, help="W/m.K (stainless steel)"), Param("D", "float", 0.005),
              Param("L", "float", 0.050), Param("T_base", "float", 80.0), Param("T_air", "float", 25.0),
              Param("h_true", "float", 25.0, help="W/m2.K, used only to make the readings"),
              Param("h_guess", "float", 100.0, help="where the network starts"),
              Param("sensors_mm", "list", [10.0, 20.0, 30.0, 40.0, 50.0]), Param("noise_C", "float", 0.5),
              Param("draws", "int", 10, minimum=1, maximum=100), Param("steps", "int", 2500, minimum=10)],
      calls="pinneapple_physics.pinn_solver.factory.pinn_factory.PINNFactory / PINN / NeuralNetwork (inverse_params) "
            "+ pinneapple_physics.closed_form.heat.pin_fin_theta / pin_fin_heat")
def pin_fin(ctx, inputs, params):
    """examples/use_cases/fin_convection_inverse/fin_convection_inverse.py: theta'' - m^2 theta = 0, theta(0) = 1,
    -k theta'(L) = h theta(L); h and the field learned together from the fin equation and the readings, against a
    least-squares fit of the closed form; the heat the fin dissipates from the network's own derivative."""
    import torch

    from pinneapple_physics.closed_form.heat import pin_fin_heat, pin_fin_theta
    from pinneapple_physics.pinn_solver.factory.pinn_factory import (
        PINN,
        NeuralNetwork,
        PINNFactory,
        PINNProblemSpec,
    )
    K, D, L = float(params["k"]), float(params["D"]), float(params["L"])
    TB, TA, H = float(params["T_base"]), float(params["T_air"]), float(params["h_true"])
    S = np.asarray(params["sensors_mm"], float)
    AREA = math.pi * D ** 2 / 4
    C_FIN, C_TIP = 4 * L ** 2 / (K * D), L / K
    torch.set_num_threads(1)

    def readings(seed):
        rng = np.random.default_rng(seed)
        t = TA + (TB - TA) * pin_fin_theta(S / 1000 / L, H, K, D, L)
        return t + rng.normal(0.0, float(params["noise_C"]), size=t.shape)

    def identify(t_meas, seed, steps):
        torch.manual_seed(seed)
        spec = PINNProblemSpec(
            pde_residuals=["Derivative(T(xi), (xi, 2)) - c_fin*h*T(xi)"],
            conditions=[{"name": "base", "equation": "T(xi) - 1", "weight": 1.0},
                        {"name": "tip", "equation": "Derivative(T(xi), xi) + c_tip*h*T(xi)", "weight": 1.0}],
            independent_vars=["xi"], dependent_vars=["T"], inverse_params=["h"],
            constants={"c_fin": C_FIN, "c_tip": C_TIP}, loss_weights={"pde": 1.0, "conditions": 10.0, "data": 100.0})
        loss_fn = PINNFactory(spec).generate_loss_function()
        net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=3, num_neurons=32, activation=torch.nn.Tanh())
        model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": float(params["h_guess"])})
        opt = torch.optim.Adam([{"params": net.parameters(), "lr": 3e-3},
                                {"params": model.inverse_params.parameters(), "lr": 0.5}])
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.3)
        batch = {"collocation": (torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True),),
                 "conditions": [(torch.zeros((1, 1), requires_grad=True),), (torch.ones((1, 1), requires_grad=True),)],
                 "data": ((torch.tensor(S / 1000 / L, dtype=torch.float32).reshape(-1, 1),),
                          torch.tensor((t_meas - TA) / (TB - TA), dtype=torch.float32).reshape(-1, 1))}
        for _ in range(steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(model, batch)
            loss.backward()
            opt.step()
            sched.step()
        xi = torch.linspace(0, 1, 401).reshape(-1, 1).requires_grad_(True)
        th = model(xi)
        dth = torch.autograd.grad(th.sum(), xi)[0]
        return (float(model.inverse_params["h"].detach()), xi.detach().numpy().ravel(), th.detach().numpy().ravel(),
                float(-K * AREA * (TB - TA) / L * dth[0, 0]))

    def lsq(t_meas):
        xi_s = S / 1000 / L

        def sse(h):
            return float(np.sum((TA + (TB - TA) * pin_fin_theta(xi_s, h, K, D, L) - t_meas) ** 2))
        a, b, g = 1.0, 200.0, (math.sqrt(5) - 1) / 2
        for _ in range(100):
            c, d = b - g * (b - a), a + g * (b - a)
            a, b = (a, d) if sse(c) < sse(d) else (c, b)
        return (a + b) / 2
    t0 = time.perf_counter()
    hp, hl, qp, ferr, prof = [], [], [], [], None
    for draw in range(int(params["draws"])):
        ctx.check_stop()
        tm = readings(100 + draw)
        h, xi, th, q = identify(tm, draw, int(params["steps"]))
        hp.append(h)
        hl.append(lsq(tm))
        qp.append(q)
        ferr.append(float(np.max(np.abs(th - pin_fin_theta(xi, H, K, D, L))) * (TB - TA)))
        prof = prof or (xi, th)
        ctx.log(f"draw {draw}: h {h:.2f} (LSQ {hl[-1]:.2f}), Q {q:.4f} W")
    hp, hl, qp = map(np.asarray, (hp, hl, qp))
    ctx.metric("h_pinn.mean", float(hp.mean()), f"over {len(hp)} noise draws")
    if len(hp) > 1:
        ctx.metric("h_pinn.std", float(hp.std(ddof=1)))
        ctx.metric("h_least_squares.std", float(hl.std(ddof=1)))
    ctx.metric("h_least_squares.mean", float(hl.mean()))
    ctx.metric("q_true_W", pin_fin_heat(H, K, D, L, TB, TA), "closed form at the true h")
    ctx.metric("q_pinn_W.mean", float(qp.mean()))
    ctx.metric("max_field_error_C", float(max(ferr)))
    ctx.metric("seconds", round(time.perf_counter() - t0, 1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(prof[0] * L * 1e3, TA + (TB - TA) * prof[1], "r--", label="PINN (draw 0)")
    ax.plot(prof[0] * L * 1e3, TA + (TB - TA) * pin_fin_theta(prof[0], H, K, D, L), "k-", label="closed form")
    ax.plot(S, readings(100), "o", label="thermocouples (draw 0)")
    ax.set(xlabel="x [mm]", ylabel="T [C]", title=f"h: PINN {hp.mean():.2f}, least squares {hl.mean():.2f}, true {H:g}")
    ax.legend()
    f = ctx.figure(fig, "fin", "fin temperature and the identified h")
    return {"metrics": ctx.report["metrics"], "figure": f}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="inverse.pin_fin", title="Convection coefficient of a fin from a few thermocouples", domain="inverse problems",
        example="use_cases", keywords={"pin fin": 3, "fin": 1.5, "convection coefficient": 3, "heat transfer coefficient": 3,
                                       "thermocouple": 2, "thermocouples": 2},
        requires_all=(("fin",),), min_score=3, nodes=("inverse.pin_fin",),
        references=("raissi2019pinn", "incropera2007"), template="use_cases/fin_convection_inverse", quick=True,
        description="examples/use_cases/fin_convection_inverse/fin_convection_inverse.py"))


_register_capabilities()
