"""Convection coefficient h of a pin fin from thermocouple readings, with a physics-informed network (PINNeAPPle).

    pip install pinneapple          (then: python fin_h_from_thermocouples.py, about 30 s on a CPU)

The network learns the temperature profile T(x) and h together from the fin equation, the wall temperature, the
convective tip and the readings; nobody tells it h. Cross-check: a least-squares fit of the analytic fin solution
to the same readings (that formula exists for a fin, not for the 2D and 3D cases).
"""
import math

import numpy as np
import torch
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec

# --- your fin ---------------------------------------------------------------------------------------------------
K = 16.0                               # conductivity, W/m.K
D_MM = 5.0                             # diameter, mm
L_MM = 50.0                            # length, mm
T_WALL = 80.0                          # wall (fin base) temperature, °C
T_AIR = 25.0                           # air temperature, °C
X_MM = [10, 20, 30, 40, 50]            # thermocouple positions from the wall, mm
T_READ = [65.1, 54.9, 47.7, 43.3, 42.9]  # their readings, °C (None: make demo readings from H_TRUE)
H_TRUE = 25.0                          # demo readings only, W/m2.K
NOISE = 0.5                            # demo readings only, ± °C
H_GUESS = 100.0                        # the network's starting value for h, W/m2.K
STEPS = 2000                           # training steps


def fin_theta(xi, h):
    """Analytic (T - T_air) / (T_wall - T_air) of a pin fin with a convective tip, xi = x / L."""
    d, length = D_MM / 1000, L_MM / 1000
    m = math.sqrt(4 * h / (K * d))
    r = h / (m * K)
    return (np.cosh(m * length * (1 - xi)) + r * np.sinh(m * length * (1 - xi))) / (
        np.cosh(m * length) + r * np.sinh(m * length))


xs = np.array(X_MM, float)
if T_READ is None:                     # demo: readings from the analytic profile plus noise
    T_READ = (T_AIR + (T_WALL - T_AIR) * fin_theta(xs / L_MM, H_TRUE)
              + np.random.default_rng(0).normal(0, NOISE, len(xs))).round(2).tolist()
    print("demo readings:", T_READ)
span = T_WALL - T_AIR
D, L = D_MM / 1000, L_MM / 1000

# --- the physics as equations (xi = x/L, T here is theta = (T - T_air) / (T_wall - T_air)) ------------------------
torch.manual_seed(0)
spec = PINNProblemSpec(
    pde_residuals=["Derivative(T(xi), (xi, 2)) - c_fin*h*T(xi)"],                    # fin equation, m^2 = 4h/(kD)
    conditions=[{"name": "wall", "equation": "T(xi) - 1"},                           # wall temperature at xi = 0
                {"name": "tip", "equation": "Derivative(T(xi), xi) + c_tip*h*T(xi)"}],   # convective tip, xi = 1
    independent_vars=["xi"], dependent_vars=["T"],
    inverse_params=["h"],                                                             # unknown: learned with T
    constants={"c_fin": 4 * L**2 / (K * D), "c_tip": L / K},
    loss_weights={"pde": 1.0, "conditions": 10.0, "data": 100.0},
)
loss_fn = PINNFactory(spec).generate_loss_function()
net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=3, num_neurons=32, activation=torch.nn.Tanh())
model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})

xi = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
batch = {
    "collocation": (xi,),
    "conditions": [(torch.zeros(1, 1, requires_grad=True),), (torch.ones(1, 1, requires_grad=True),)],
    "data": ((torch.tensor(xs / L_MM, dtype=torch.float32).reshape(-1, 1),),
             torch.tensor((np.array(T_READ) - T_AIR) / span, dtype=torch.float32).reshape(-1, 1)),
}
opt = torch.optim.Adam([{"params": net.parameters(), "lr": 3e-3},
                        {"params": model.inverse_params.parameters(), "lr": 0.005 * H_GUESS}])
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, STEPS * 2 // 5), gamma=0.3)
for step in range(STEPS + 1):
    opt.zero_grad()
    loss, _ = loss_fn(model, batch)
    loss.backward()
    opt.step()
    sched.step()
    with torch.no_grad():
        model.inverse_params["h"].clamp_(min=1e-3)
    if step % 250 == 0:
        print(f"step {step:5d}  h = {model.inverse_params['h'].item():8.2f} W/m2K")

# --- results ------------------------------------------------------------------------------------------------------
h_pinn = model.inverse_params["h"].item()
x0 = torch.zeros(1, 1, requires_grad=True)                       # heat the fin removes: Fourier's law at the wall,
dtheta = torch.autograd.grad(model(x0).sum(), x0)[0].item()     # from the network's own derivative
q_pinn = -K * math.pi * D**2 / 4 * span / L * dtheta

sse = lambda h: float(np.sum((T_AIR + span * fin_theta(xs / L_MM, h) - np.array(T_READ)) ** 2))  # noqa: E731
a, b, g = 0.01, max(2000.0, 10 * H_GUESS), (math.sqrt(5) - 1) / 2  # golden-section search for the least-squares h
for _ in range(80):
    c, d = b - g * (b - a), a + g * (b - a)
    a, b = (a, d) if sse(c) < sse(d) else (c, b)
h_lsq = (a + b) / 2
print(f"\nleast squares on the analytic profile: h = {h_lsq:.1f} W/m2K")
print(f"Biot number hD/2k = {h_pinn * D / (2 * K):.1e} (the 1D model needs it well below 0.1)")
print(f"h = {h_pinn:.1f} W/m2K, Q = {q_pinn:.3f} W")
