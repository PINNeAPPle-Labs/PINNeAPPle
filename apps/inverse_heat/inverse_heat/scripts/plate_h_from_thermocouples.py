"""2D: convection coefficient h of a heat-spreader plate from eight thermocouples, and the hot spot it implies.

    pip install pinneapple          (then: python plate_h_from_thermocouples.py, about 2-4 min on a CPU)

A thin plate carries a device and cools from both faces with an unknown h. No formula exists for this temperature
map: the PINN learns T(x, y) and h together from the plate equation, the adiabatic edges and the readings,

    k t (T_xx + T_yy) - 2 h (T - T_air) + q''(x, y) = 0.

Check: an independent finite-volume solution of the same plate (also used to make demo readings).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.interpolate import RegularGridInterpolator
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec

# --- your plate ---------------------------------------------------------------------------------------------------
LX, LY, THK = 0.100, 0.060, 0.001      # plate size and thickness, m
K = 200.0                              # conductivity, W/m.K (aluminium)
POWER = 8.0                            # device power, W
T_AIR = 25.0                           # air, °C
DEVICE = (0.030, 0.030, 0.010, 0.002)  # device centre x, centre y, half side, edge smoothing (m)
SENSORS_MM = [[10, 10], [50, 10], [90, 10], [70, 30], [90, 50], [50, 50], [10, 50], [30, 55]]
T_READ = None                          # your readings, °C, one per sensor (None: demo readings from H_TRUE)
H_TRUE = 15.0                          # demo readings and the check, W/m2.K
NOISE = 0.5                            # demo readings only, ± °C
H_GUESS = 60.0                         # the network's starting value for h, W/m2.K
STEPS = 3000                           # training steps


def footprint(x, y):
    """Smooth square under the device: 1 inside, 0 outside (works on numpy arrays and torch tensors)."""
    xc, yc, half, w = DEVICE
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh
    sig = lambda s: 0.5 * (1 + tanh(s / w))  # noqa: E731
    return sig(half - abs(x - xc)) * sig(half - abs(y - yc))


def plate_fv(h, nx=100, ny=60):
    """Finite-volume temperature (°C) at the cell centres, edges adiabatic; returns x, y, T[ny, nx]."""
    dx, dy = LX / nx, LY / ny
    x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y)
    fp = footprint(xx, yy)
    q = POWER / (fp.sum() * dx * dy) * fp                         # W/m2, integrates to POWER
    idx = np.arange(nx * ny).reshape(ny, nx)
    diag = np.full(nx * ny, 2 * h * dx * dy)                     # convection from both faces
    rows, cols, vals = [], [], []
    for a, b, g in ((idx[:, :-1], idx[:, 1:], K * THK * dy / dx), (idx[:-1, :], idx[1:, :], K * THK * dx / dy)):
        for p, n in ((a.ravel(), b.ravel()), (b.ravel(), a.ravel())):   # conduction to each neighbour
            rows.append(p); cols.append(n); vals.append(np.full(p.size, -g))  # noqa: E702
            np.add.at(diag, p, g)
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag)  # noqa: E702
    a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    rhs = (2 * h * T_AIR + q).ravel() * dx * dy
    return x, y, spla.spsolve(a, rhs).reshape(ny, nx)


def at_sensors(x, y, t):
    s = np.array(SENSORS_MM, float) / 1000
    return RegularGridInterpolator((y, x), t, bounds_error=False, fill_value=None)(s[:, ::-1])


DEMO = T_READ is None
x, y, t_ref = plate_fv(H_TRUE)
if DEMO:
    T_READ = (at_sensors(x, y, t_ref) + np.random.default_rng(0).normal(0, NOISE, len(SENSORS_MM))).round(2).tolist()
    print("demo readings:", T_READ)

# --- the physics as equations (xi = x/L, eta = y/L, T here is (T - T_air) / dT) -----------------------------------
L, DT = LX, max(10.0, max(T_READ) - T_AIR)
AR = LY / L
dx_, dy_ = LX / 2000, LY / 2000
gx_, gy_ = np.meshgrid((np.arange(2000) + 0.5) * dx_, (np.arange(2000) + 0.5) * dy_)
Q_FLUX = POWER / (footprint(gx_, gy_).sum() * dx_ * dy_)          # W/m2 under the device
torch.manual_seed(0)
spec = PINNProblemSpec(
    pde_residuals=["Derivative(T(xi, eta), (xi, 2)) + Derivative(T(xi, eta), (eta, 2))"
                   " - c_h*h*T(xi, eta) + c_q*S(xi, eta)"],
    conditions=[{"name": "left", "equation": "Derivative(T(xi, eta), xi)"},      # adiabatic edges
                {"name": "right", "equation": "Derivative(T(xi, eta), xi)"},
                {"name": "bottom", "equation": "Derivative(T(xi, eta), eta)"},
                {"name": "top", "equation": "Derivative(T(xi, eta), eta)"}],
    independent_vars=["xi", "eta"], dependent_vars=["T"], inverse_params=["h"],
    constants={"c_h": 2 * L**2 / (K * THK), "c_q": Q_FLUX * L**2 / (K * THK * DT)},
    exogenous={"S": lambda xi, eta: footprint(xi * L, eta * L)},                  # the device, as a function
    loss_weights={"pde": 1.0, "conditions": 1.0, "data": 50.0},
)
loss_fn = PINNFactory(spec).generate_loss_function()
net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=4, num_neurons=48, activation=torch.nn.Tanh())
model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})
opt = torch.optim.Adam([{"params": net.parameters(), "lr": 2e-3},
                        {"params": model.inverse_params.parameters(), "lr": 0.2 * H_GUESS / 60}])
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, STEPS // 3), gamma=0.3)
gen = torch.Generator().manual_seed(0)
rand = lambda n, hi=1.0: hi * torch.rand(n, 1, generator=gen)  # noqa: E731


def collocation(n):
    """Uniform points plus a third of them around the device, where the source changes fastest."""
    m, (xc, yc, half) = n // 3, (v / L for v in DEVICE[:3])
    xi = torch.cat([rand(n - m), xc + (2 * rand(m) - 1) * 1.6 * half]).clamp(0, 1)
    eta = torch.cat([rand(n - m, AR), yc + (2 * rand(m) - 1) * 1.6 * half]).clamp(0, AR)
    return xi.requires_grad_(True), eta.requires_grad_(True)


def edge(n, along_eta, value):
    s, v = rand(n), torch.full((n, 1), value)
    return tuple(t.requires_grad_(True) for t in ((v, s * AR) if along_eta else (s, v)))


s_xy = torch.tensor(SENSORS_MM, dtype=torch.float32) / 1000 / L
data = ((s_xy[:, :1], s_xy[:, 1:]), torch.tensor((np.array(T_READ) - T_AIR) / DT, dtype=torch.float32).reshape(-1, 1))
for step in range(STEPS + 1):
    if step % 250 == 0:                                            # fresh collocation points
        batch = {"collocation": collocation(2500), "data": data,
                 "conditions": [edge(150, True, 0.0), edge(150, True, 1.0), edge(150, False, 0.0), edge(150, False, AR)]}
    opt.zero_grad()
    loss, _ = loss_fn(model, batch)
    loss.backward()
    opt.step()
    sched.step()
    with torch.no_grad():
        model.inverse_params["h"].clamp_(min=0.1)
    if step % 250 == 0:
        print(f"step {step:5d}  h = {model.inverse_params['h'].item():8.2f} W/m2K")

# --- results and the finite-volume check ---------------------------------------------------------------------------
h_pinn = model.inverse_params["h"].item()
xx, yy = np.meshgrid(x, y)
with torch.no_grad():
    t_pinn = T_AIR + DT * model(torch.tensor(xx.reshape(-1, 1) / L, dtype=torch.float32),
                                torch.tensor(yy.reshape(-1, 1) / L, dtype=torch.float32)).numpy().reshape(t_ref.shape)
_, _, t_chk = plate_fv(h_pinn)                                  # the same plate, solved with the learned h
print(f"\nhot spot (never measured): {t_pinn.max():.1f} °C")
print(f"finite volumes with the learned h: hot spot {t_chk.max():.1f} °C, worst difference {np.abs(t_pinn - t_chk).max():.2f} °C")
if DEMO:
    print(f"demo check: true h = {H_TRUE}, hot spot {t_ref.max():.1f} °C, worst map error {np.abs(t_pinn - t_ref).max():.2f} °C")
print(f"h = {h_pinn:.1f} W/m2K")
