"""3D: how hot is the chip nobody can reach? h of a fan from nine thermocouples on top of a block, and the chip
temperature under it, with a physics-informed network (PINNeAPPle).

    pip install pinneapple          (then: python block_chip_temperature.py; ~30 min on a CPU, minutes on a GPU)

A chip heats the bottom of a steel block, a fan cools the top with an unknown h, thermocouples sit just under the top
face. The PINN learns T(x, y, z) and h together from Laplace's equation, the chip flux, the convective top, the
adiabatic sides (built into the network) and the readings:

    T_xx + T_yy + T_zz = 0,   -k T_z = q''(x, y) at z = 0,   -k T_z = h (T - T_air) at z = H.

Two values of h are printed: the network's own parameter, and the energy balance on the learned field (all the chip's
power leaves through the top, h = P / integral of (T_top - T_air)). In this thick conductive block the field is pinned
down well by the readings and the energy balance lands on h; the parameter settles a few percent low.
Check: an independent finite-volume solution of the same block (also used to make demo readings).
"""
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.interpolate import RegularGridInterpolator
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, PINNFactory, PINNProblemSpec

# --- your block -----------------------------------------------------------------------------------------------------
LX, LY, LZ = 0.040, 0.040, 0.010       # block, m
K = 50.0                               # conductivity, W/m.K (carbon steel)
POWER = 10.0                           # chip power, W
T_AIR = 25.0                           # air, °C
DEVICE = (0.020, 0.020, 0.005, 0.0025)  # chip centre x, centre y, half side, edge smoothing (m)
SENSORS_MM = [[x, y] for y in (8, 20, 32) for x in (8, 20, 32)]   # on the top face
SENSOR_DEPTH_MM = 0.417                # how far under the top face they sit
T_READ = None                          # your readings, °C, one per sensor (None: demo readings from H_TRUE)
H_TRUE = 150.0                         # demo readings and the check, W/m2.K
NOISE = 0.5                            # demo readings only, ± °C
H_GUESS = 40.0                         # the network's starting value for h, W/m2.K
STEPS = 3000                           # Adam steps
LBFGS_ITERS = 800                      # L-BFGS polish iterations


def footprint(x, y):
    xc, yc, half, w = DEVICE
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh
    sig = lambda s: 0.5 * (1 + tanh(s / w))  # noqa: E731
    return sig(half - abs(x - xc)) * sig(half - abs(y - yc))


def block_fv(h, nx=40, ny=40, nz=12):
    """Finite-volume temperature (°C) at the cell centres; returns x, y, z, T[nz, ny, nx]."""
    dx, dy, dz = LX / nx, LY / ny, LZ / nz
    x, y, z = ((np.arange(n) + 0.5) * d for n, d in ((nx, dx), (ny, dy), (nz, dz)))
    fp = footprint(*np.meshgrid(x, y))
    q = POWER / (fp.sum() * dx * dy) * fp
    idx = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    diag, rhs = np.zeros(idx.size), np.zeros(idx.size)
    rows, cols, vals = [], [], []
    for a, b, g in ((idx[:, :, :-1], idx[:, :, 1:], K * dy * dz / dx), (idx[:, :-1, :], idx[:, 1:, :], K * dx * dz / dy),
                    (idx[:-1], idx[1:], K * dx * dy / dz)):
        for p, n in ((a.ravel(), b.ravel()), (b.ravel(), a.ravel())):
            rows.append(p); cols.append(n); vals.append(np.full(p.size, -g))  # noqa: E702
            np.add.at(diag, p, g)
    g_top = h * dx * dy / (1 + h * dz / (2 * K))                  # half cell + air film in series
    diag[idx[-1].ravel()] += g_top
    rhs[idx[-1].ravel()] += g_top * T_AIR
    rhs[idx[0].ravel()] += (q * dx * dy).ravel()                  # chip flux into the bottom cells
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag)  # noqa: E702
    a = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(idx.size,) * 2)
    return x, y, z, spla.spsolve(a, rhs).reshape(nz, ny, nx)


DEMO = T_READ is None
x, y, z, t_ref = block_fv(H_TRUE)
if DEMO:                               # readings at the top cell centres (SENSOR_DEPTH_MM under the face)
    s = np.array(SENSORS_MM, float) / 1000
    t_top = RegularGridInterpolator((y, x), t_ref[-1], bounds_error=False, fill_value=None)(s[:, ::-1])
    T_READ = (t_top + np.random.default_rng(0).normal(0, NOISE, len(s))).round(2).tolist()
    print("demo readings:", T_READ)

# --- the physics as equations (xi = x/L, ..., T here is (T - T_air) / dT) -------------------------------------------
L, DT, ZH = LX, 50.0, LZ / LX
xs_, ys_ = np.meshgrid((np.arange(2000) + 0.5) * LX / 2000, (np.arange(2000) + 0.5) * LY / 2000)
Q_FLUX = POWER / (footprint(xs_, ys_).sum() * LX * LY / 2000**2)
d = lambda v: f"Derivative(T(xi, eta, zeta), {v})"  # noqa: E731
spec = PINNProblemSpec(
    pde_residuals=[" + ".join(f"Derivative(T(xi, eta, zeta), ({v}, 2))" for v in ("xi", "eta", "zeta"))],  # Laplace
    conditions=[{"name": "chip (bottom)", "equation": f"{d('zeta')} + c_q*S(xi, eta, zeta)", "weight": 10.0},
                {"name": "fan (top)", "equation": f"{d('zeta')} + c_t*h*T(xi, eta, zeta)", "weight": 10.0}],
    independent_vars=["xi", "eta", "zeta"], dependent_vars=["T"], inverse_params=["h"],
    constants={"c_q": Q_FLUX * L / (K * DT), "c_t": L / K},
    exogenous={"S": lambda xi, eta, *_: footprint(xi * L, eta * L)},
    loss_weights={"pde": 5.0, "conditions": 10.0, "data": 50.0},
)
loss_fn = PINNFactory(spec).generate_loss_function()


class SideAdiabaticNet(torch.nn.Module):
    """Adiabatic sides built in: the inputs cos(pi xi), cos(pi eta) have zero slope at xi, eta = 0 and 1."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, n = [], 3
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.Tanh()]
            n = width
        self.mlp = torch.nn.Sequential(*layers, torch.nn.Linear(n, 1))

    def forward(self, x):
        return self.mlp(torch.cat([torch.cos(math.pi * x[:, :1]), torch.cos(math.pi * x[:, 1:2]), 4 * x[:, 2:3]], 1))


torch.manual_seed(0)
net = SideAdiabaticNet()
model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})
gen = torch.Generator().manual_seed(0)
rand = lambda n, hi=1.0: hi * torch.rand(n, 1, generator=gen)  # noqa: E731
XC, YC, HALF = (v / L for v in DEVICE[:3])


def near_chip(n, m):
    xi = torch.cat([rand(n - m), XC + (2 * rand(m) - 1) * 1.8 * HALF]).clamp(0, 1)
    eta = torch.cat([rand(n - m), YC + (2 * rand(m) - 1) * 1.8 * HALF]).clamp(0, 1)
    return xi, eta


def make_batch(n_in, n_bottom, n_top):
    xi, eta = near_chip(n_in, n_in // 3)                          # a third of the points under/around the chip
    zeta = torch.cat([rand(n_in - n_in // 3, ZH), rand(n_in // 3, ZH / 2)])
    bx, by = near_chip(n_bottom, n_bottom // 2)
    top = (rand(n_top), rand(n_top), torch.full((n_top, 1), ZH))
    grad = lambda *t: tuple(v.requires_grad_(True) for v in t)  # noqa: E731
    return {"collocation": grad(xi, eta, zeta), "conditions": [grad(bx, by, torch.zeros_like(bx)), grad(*top)],
            "data": data}


s = torch.tensor(SENSORS_MM, dtype=torch.float32) / 1000 / L
data = ((s[:, :1], s[:, 1:], torch.full((len(s), 1), (LZ - SENSOR_DEPTH_MM / 1000) / L)),
        torch.tensor((np.array(T_READ) - T_AIR) / DT, dtype=torch.float32).reshape(-1, 1))
opt = torch.optim.Adam([{"params": net.parameters(), "lr": 2e-3},
                        {"params": model.inverse_params.parameters(), "lr": 0.5}])
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, STEPS * 13 // 30), gamma=0.3)
for step in range(STEPS + 1):
    if step % 250 == 0:
        batch = make_batch(2500, 600, 400)
    opt.zero_grad()
    loss, _ = loss_fn(model, batch)
    loss.backward()
    opt.step()
    sched.step()
    if step % 250 == 0:
        print(f"step {step:5d}  h = {model.inverse_params['h'].item():8.2f} W/m2K")

# L-BFGS polish on a larger fixed batch: Adam leaves a residual that leaks heat, and h follows it
batch = make_batch(8000, 1500, 800)
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, history_size=50,
                          line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)


def closure():
    lbfgs.zero_grad()
    loss, _ = loss_fn(model, batch)
    loss.backward()
    return loss


lbfgs.step(closure)

# --- results and the finite-volume check ----------------------------------------------------------------------------
h_pinn = model.inverse_params["h"].item()
zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
with torch.no_grad():
    t_pinn = T_AIR + DT * model(*(torch.tensor(a.reshape(-1, 1) / L, dtype=torch.float32) for a in (xx, yy, zz))
                                ).numpy().reshape(t_ref.shape)
    g1 = (torch.arange(80, dtype=torch.float32) + 0.5) / 80
    gx, gy = torch.meshgrid(g1, g1, indexing="ij")
    top = model(gx.reshape(-1, 1), gy.reshape(-1, 1), torch.full((6400, 1), ZH))
h_energy = POWER / (float(top.mean()) * DT * LX * LY)               # energy balance on the learned top face
print(f"\nchip temperature (never measured): {t_pinn[0].max():.1f} °C")
print(f"h from the energy balance on the learned field: {h_energy:.1f} W/m2K")
if DEMO:
    print(f"demo check: true h = {H_TRUE}, chip {t_ref[0].max():.1f} °C (finite volumes), "
          f"worst field error {np.abs(t_pinn - t_ref).max():.2f} °C")
print(f"h = {h_pinn:.1f} W/m2K (network parameter)")
