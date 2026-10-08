"""Neural operator on Field data: learn the solution operator f -> u of -u'' = f.

Training pairs live on a 1D grid Field. Field.gradient measures the derivative error of the
prediction, Field.integrate its energy, and Field.interpolate moves a sample to a finer
grid to test resolution transfer (the Fourier operator is mesh-independent).

    python examples/core_primitives/02_operator_poisson_1d.py
"""
import numpy as np
import torch

from pinneapple_core import Field
from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator

rng = np.random.default_rng(0)
torch.manual_seed(0)
K = np.arange(1, 6)


def make_pairs(n, grid):
    a = rng.normal(size=(n, K.size)) / K**1.5
    modes = np.sin(np.pi * np.outer(K, grid))  # (K, N)
    u = a @ modes
    f = (a * (np.pi * K) ** 2) @ modes  # f = -u''
    return f, u


x = np.linspace(0, 1, 128)
f_tr, u_tr = make_pairs(400, x)
f_te, u_te = make_pairs(100, x)
t = lambda a: torch.tensor(a[:, None, :], dtype=torch.float32)

model = FourierNeuralOperator(1, 1, width=32, modes=8, layers=3, use_grid=True)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
scale = f_tr.std()
for epoch in range(101):
    perm = torch.randperm(f_tr.shape[0])
    for idx in perm.split(64):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(t(f_tr)[idx] / scale).y, t(u_tr)[idx])
        loss.backward()
        opt.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch:4d}  loss {loss.item():.3e}")


def predict(f, grid):
    with torch.no_grad():
        return model(t(f) / scale).y[:, 0].numpy()


pred = predict(f_te, x)
rel = np.linalg.norm(pred - u_te, axis=1) / np.linalg.norm(u_te, axis=1)
print(f"test relative L2 error (128 pts)   {rel.mean():.3e}")

# Derivative of the prediction against the derivative of the truth, via Field.gradient
du = lambda v: Field.on_grid([x], v).gradient().values[:, 0]
grad_err = [np.linalg.norm(du(p) - du(u)) / np.linalg.norm(du(u)) for p, u in zip(pred, u_te)]
print(f"mean relative du/dx error          {np.mean(grad_err):.3e}")
energy = Field.on_grid([x], 0.5 * pred[0] * f_te[0]).integrate()
print(f"energy 1/2 int(u f) of sample 0    {energy:.4f}")

# Resolution transfer: resample f on 4x finer grid with Field.interpolate
x_fine = np.linspace(0, 1, 512)
f_fine = np.stack([Field.on_grid([x], f).interpolate(x_fine[:, None]) for f in f_te])
pred_fine = predict(f_fine, x_fine)
pred_on_coarse = np.stack([Field.on_grid([x_fine], p).interpolate(x[:, None]) for p in pred_fine])
rel_fine = np.linalg.norm(pred_on_coarse - u_te, axis=1) / np.linalg.norm(u_te, axis=1)
print(f"test relative L2 error (512 pts, trained on 128)  {rel_fine.mean():.3e}")
