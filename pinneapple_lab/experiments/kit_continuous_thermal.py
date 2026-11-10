"""Continuous-time lumped thermal physics for the KIT benchmark.

Physical equation: C dT/dt = H (T_ambient - T) + eta Q + P_bias.
This is a lumped ODE, not a spatial heat-transfer PDE. SI time is seconds;
Q is delivered heat in watts. Positivity of C, H and eta is enforced.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class LumpedThermalPINN(nn.Module):
    """Time-coordinate PINN with autograd physical residual.

    Inputs to forward are time in seconds, ambient temperature in Celsius,
    and delivered heat in watts. Time normalization only changes the neural
    input; the autograd derivative is taken w.r.t. physical seconds.
    """
    def __init__(self, time_scale_seconds: float, width: int = 48):
        super().__init__()
        if time_scale_seconds <= 0:
            raise ValueError("time_scale_seconds must be positive")
        self.time_scale = float(time_scale_seconds)
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )
        self.log_c = nn.Parameter(torch.tensor(13.0))
        self.log_h = nn.Parameter(torch.tensor(0.0))
        self.log_eta = nn.Parameter(torch.tensor(0.0))
        self.bias_w = nn.Parameter(torch.tensor(0.0))

    def physical_parameters(self):
        return {
            "capacity_j_per_k": torch.exp(self.log_c),
            "conductance_w_per_k": torch.exp(self.log_h),
            "heating_efficiency": torch.sigmoid(self.log_eta),
            "bias_w": self.bias_w,
        }

    def forward(self, time_seconds, ambient_c, heat_w):
        x = torch.cat((time_seconds / self.time_scale,
                       ambient_c / 20.0, heat_w / 1000.0), dim=-1)
        return self.net(x)

    def residual(self, time_seconds, ambient_c, heat_w):
        if not time_seconds.requires_grad:
            raise ValueError("time_seconds must require grad for physics residual")
        temp = self.forward(time_seconds, ambient_c, heat_w)
        dtdt = torch.autograd.grad(
            temp.sum(), time_seconds, create_graph=True
        )[0]
        p = self.physical_parameters()
        # Divide by C to express residual as kelvin per second.
        rhs = (p["conductance_w_per_k"] * (ambient_c - temp)
               + p["heating_efficiency"] * heat_w + p["bias_w"])
        return dtdt - rhs / p["capacity_j_per_k"]


def train_continuous_pinn(time_s, ambient_c, heat_w, measured_c,
                          train_idx, val_idx, *, seed=0, epochs=400,
                          device="cpu", physics_weight=1.0):
    """Fit only training observations; select checkpoint on validation.

    Residual is scaled to kelvin/hour to balance against Celsius data loss.
    Does not access held-out test observations for optimization.
    """
    import numpy as np
    torch.manual_seed(int(seed))
    time_s = np.asarray(time_s, dtype=np.float64)
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time must increase")
    offset = float(time_s[0])
    span = float(time_s[-1] - offset)
    model = LumpedThermalPINN(span).to(device)
    t = torch.tensor((time_s-offset)[:,None],dtype=torch.float32,device=device)
    a = torch.tensor(np.asarray(ambient_c)[:,None],dtype=torch.float32,device=device)
    q = torch.tensor(np.asarray(heat_w)[:,None],dtype=torch.float32,device=device)
    y = torch.tensor(np.asarray(measured_c)[:,None],dtype=torch.float32,device=device)
    ti = torch.tensor(train_idx,device=device)
    vi = torch.tensor(val_idx,device=device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    best = float("inf")
    best_state = None
    hist = []
    for epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        tt = t[ti].detach().clone().requires_grad_(True)
        data_loss = (model(tt,a[ti],q[ti])-y[ti]).square().mean()
        physics_loss = (model.residual(tt,a[ti],q[ti])*3600).square().mean()
        loss = data_loss + float(physics_weight)*physics_loss
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = (model(t[vi],a[vi],q[vi])-y[vi]).square().mean().item()
        hist.append((float(data_loss.item()),float(physics_loss.item()),val_loss))
        if val_loss < best:
            best = val_loss
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("epochs must be positive")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(t,a,q).cpu().numpy().ravel()
        params = {k:float(v.item()) for k,v in model.physical_parameters().items()}
    return pred, np.asarray(hist), params
