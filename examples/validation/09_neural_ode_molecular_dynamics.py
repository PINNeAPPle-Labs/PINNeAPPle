"""Neural ODEs for molecular dynamics: a Lennard-Jones dimer, where total energy must be conserved (NVE).

1. Reference: ``particle_dynamics.MolecularDynamicsSimulator`` (velocity Verlet, symplectic). Two models learn the
   vector field ``dz/dt`` of the state ``z = (x1, x2, v1, v2)`` from sampled states: a Hamiltonian network
   (``HamiltonianNeuralNetwork``: the field is the symplectic gradient of a learned H) and a plain ``NeuralODE`` (an
   unconstrained field); both are integrated with RK4 at the same step.
2. ``pp.validate(step, {"rollout": {..., "conserved": ["energy"], "invariants": {"energy": E}}})``: the rollout
   from a fresh state against the MD reference, energy drift, divergence time.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: the plain neural ODE's energy drifts (``rollout.invariant.energy``, 19 % over 10 time units)
   although its training loss is lower than the Hamiltonian network's, whose energy only oscillates (2 %).

Run: ``python examples/validation/09_neural_ode_molecular_dynamics.py [--full]``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.architectures.continuous.hamiltonian import HamiltonianNeuralNetwork
from pinneapple_neural.architectures.continuous.neural_ode import NeuralODE
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.particle_dynamics.molecular_dynamics import (
    MDState,
    MolecularDynamicsSimulator,
)

FAMILY, PHYSICS, CATCHES = "neural-ode", "particle-dynamics", "rollout.invariant"
MD = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0, mass=1.0, dim=1)
FRAME, SUB = 0.05, 5                                   # time between frames; model sub-steps per frame


def state(z) -> MDState:
    z = torch.as_tensor(np.asarray(z, float), dtype=torch.float64)
    return MDState(pos=z[:2, None], vel=z[2:, None])


def energy(traj, ndim=0):
    return np.array([float(MD.total_energy(state(z))) for z in np.asarray(traj).reshape(len(traj), 4)])


def field(z):                                           # exact dz/dt = (v, F / m)
    s = state(z)
    return np.concatenate([s.vel[:, 0].numpy(), MD.compute_forces(s.pos)[:, 0].numpy()])


def md_trajectory(z0, frames):
    s, out = state(z0), [np.asarray(z0, float)]
    for _ in range(frames):
        s = MD.forward(s, n_steps=int(FRAME / 1e-3), dt=1e-3)
        out.append(np.concatenate([s.pos[:, 0].numpy(), s.vel[:, 0].numpy()]))
    return np.stack(out)


def samples(n, rng):
    x1 = rng.uniform(-1, 1, n)
    r = rng.uniform(1.0, 1.45, n)
    v = rng.uniform(-0.6, 0.6, (n, 2))
    Z = np.stack([x1, x1 + r, v[:, 0], v[:, 1]], 1)
    return Z, np.stack([field(z) for z in Z])


def run(smoke: bool = True, out_dir=None):
    n, steps, frames = (1500, 3000, 200) if smoke else (6000, 10000, 1000)
    rng = np.random.default_rng(0)
    Z, dZ = samples(n, rng)
    torch.manual_seed(0)
    hnn = HamiltonianNeuralNetwork(dim_q=2, hidden=64, num_layers=3)
    fit_supervised(hnn, Z, dZ, steps=steps, lr=2e-3, batch=128, forward=lambda m, x: m(x).y)
    torch.manual_seed(0)
    node = NeuralODE(state_dim=4, hidden=64, num_layers=3)
    fit_supervised(node, Z, dZ, steps=steps, lr=2e-3, batch=128,
                   forward=lambda m, x: m.dynamics(torch.zeros(()), x))

    def f(y):                                                           # the learned Hamiltonian field
        return hnn(y).y.detach()

    def hnn_step(z):
        x, h = torch.tensor(z[None], dtype=torch.float32), FRAME / SUB
        for _ in range(SUB):                                            # RK4
            k1 = f(x)
            k2 = f(x + h / 2 * k1)
            k3 = f(x + h / 2 * k2)
            k4 = f(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x[0].numpy()

    def node_step(z):
        with torch.no_grad():
            t = torch.linspace(0, FRAME, SUB + 1)
            return node(torch.tensor(z[None], dtype=torch.float32), t).y[0, -1].numpy()
    z0 = np.array([0.0, 1.15, 0.3, -0.3])                               # a fresh state (zero total momentum)
    problem = {"rollout": {"ndim": 0, "dt": FRAME, "conserved": ["energy"], "invariants": {"energy": energy},
                           "steps": frames, "tolerances": {"drift": 0.05, "invariants": 0.05}},
               "name": "Lennard-Jones dimer, NVE"}
    data = {"u0": z0, "reference": md_trajectory(z0, frames)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(hnn_step, problem, data=data, level="standard")
        bad = pp.validate(node_step, problem, data=data, level="standard")
    return conclude("09_neural_ode_molecular_dynamics", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
