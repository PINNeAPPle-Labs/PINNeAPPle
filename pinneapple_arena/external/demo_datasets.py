"""
Standardized synthetic dataset generators — one per physical-problem shape,
reused across many models' demo runs (`pinneapple_arena/external/models_data.py` points
each ModelSpec at one of these via `dataset_fn`).

Deliberately tiny (seconds, not minutes): these exist to prove a model can
be loaded and actually run end-to-end, not to benchmark it scientifically.
Ported unchanged from the original suite_core.py.
"""

from __future__ import annotations

import numpy as np

_RNG = np.random.RandomState(0)


def ds_pde_1d(n: int = 64) -> dict:
    x = np.linspace(-1, 1, n).reshape(-1, 1).astype(np.float64)
    return {"name": "pde_1d_collocation",
            "description": f"{n} collocation points in x∈[-1,1] for a generic 1D PDE/ODE forward pass.",
            "x": x, "n": n}


def ds_burgers(nx: int = 64, nt: int = 32) -> dict:
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    u0 = -np.sin(np.pi * x)
    return {"name": "burgers_1d",
            "description": f"Classic viscous Burgers' benchmark grid ({nx}x{nt}, x,t) with sine initial condition.",
            "x": x, "t": t, "X": X, "T": T, "u0": u0, "nx": nx, "nt": nt}


def ds_grid_2d(nx: int = 16, ny: int = 16) -> dict:
    field_ = _RNG.rand(nx, ny).astype(np.float32)
    return {"name": "grid_2d",
            "description": f"{nx}x{ny} synthetic scalar field for CFD/thermal/EM grid solvers.",
            "field": field_, "nx": nx, "ny": ny}


def ds_molecular(n_atoms: int = 6) -> dict:
    positions = _RNG.uniform(0, 5, size=(n_atoms, 3)).astype(np.float64)
    species = (np.array([1, 1, 8, 1, 1, 8] * ((n_atoms // 6) + 1))[:n_atoms]).astype(np.int64)
    box = np.array([10.0, 10.0, 10.0])
    return {"name": "molecular_system",
            "description": f"{n_atoms}-atom synthetic H/O system in a 10Å cubic box for MD potential testing.",
            "positions": positions, "species": species, "box": box, "n_atoms": n_atoms}


def ds_weather_grid(nlat: int = 16, nlon: int = 32, nlevels: int = 4) -> dict:
    temp = 273 + _RNG.rand(nlevels, nlat, nlon).astype(np.float32) * 30
    return {"name": "weather_grid",
            "description": f"{nlevels}x{nlat}x{nlon} synthetic atmospheric temperature field.",
            "temperature": temp, "nlat": nlat, "nlon": nlon, "nlevels": nlevels}


def ds_em_grid(nx: int = 20, ny: int = 20) -> dict:
    eps_r = np.ones((nx, ny), dtype=np.float64)
    eps_r[8:12, 8:12] = 4.0
    return {"name": "em_permittivity_grid",
            "description": f"{nx}x{ny} permittivity grid with a dielectric square inclusion.",
            "eps_r": eps_r, "nx": nx, "ny": ny}


def ds_seismic(nx: int = 32, nz: int = 32) -> dict:
    v = np.full((nz, nx), 1500.0, dtype=np.float32)
    v[nz // 2:, :] = 2500.0
    return {"name": "seismic_velocity_model",
            "description": f"{nz}x{nx} two-layer synthetic P-wave velocity model.",
            "velocity": v, "nx": nx, "nz": nz}


def ds_spin_lattice(n_sites: int = 6) -> dict:
    return {"name": "spin_chain",
            "description": f"{n_sites}-site spin-1/2 chain configuration.",
            "n_sites": n_sites}


def ds_power_system() -> dict:
    buses = np.array([[1, 1.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 0.0]])
    lines = np.array([[1, 2, 0.02], [2, 3, 0.03]])
    return {"name": "3bus_power_system",
            "description": "Synthetic 3-bus power system (voltage magnitude/angle, line reactance).",
            "buses": buses, "lines": lines}


def ds_traffic(nx: int = 50, nt: int = 50) -> dict:
    density = _RNG.rand(nt, nx).astype(np.float32)
    return {"name": "traffic_density_grid",
            "description": f"{nt}x{nx} synthetic traffic density space-time grid.",
            "density": density, "nx": nx, "nt": nt}


def ds_epidemic(n_days: int = 60) -> dict:
    t = np.arange(n_days)
    infected = 100 * np.exp(-0.5 * ((t - 30) / 10.0) ** 2) + _RNG.rand(n_days) * 5
    return {"name": "sir_epidemic_curve",
            "description": f"{n_days}-day synthetic infection curve (Gaussian outbreak + noise).",
            "t": t, "I": infected.astype(np.float64), "n_days": n_days}


def ds_combustion(n_species: int = 5, n_t: int = 50) -> dict:
    Y = _RNG.dirichlet(np.ones(n_species), size=n_t).astype(np.float32)
    temperature = 300 + _RNG.rand(n_t).astype(np.float32) * 2000
    return {"name": "combustion_state",
            "description": f"{n_t}-step synthetic species mass-fraction + temperature trajectory ({n_species} species).",
            "Y": Y, "temperature": temperature, "n_species": n_species, "n_t": n_t}


def ds_cosmology(n_particles: int = 200) -> dict:
    positions = _RNG.uniform(0, 25, size=(n_particles, 3)).astype(np.float32)
    return {"name": "nbody_particles",
            "description": f"{n_particles} dark-matter particles in a 25 Mpc/h synthetic box.",
            "positions": positions, "n_particles": n_particles}


def ds_room() -> dict:
    return {"name": "shoebox_room",
            "description": "5x4x3 m synthetic shoebox room for acoustic simulation.",
            "dims": [5.0, 4.0, 3.0]}


def ds_graph_mesh(n_nodes: int = 30) -> dict:
    positions = _RNG.rand(n_nodes, 2).astype(np.float32)
    edges = np.array([[i, (i + 1) % n_nodes] for i in range(n_nodes)])
    return {"name": "ring_mesh_graph",
            "description": f"{n_nodes}-node ring mesh graph with 2D node positions.",
            "positions": positions, "edges": edges, "n_nodes": n_nodes}


def ds_rigidbody(n: int = 8) -> dict:
    return {"name": "free_particle_state",
            "description": f"{n}-particle free-fall initial state for rigid-body/robotics engines.",
            "n": n}


def ds_image_field(size: int = 32) -> dict:
    img = _RNG.rand(size, size).astype(np.float32)
    return {"name": "scalar_image_field",
            "description": f"{size}x{size} synthetic scalar field image.",
            "field": img, "size": size}


def ds_timeseries(n: int = 100) -> dict:
    t = np.linspace(0, 10, n)
    y = np.sin(t) + _RNG.rand(n) * 0.1
    return {"name": "generic_timeseries",
            "description": f"{n}-point synthetic sensor time series.",
            "t": t, "y": y, "n": n}


def ds_pedestrians(n: int = 20) -> dict:
    state = np.zeros((n, 6))
    state[:, 0:2] = _RNG.uniform(0, 10, size=(n, 2))
    state[:, 2:4] = _RNG.uniform(-1, 1, size=(n, 2))
    state[:, 4:6] = _RNG.uniform(8, 10, size=(n, 2))
    return {"name": "pedestrian_crowd_state",
            "description": f"{n}-pedestrian synthetic state (position, velocity, destination).",
            "state": state, "n": n}


def ds_airfoil() -> dict:
    return {"name": "naca_airfoil",
            "description": "NACA 2412 airfoil at alpha=5deg, Re=1e6, Mach=0.2 — standard aerodynamic test case.",
            "naca": "2412", "alpha": 5.0, "Re": 1e6, "mach": 0.2}


def ds_option_pricing(n: int = 64) -> dict:
    S = np.linspace(50, 150, n)
    t = np.linspace(0, 1, n)
    K = 100.0
    r = 0.05
    sigma = 0.2
    return {"name": "black_scholes_grid",
            "description": f"{n}x{n} synthetic (S,t) grid for a European call, K={K}, r={r}, sigma={sigma}.",
            "S": S, "t": t, "K": K, "r": r, "sigma": sigma, "n": n}


def ds_lorenz(n_steps: int = 2000, dt: float = 0.01) -> dict:
    """Classic chaotic Lorenz-63 trajectory — used both to *discover* (PySINDy)
    and to *integrate forward* (Diffrax) the same governing ODE."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    xyz = np.empty((n_steps, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    for i in range(n_steps - 1):
        x, y, z = xyz[i]
        dxyz = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[i + 1] = xyz[i] + dxyz * dt
    t = np.arange(n_steps) * dt
    return {"name": "lorenz63_trajectory",
            "description": f"{n_steps}-step Lorenz-63 chaotic trajectory (dt={dt}), Euler-integrated from (1,1,1).",
            "t": t, "xyz": xyz, "n_steps": n_steps, "dt": dt,
            "sigma": sigma, "rho": rho, "beta": beta}


def ds_tle() -> dict:
    """A real, valid two-line element set (ISS, illustrative epoch) for
    orbit-propagation demos — TLE format needs a real-looking checksum/epoch
    to parse, so this is copied from a genuine public ISS TLE rather than
    invented."""
    return {"name": "iss_tle",
            "description": "Two-line element set for the ISS, used as a real orbit-propagation input.",
            "line1": "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9007",
            "line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560971 12345"}


def ds_quantum_circuit(n_qubits: int = 2) -> dict:
    return {"name": "bell_pair_circuit",
            "description": f"{n_qubits}-qubit circuit (H + CNOT ladder) for variational/quantum-circuit simulator demos.",
            "n_qubits": n_qubits}


def ds_crystal() -> dict:
    """A tiny synthetic cubic unit cell (plain arrays only — no pymatgen/ASE
    dependency here; the consuming infer_* function builds a real Structure
    object from these if the library needs one)."""
    lattice_a = 3.5
    species = ["Fe", "Fe"]
    frac_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return {"name": "bcc_iron_cell",
            "description": "Small synthetic BCC-iron-like cubic unit cell for materials-ML potential demos.",
            "lattice_a": lattice_a, "species": species, "frac_coords": frac_coords}


DATASETS = {
    name: fn for name, fn in list(globals().items())
    if name.startswith("ds_") and callable(fn)
}
