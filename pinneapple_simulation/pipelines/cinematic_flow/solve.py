"""The solve half of a film: place the tunnel, voxelize the body, run the flow, write what the render needs.

The box is measured in lengths of the body, so the same numbers work for a 4 m car and a 12 m launcher. The
velocity is integrated in the lattice, and at every requested instant the vorticity magnitude and the speed
deficit are written — subsampled, because a volume render reads a smooth field and a factor of two costs a
quarter of the file for no visible difference.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from .cases import Case

__all__ = ["Tunnel", "plan_tunnel", "voxelize", "run_tunnel", "speed_of_sound_units"]


@dataclass
class Tunnel:
    """The box a case is solved in, and the numbers a caption has to quote with the picture."""

    shape: tuple[int, int, int]
    dx: float
    origin: tuple[float, float, float]
    u_lattice: float
    nu: float
    re: float
    steps: int
    cells_per_length: int
    upstream: float
    downstream: float

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))

    def report(self, case: Case, *, u_inf: float = 30.0, nu_air: float = 1.5e-5) -> dict:
        """What the run is, in the units of both the lattice and the wind it stands for."""
        return {
            "caso": case.name,
            "descricao": case.caption,
            "malha": list(self.shape),
            "celulas": self.cells,
            "dx_m": self.dx,
            "celulas_por_comprimento": self.cells_per_length,
            "passos": self.steps,
            "u_lattice": self.u_lattice,
            "re_lattice": self.re,
            "u_inf_ms": u_inf,
            "re_fisico": u_inf * case.length / nu_air,
            "comprimento_m": case.length,
            "tunel": {"a_montante": self.upstream, "a_jusante": self.downstream},
        }


def plan_tunnel(case: Case, *, cells_per_length: int = 140, steps: int = 2400, u_lattice: float = 0.08,
                re: float = 6000.0) -> Tunnel:
    """The box around ``case`` and the viscosity that reaches Reynolds ``re`` in lattice units.

    ``re`` is capped by what the collision model can hold at that resolution (``tau`` stays off 0.5); the cap is
    recorded in the tunnel, not silently applied, because a caption that claims the requested number when the
    grid could not reach it would be wrong."""
    lo, hi = case.bounds
    length = case.length
    # the room behind the body is measured on its largest dimension: a wing is a short chord and a long span,
    # and its wake is as long as the span, not as the chord
    ref = max(length, float(np.max(hi - lo)))
    dx = ref / max(int(cells_per_length), 8)
    width = float(hi[1] - lo[1])
    height = float(hi[2] - lo[2])
    x0, x1 = lo[0] - case.upstream * ref, hi[0] + case.downstream * ref
    y_half = 0.5 * width + case.lateral * 0.5 * width
    z0 = case.ground if case.ground is not None else lo[2] - case.overhead * height
    z1 = hi[2] + case.overhead * height
    shape = (int(round((x1 - x0) / dx)), int(round(2 * y_half / dx)), int(round((z1 - z0) / dx)))
    re_max = u_lattice * cells_per_length * 3.0 / (0.505 - 0.5)
    re_used = float(min(re, re_max))
    nu = u_lattice * cells_per_length / max(re_used, 1e-9)
    return Tunnel(shape, dx, (x0, -y_half, z0), u_lattice, nu, re_used, int(steps),
                  int(cells_per_length), case.upstream, case.downstream)


def _fill_sealed(mask: np.ndarray) -> np.ndarray:
    """Fill fluid pockets the inlet cannot reach (a cabin, the inside of a rim grid) — they are solid, not flow."""
    from scipy import ndimage

    fluid = ~mask
    lab, n = ndimage.label(fluid)
    if n <= 1:
        return mask
    inlet = np.unique(lab[0])
    inlet = inlet[inlet > 0]
    reachable = np.isin(lab, inlet)
    return mask | (fluid & ~reachable)


def voxelize(case: Case, tunnel: Tunnel) -> np.ndarray:
    """Solid mask of the union of the case's closed surfaces, on the tunnel grid."""
    import trimesh

    from pinneapple_design.geometry.ops.lbm_bridge import mesh_to_obstacle_mask

    x0, y0, z0 = tunnel.origin
    nx, ny, nz = tunnel.shape
    bounds = {"x": (x0, x0 + nx * tunnel.dx), "y": (y0, y0 + ny * tunnel.dx), "z": (z0, z0 + nz * tunnel.dx)}
    mask = np.zeros(tunnel.shape, dtype=bool)
    for verts, faces in case.bodies.values():
        mesh = trimesh.Trimesh(vertices=np.asarray(verts, float), faces=np.asarray(faces, int), process=False)
        mask |= np.asarray(mesh_to_obstacle_mask(mesh, bounds, tunnel.shape, oracle_supersample=1), dtype=bool)
    return _fill_sealed(mask)


def q_criterion(u: np.ndarray, dx: float) -> np.ndarray:
    """``Q = (|Ω|² - |S|²)/2`` of the velocity gradient, positive inside a vortex core.

    Vorticity magnitude alone will not do here: at a wall it is the boundary layer, and the road under a car
    carries more of it than the wake does, so a picture built on |ω| paints the floor and leaves the wake empty.
    Q is zero for pure shear and positive only where the rotation beats the strain, which is exactly the
    structure a flow picture is trying to show.
    """
    g = [[np.gradient(u[i], dx, axis=j) for j in range(3)] for i in range(3)]   # g[i][j] = du_i/dx_j
    s2 = 0.0
    w2 = 0.0
    for i in range(3):
        for j in range(3):
            s = 0.5 * (g[i][j] + g[j][i])
            w = 0.5 * (g[i][j] - g[j][i])
            s2 = s2 + s * s
            w2 = w2 + w * w
    return 0.5 * (w2 - s2)


def density_from(speed: np.ndarray, q: np.ndarray, *, percentile: float = 99.0, floor: float = 0.15,
                 vorticity_weight: float = 1.0, deficit_weight: float = 0.35) -> np.ndarray:
    """The scalar the smoke shows, from the two stored fields (both already normalised).

    ``speed`` is |u|/U∞ (1 in the free stream) and ``q`` is the Q-criterion in units of (U/dx)². Only the
    positive part of Q is used — negative Q is strain, which is the wall, not the vortex — and the reference is
    a high percentile over the fluid, because a lattice leaves a low-level floor of Q everywhere and without the
    cut the whole tunnel renders as fog.
    """
    fluid = speed > 1e-6
    pos = np.maximum(q, 0.0)
    ref = float(np.percentile(pos[fluid], percentile)) if fluid.any() else 1.0
    on = np.clip(pos / max(ref, 1e-12), 0.0, 1.0)
    deficit = np.clip(1.0 - speed, 0.0, 1.0)
    d = vorticity_weight * on + deficit_weight * deficit
    d = np.clip((d - floor) / max(1.0 - floor, 1e-9), 0.0, 1.0)
    return (d * fluid).astype(np.float32)


def run_tunnel(case: Case, tunnel: Tunnel, *, device: str = "mps", snapshots: int = 12, stride: int = 1,
               out_dir: str | None = None, verbose: bool = True) -> dict:
    """Run the flow and write ``snap_XXX.npz`` (density, speed, vorticity) plus ``meta.json``.

    The snapshots are taken after the wake has formed (the first third of the run is discarded as start-up), so
    every frame of the film is a developed flow and not a transient from rest."""
    from pinneapple_simulation.numerical_solvers.lbm_tunnel import WindTunnelLBM

    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    faces = {"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall", "z+": "wall"}
    faces["z-"] = "moving" if case.ground is not None else "wall"
    mask = voxelize(case, tunnel)
    if not mask.any():
        raise RuntimeError("a malha não resolve o corpo: aumente cells_per_length")
    tunnel_idx = WindTunnelLBM(tunnel.shape, nu=tunnel.nu, u_in=(tunnel.u_lattice, 0.0, 0.0),
                               smagorinsky=0.14, device=device, faces=faces)
    tunnel_idx.add_solid(mask)

    total = tunnel.steps
    warmup = int(total * 0.35)
    marks = np.linspace(warmup, total, snapshots).astype(int)
    t0 = time.time()
    done = 0
    written = []
    for k, mark in enumerate(marks):
        tunnel_idx.run(int(mark) - done)
        done = int(mark)
        u = tunnel_idx.fields()["u"]                               # (3, nx, ny, nz)
        bad = int(np.isnan(u).sum())
        if bad:
            raise RuntimeError(
                f"o solver divergiu no passo {done} ({bad} valores NaN): baixe --re ou --u-lattice, "
                f"ou aumente a viscosidade; os instantes já escritos em {out_dir} não servem")
        sub = (slice(None, None, stride),) * 3
        speed = (np.linalg.norm(u, axis=0)[sub] / tunnel.u_lattice).astype(np.float32)
        # Q in the same units as the speed: Q (dx/U)^2, so a threshold reads the same on every case
        q = (q_criterion(u, tunnel.dx) * (tunnel.dx / tunnel.u_lattice) ** 2)[sub].astype(np.float32)
        path = os.path.join(out_dir, f"snap_{k:03d}.npz")
        np.savez_compressed(path, speed=speed, q=q)
        written.append(path)
        if verbose:
            el = time.time() - t0
            print(f"  snapshot {k + 1}/{snapshots}  passo {done}/{total}  {el:.0f}s", flush=True)
    meta = {
        "tunnel": asdict(tunnel),
        "report": tunnel.report(case),
        "shape_sub": [int(np.ceil(s / stride)) for s in tunnel.shape],
        "stride": stride,
        "snapshots": written,
        "warmup_steps": warmup,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def speed_of_sound_units() -> float:
    """Lattice sound speed of the D2Q9/D3Q19 set (``1/sqrt(3)``) — the Mach the film runs at is quoted with it."""
    return 1.0 / np.sqrt(3.0)
