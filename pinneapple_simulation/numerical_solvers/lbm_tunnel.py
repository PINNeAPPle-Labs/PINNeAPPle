"""3-D lattice Boltzmann wind tunnel in PyTorch, with the memory and boundary techniques FluidX3D made popular.

Written from the published methods (no FluidX3D code is used):

- velocity sets D3Q19 and D3Q27; SRT (BGK) or TRT collision (magic parameter 3/16);
- Smagorinsky-Lilly subgrid model for high Reynolds numbers (Hou et al. 1994);
- **DDF shifting**: the distributions are stored as g_i = f_i - w_i, so the stored numbers are small deviations and keep
  their precision (Lehmann et al. 2022, "Accuracy and performance of the lattice Boltzmann method with 64-bit, 32-bit,
  and customized 16-bit number formats");
- **16-bit storage** ``storage="fp16s"``: the shifted distributions are kept in IEEE half precision, scaled so they stay
  in the normal range; all arithmetic is still FP32. Memory per cell drops from 4q to 2q bytes for the distributions;
- equilibrium boundaries for inflow and open sides (cells fixed to the equilibrium of a given density and velocity),
  a zero-gradient outflow, no-slip walls, a moving floor and moving solids (bounce-back with the wall-velocity term);
- a global volume force (Guo forcing), for pressure-driven ducts;
- the force on a solid by momentum exchange on its boundary links;
- STL voxelization (trimesh) and unit conversion between SI and lattice units;
- VTK export of the fields in the legacy structured-points layout FluidX3D also writes, so its output and this one are
  read by the same function (:func:`read_vtk`).

    from pinneapple_simulation.numerical_solvers.lbm_tunnel import WindTunnelLBM, Units, sphere
    lbm = WindTunnelLBM((160, 80, 80), nu=0.02, u_in=(0.05, 0, 0))
    lbm.add_solid(sphere(lbm.shape, center=(40, 40, 40), radius=8))
    lbm.run(4000)
    Fx, Fy, Fz = lbm.force_on_solid()
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["WindTunnelLBM", "Units", "velocity_set", "sphere", "box", "voxelize_stl", "write_vtk", "read_vtk",
           "drag_sphere_schiller_naumann", "stokes_sphere_field"]

FP16_SCALE = 1024.0          # shifted DDFs are O(1e-3..1e-1); x1024 keeps the small ones out of the FP16 subnormals


def velocity_set(name: str = "D3Q19"):
    """(c [q,3] int, w [q], opposite index [q]) for D3Q19 or D3Q27."""
    if name == "D3Q19":
        c = [(0, 0, 0)] + [v for v in _axes()] + [v for v in _edges()]
        wts = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
    elif name == "D3Q27":
        c = [(0, 0, 0)] + _axes() + _edges() + _corners()
        wts = [8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8
    else:
        raise ValueError("velocity set must be D3Q19 or D3Q27")
    c = np.array(c, dtype=np.int64)
    opp = np.array([next(j for j in range(len(c)) if (c[j] == -c[i]).all()) for i in range(len(c))])
    return c, np.array(wts), opp


def _axes():
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _edges():
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa in (1, -1):
            for sb in (1, -1):
                v = [0, 0, 0]
                v[a], v[b] = sa, sb
                out.append(tuple(v))
    return out


def _corners():
    return [(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]


# ------------------------------------------------------------------------------------------------- units
@dataclass
class Units:
    """Conversion between SI and lattice units, fixed by a length, a velocity and a density given in both
    systems (lattice length in cells, lattice velocity in cells/step, lattice density usually 1)."""
    m: float          # metres per lattice length unit
    s: float          # seconds per time step
    kg: float         # kilograms per lattice mass unit

    @classmethod
    def from_si(cls, lbm_length: float, lbm_velocity: float, lbm_rho: float, si_length: float, si_velocity: float,
                si_rho: float) -> Units:
        m = si_length / lbm_length
        s = lbm_velocity / si_velocity * m
        kg = si_rho / lbm_rho * m ** 3
        return cls(m, s, kg)

    # SI -> lattice
    def x(self, si): return si / self.m                                                   # noqa: E704
    def u(self, si): return si * self.s / self.m                                          # noqa: E704
    def t(self, si): return si / self.s                                                   # noqa: E704
    def nu(self, si): return si * self.s / self.m ** 2                                    # noqa: E704
    def rho(self, si): return si * self.m ** 3 / self.kg                                  # noqa: E704
    def F(self, si): return si * self.s ** 2 / (self.kg * self.m)                         # noqa: E704
    def f(self, si): return si * (self.m * self.s) ** 2 / self.kg                         # noqa: E704  force/volume

    # lattice -> SI
    def si_x(self, v): return v * self.m                                                  # noqa: E704
    def si_u(self, v): return v * self.m / self.s                                         # noqa: E704
    def si_t(self, v): return v * self.s                                                  # noqa: E704
    def si_F(self, v): return v * self.kg * self.m / self.s ** 2                          # noqa: E704
    def si_p(self, v): return v * self.kg / (self.m * self.s ** 2)                        # noqa: E704  pressure


# ------------------------------------------------------------------------------------------------- geometry
def sphere(shape, center, radius) -> torch.Tensor:
    """Solid mask of a sphere (grid indices ``center`` and ``radius``) on a grid of ``shape``."""
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return torch.as_tensor((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius ** 2)


def box(shape, lo, hi) -> torch.Tensor:
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return torch.as_tensor(m)


def voxelize_stl(mesh, shape, center, size: float, axis: int = 0) -> torch.Tensor:
    """Solid mask of a closed surface (path or trimesh.Trimesh), scaled so its extent along ``axis`` is ``size``
    cells and centred at ``center`` (cells). Inside test by ray parity (trimesh)."""
    import trimesh

    m = trimesh.load(mesh, force="mesh") if isinstance(mesh, (str, os.PathLike)) else mesh.copy()
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2)
    m.apply_scale(size / (hi - lo)[axis])
    m.apply_translation(np.asarray(center, float))
    blo, bhi = np.floor(m.bounds[0]).astype(int), np.ceil(m.bounds[1]).astype(int) + 1
    blo, bhi = np.maximum(blo, 0), np.minimum(bhi, shape)
    g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(blo, bhi, strict=True)], indexing="ij"), -1)
    inside = m.contains(g.reshape(-1, 3).astype(float)).reshape(g.shape[:3])
    out = np.zeros(shape, bool)
    out[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]] = inside
    return torch.as_tensor(out)


# ------------------------------------------------------------------------------------------------- solver
class WindTunnelLBM:
    """3-D lattice Boltzmann flow in a box (lattice units: cell = 1, step = 1, rho0 = 1).

    ``faces`` sets each side: ``"inlet"`` / ``"open"`` (equilibrium at rho = 1 and ``u_in``), ``"outlet"``
    (equilibrium at rho = 1 with the velocity of the next interior layer), ``"wall"`` (no slip), ``"moving"``
    (wall moving with ``u_in``, e.g. a road) or ``"periodic"``. Default: inlet at x-, outlet at x+, open sides.
    """

    FACES = ("x-", "x+", "y-", "y+", "z-", "z+")

    def __init__(self, shape: Sequence[int], nu: float, u_in: Sequence[float] = (0.0, 0.0, 0.0),
                 velocity_set_name: str = "D3Q19", collision: str = "srt", smagorinsky: float | None = None,
                 storage: str = "fp32", force: Sequence[float] = (0.0, 0.0, 0.0), faces: dict | None = None,
                 device: str | torch.device = "cpu"):
        if collision not in ("srt", "trt"):
            raise ValueError("collision must be 'srt' or 'trt'")
        if storage not in ("fp32", "fp16s"):
            raise ValueError("storage must be 'fp32' or 'fp16s'")
        self.shape = tuple(int(s) for s in shape)
        self.dev = torch.device(device)
        c, w, opp = velocity_set(velocity_set_name)
        self.q = len(w)
        self.c = torch.as_tensor(c, device=self.dev)
        self.cf = self.c.float()
        self.w = torch.as_tensor(w, dtype=torch.float32, device=self.dev)
        self.opp = torch.as_tensor(opp, device=self.dev)
        self.nu, self.tau = float(nu), 3.0 * float(nu) + 0.5
        if self.tau <= 0.5:
            raise ValueError("nu must be > 0")
        self.collision, self.cs_les, self.storage = collision, smagorinsky, storage
        self.u_in = torch.tensor(u_in, dtype=torch.float32, device=self.dev)
        self.F = torch.tensor(force, dtype=torch.float32, device=self.dev)
        f = {"x-": "inlet", "x+": "outlet", "y-": "open", "y+": "open", "z-": "open", "z+": "open"}
        f.update(faces or {})
        bad = [k for k in f if k not in self.FACES]
        if bad:
            raise ValueError(f"unknown faces {bad}")
        self.faces = f
        self.solid = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        self.u_solid = torch.zeros((3, *self.shape), device=self.dev)     # velocity of moving solids
        self._wall_masks()
        n = int(np.prod(self.shape))
        ar = torch.arange(n, device=self.dev).view(self.shape)
        self._src = torch.stack([torch.roll(ar, shifts=ci, dims=(0, 1, 2)).flatten() for ci in self.c.tolist()])
        self.t = 0
        rho = torch.ones(self.shape, device=self.dev)
        u = self.u_in.view(3, 1, 1, 1).expand(3, *self.shape).clone()
        self._store(self._feq(rho, u))

    # ---------------------------------------------------------------- storage (DDF shifting, FP16)
    def _store(self, f: torch.Tensor) -> None:
        g = f - self.w.view(-1, 1, 1, 1)
        self.g = (g * FP16_SCALE).half() if self.storage == "fp16s" else g.contiguous()

    def _load(self) -> torch.Tensor:
        g = self.g.float() / FP16_SCALE if self.storage == "fp16s" else self.g
        return g + self.w.view(-1, 1, 1, 1)

    def bytes_per_cell(self) -> int:
        """Memory of the distributions per cell (one copy; the solver keeps one stored copy between steps)."""
        return self.q * (2 if self.storage == "fp16s" else 4)

    # ---------------------------------------------------------------- geometry
    def _wall_masks(self) -> None:
        nx, ny, nz = self.shape
        self.eq_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.dev)
        self.out_mask = torch.zeros_like(self.eq_mask)
        self.wall = torch.zeros_like(self.eq_mask)
        self.moving = torch.zeros_like(self.eq_mask)
        sl = {"x-": (0, slice(0, 1)), "x+": (0, slice(nx - 1, nx)), "y-": (1, slice(0, 1)), "y+": (1, slice(ny - 1, ny)),
              "z-": (2, slice(0, 1)), "z+": (2, slice(nz - 1, nz))}
        for face, kind in self.faces.items():
            ax, s = sl[face]
            idx = [slice(None)] * 3
            idx[ax] = s
            idx = tuple(idx)
            if kind in ("inlet", "open"):
                self.eq_mask[idx] = True
            elif kind == "outlet":
                self.out_mask[idx] = True
            elif kind == "wall":
                self.wall[idx] = True
            elif kind == "moving":
                self.wall[idx] = True
                self.moving[idx] = True
            elif kind != "periodic":
                raise ValueError(f"face kind {kind!r}")
        self.eq_mask &= ~self.wall
        self.out_mask &= ~(self.wall | self.eq_mask)
        self.outlet_axis = 0 if self.faces.get("x+") == "outlet" else None

    def initialize(self, u: np.ndarray | torch.Tensor | None = None, rho: np.ndarray | torch.Tensor | None = None,
                   fix_boundaries: bool = True) -> None:
        """Start from a given velocity (3, nx, ny, nz) and density field. With ``fix_boundaries`` the equilibrium
        cells keep this velocity (instead of ``u_in``), e.g. an analytic far field."""
        u = self.u_in.view(3, 1, 1, 1).expand(3, *self.shape) if u is None else torch.as_tensor(u, dtype=torch.float32,
                                                                                                device=self.dev)
        rho = torch.ones(self.shape, device=self.dev) if rho is None else torch.as_tensor(rho, dtype=torch.float32,
                                                                                         device=self.dev)
        if fix_boundaries and self.eq_mask.any():
            self._feq_in = self._feq(rho, u)[:, self.eq_mask]
        self._store(self._feq(rho, u))

    def add_solid(self, mask: torch.Tensor, velocity: Sequence[float] | None = None) -> None:
        """Mark cells as solid (optionally moving with ``velocity``, lattice units)."""
        mask = torch.as_tensor(mask, device=self.dev).bool()
        self.solid |= mask
        if velocity is not None:
            self.u_solid[:, mask] = torch.tensor(velocity, dtype=torch.float32, device=self.dev).view(3, 1)

    @property
    def _all_solid(self) -> torch.Tensor:
        return self.solid | self.wall

    # ---------------------------------------------------------------- physics
    def _feq(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cu = torch.einsum("qd,dxyz->qxyz", self.cf, u)
        uu = (u * u).sum(0)
        return self.w.view(-1, 1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu)

    def macroscopic(self, f: torch.Tensor | None = None):
        f = self._load() if f is None else f
        rho = f.sum(0)
        j = torch.einsum("qd,qxyz->dxyz", self.cf, f) + 0.5 * self.F.view(3, 1, 1, 1)
        u = j / rho
        return rho, u

    def _collide(self, f: torch.Tensor) -> torch.Tensor:
        rho, u = self.macroscopic(f)
        feq = self._feq(rho, u)
        tau = self.tau
        if self.cs_les:                                               # Smagorinsky-Lilly (Hou et al. 1994)
            fneq = f - feq
            pi = torch.einsum("qa,qb,qxyz->abxyz", self.cf, self.cf, fneq)
            qn = torch.sqrt(2.0 * (pi * pi).sum((0, 1)))
            tau = 0.5 * (tau + torch.sqrt(tau * tau + 18.0 * math.sqrt(2.0) * self.cs_les ** 2 * qn / rho))
        taum = tau
        if self.collision == "srt":
            out = f - (f - feq) / tau
        else:                                                         # TRT, magic parameter 3/16
            fo, feqo = f[self.opp], feq[self.opp]
            fp, fm = 0.5 * (f + fo), 0.5 * (f - fo)
            ep, em = 0.5 * (feq + feqo), 0.5 * (feq - feqo)
            taum = 3.0 / 16.0 / (tau - 0.5) + 0.5
            out = f - (fp - ep) / tau - (fm - em) / taum
        if bool(self.F.abs().sum()):                                  # Guo forcing, split into even and odd parts
            cu = torch.einsum("qd,dxyz->qxyz", self.cf, u)
            cF = (self.cf @ self.F).view(-1, 1, 1, 1)
            uF = (u * self.F.view(3, 1, 1, 1)).sum(0)
            w = self.w.view(-1, 1, 1, 1)
            out = out + (1 - 0.5 / tau) * w * (9 * cu * cF - 3 * uF) + (1 - 0.5 / taum) * w * 3 * cF
        return out

    def _links(self):
        """Static bounce-back links: for each direction i, the fluid cells whose upstream neighbour is solid, and
        c_i . u_wall of that neighbour (moving walls and solids)."""
        key = (int(self._all_solid.sum()), float(self.u_solid.abs().sum()))
        if getattr(self, "_links_key", None) == key:
            return self._links_cache
        solid = self._all_solid
        fluid = ~solid
        wall_u = torch.where(self.moving, 1.0, 0.0) * self.u_in.view(3, 1, 1, 1) + self.u_solid
        links = []
        for i, ci in enumerate(self.c.tolist()):
            if i == 0:
                continue
            m = torch.roll(solid, shifts=ci, dims=(0, 1, 2)) & fluid
            if not m.any():
                continue
            uw = torch.roll(wall_u, shifts=ci, dims=(1, 2, 3))
            cu = (self.cf[i].view(3, 1, 1, 1) * uw).sum(0)
            links.append((i, int(self.opp[i]), m, cu if bool(cu.abs().sum()) else None))
        self._links_key, self._links_cache = key, links
        return links

    def step(self, n: int = 1) -> None:
        solid = self._all_solid
        links = self._links()
        for _ in range(n):
            post = self._collide(self._load())
            new = post.reshape(self.q, -1).gather(1, self._src).view_as(post)      # pull streaming
            rho_l = post.sum(0) if any(cu is not None for *_, cu in links) else None
            for i, j, m, cu in links:                                 # halfway bounce-back (+ moving-wall term)
                val = post[j] if cu is None else post[j] + 6.0 * self.w[i] * rho_l * cu
                new[i] = torch.where(m, val, new[i])
            self._boundaries(new)
            new[:, solid] = self.w.view(-1, 1)                        # solid cells hold rest equilibrium
            self._post = post
            self._store(new)
            self.t += 1

    def _boundaries(self, f: torch.Tensor) -> None:
        if self.eq_mask.any():                                        # equilibrium cells: rho = 1, u = u_in
            if not hasattr(self, "_feq_in"):
                one = torch.ones((1, 1, 1), device=self.dev)
                self._feq_in = self._feq(one, self.u_in.view(3, 1, 1, 1)).view(-1, 1)
            f[:, self.eq_mask] = self._feq_in
        if self.out_mask.any():                                       # outflow: rho = 1, u of the layer upstream
            if not hasattr(self, "_out_src"):
                ar = torch.arange(int(np.prod(self.shape)), device=self.dev).view(self.shape)
                self._out_src = torch.roll(ar, shifts=1, dims=0)[self.out_mask]
            fs = f.reshape(self.q, -1)[:, self._out_src]
            rho = fs.sum(0)
            u = (self.cf.T @ fs + 0.5 * self.F.view(3, 1)) / rho
            cu = self.cf @ u
            feq = self.w.view(-1, 1) * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (u * u).sum(0))
            f[:, self.out_mask] = feq

    def run(self, steps: int, every: int = 0, callback: Callable[[WindTunnelLBM], None] | None = None) -> None:
        done = 0
        while done < steps:
            k = min(every or steps, steps - done)
            self.step(k)
            done += k
            if callback is not None:
                callback(self)

    # ---------------------------------------------------------------- outputs
    def fields(self) -> dict[str, np.ndarray]:
        rho, u = self.macroscopic()
        rho = rho.masked_fill(self._all_solid, 1.0)
        u = u.masked_fill(self._all_solid, 0.0)
        return {"rho": rho.cpu().numpy(), "u": u.cpu().numpy(), "flags": self._all_solid.cpu().numpy().astype(np.uint8)}

    def force_on_solid(self, mask: torch.Tensor | None = None) -> np.ndarray:
        """Force (lattice units) on the solid cells in ``mask`` (default: all added solids, not the tunnel walls),
        by momentum exchange on the links between fluid and solid of the last step."""
        if not hasattr(self, "_post"):
            raise RuntimeError("run at least one step first")
        mask = self.solid if mask is None else torch.as_tensor(mask, device=self.dev).bool()
        fluid = ~self._all_solid
        post = self._post
        rho_l, _ = self.macroscopic(post)
        F = torch.zeros(3, device=self.dev)
        for j, cj in enumerate(self.c.tolist()):
            if j == 0:
                continue
            nb_solid = torch.roll(mask, shifts=[-v for v in cj], dims=(0, 1, 2)) & fluid   # neighbour along c_j
            if not nb_solid.any():
                continue
            uw = torch.roll(self.u_solid, shifts=[-v for v in cj], dims=(1, 2, 3))
            i = int(self.opp[j])
            cu_w = (self.cf[i].view(3, 1, 1, 1) * uw).sum(0)
            back = post[j] + 6.0 * self.w[i] * rho_l * cu_w        # what comes back along c_i
            dP = post[j] * self.cf[j].view(3, 1, 1, 1) - back * self.cf[i].view(3, 1, 1, 1)
            F += dP[:, nb_solid].sum(1)
        return F.cpu().numpy()

    def write_vtk(self, path: str, origin=(0.0, 0.0, 0.0), spacing: float = 1.0) -> list[str]:
        """``<path>-u.vtk``, ``<path>-rho.vtk`` and ``<path>-flags.vtk`` (legacy binary structured points)."""
        fld = self.fields()
        out = []
        for name, arr in (("u", fld["u"]), ("rho", fld["rho"]), ("flags", fld["flags"].astype(np.float32))):
            p = f"{path}-{name}.vtk"
            write_vtk(p, arr, origin, spacing)
            out.append(p)
        return out


# ------------------------------------------------------------------------------------------------- VTK I/O
def write_vtk(path: str, data: np.ndarray, origin=(0.0, 0.0, 0.0), spacing: float = 1.0) -> None:
    """Legacy VTK, BINARY big-endian STRUCTURED_POINTS. ``data`` is (nx, ny, nz) or (3, nx, ny, nz)."""
    vec = data.ndim == 4
    nx, ny, nz = data.shape[-3:]
    comps = 3 if vec else 1
    arr = np.moveaxis(data, 0, -1) if vec else data[..., None]
    arr = np.transpose(arr, (2, 1, 0, 3)).astype(">f4")            # x fastest
    head = (f"# vtk DataFile Version 3.0\nPINNeAPPle lattice Boltzmann\nBINARY\nDATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx} {ny} {nz}\nORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
            f"SPACING {spacing} {spacing} {spacing}\nPOINT_DATA {nx * ny * nz}\n"
            f"SCALARS data float {comps}\nLOOKUP_TABLE default\n")
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(arr.tobytes())


def read_vtk(path: str) -> dict:
    """Legacy VTK structured points (binary big-endian or ASCII), as written by FluidX3D's VTK export and by
    :func:`write_vtk`. Returns {"data": (nx,ny,nz) or (k,nx,ny,nz), "origin", "spacing", "dimensions", "name"}."""
    with open(path, "rb") as fh:
        raw = fh.read()
    pos, info, binary = 0, {}, True
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode(errors="replace").strip()
        pos = end + 1
        key = line.split(" ")[0].upper() if line else ""
        if key == "ASCII":
            binary = False
        elif key == "DIMENSIONS":
            info["dimensions"] = [int(v) for v in line.split()[1:4]]
        elif key == "ORIGIN":
            info["origin"] = [float(v) for v in line.split()[1:4]]
        elif key in ("SPACING", "ASPECT_RATIO"):
            info["spacing"] = [float(v) for v in line.split()[1:4]]
        elif key in ("SCALARS", "VECTORS"):
            parts = line.split()
            info["name"] = parts[1]
            info["comps"] = 3 if key == "VECTORS" else (int(parts[3]) if len(parts) > 3 else 1)
            info["dtype"] = parts[2]
            if key == "VECTORS":
                break
        elif key == "LOOKUP_TABLE":
            break
    nx, ny, nz = info["dimensions"]
    k = info["comps"]
    n = nx * ny * nz * k
    dt = {"float": ">f4", "double": ">f8", "unsigned_char": "u1", "char": "i1", "int": ">i4",
          "unsigned_int": ">u4", "short": ">i2", "unsigned_short": ">u2"}[info["dtype"]]
    if binary:
        vals = np.frombuffer(raw, dtype=dt, count=n, offset=pos)
    else:
        vals = np.array(raw[pos:].split()[:n], dtype=float)
    a = vals.astype(np.float32).reshape(nz, ny, nx, k).transpose(3, 2, 1, 0)
    return {"data": a[0] if k == 1 else a, "origin": info.get("origin", [0, 0, 0]),
            "spacing": info.get("spacing", [1, 1, 1]), "dimensions": info["dimensions"], "name": info.get("name")}


# ------------------------------------------------------------------------------------------------- references
def drag_sphere_schiller_naumann(re: float) -> float:
    """Drag coefficient of a sphere, Schiller & Naumann (1933), valid up to Re ~ 800.

    Returns
    -------
    Cd : float [-]
        Drag coefficient."""
    return 24.0 / re * (1 + 0.15 * re ** 0.687)


def stokes_sphere_field(shape, center, radius: float, u_inf: Sequence[float]):
    """Stokes flow past a fixed sphere (uniform stream u_inf), velocity (3, nx, ny, nz); zero inside."""
    x, y, z = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    r = np.stack([x - center[0], y - center[1], z - center[2]])
    rn = np.sqrt((r ** 2).sum(0))
    rn_safe = np.maximum(rn, 1e-9)
    U = np.asarray(u_inf, float).reshape(3, 1, 1, 1)
    ur = (U * r).sum(0) / rn_safe ** 2
    a = radius
    u = (U * (1 - 0.75 * a / rn_safe - 0.25 * a ** 3 / rn_safe ** 3)
         - r * ur * (0.75 * a / rn_safe - 0.75 * a ** 3 / rn_safe ** 3))
    u[:, rn <= a] = 0.0
    return u
