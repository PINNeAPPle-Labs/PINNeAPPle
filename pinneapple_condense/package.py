"""A synthetic multilayer chip package built from one repeated unit cell (a *block*).

The package is a stack of layers (substrate, C4 bumps in underfill, silicon interposer with TSVs, microbumps in
underfill, die, TIM, copper lid) over a lateral array of identical-pitch blocks. A block is one unit cell of the
repeated structures: one C4 bump, one TSV and one microbump, all square in plan (square inclusions rasterise exactly
at every resolution, which keeps the mesh-convergence study free of staircase noise). Every layer spans the whole
array: the lid does not overhang the die in this benchmark.

Materials (W/m/K, J/m^3/K) are representative room-temperature values, not data for a real product:

* silicon k = 148, rho c = 1.63e6 (Incropera et al., *Fundamentals of Heat and Mass Transfer*, 6th ed., Table A.1);
* copper k = 401, rho c = 3.44e6 (same table);
* SAC solder k = 58, underfill k = 0.5, TIM k = 4, organic substrate k = 0.8 (typical datasheet ranges; synthetic).

A :class:`Variant` changes a block without changing its geometry: a voided microbump (conductivity factor), a
missing bump (inclusion replaced by underfill), a degraded TIM contact (contact-resistance factor). Blocks with the
same variant have the same conduction operator; that is what the condensation reuses.

Units: SI. Temperatures are rises above the ambient of the Robin boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

#: name -> (k [W/m/K], rho*c [J/m^3/K])
MATERIALS: dict[str, tuple[float, float]] = {
    "silicon": (148.0, 1.63e6),
    "copper": (401.0, 3.44e6),
    "solder": (58.0, 1.67e6),
    "underfill": (0.5, 1.8e6),
    "tim": (4.0, 2.5e6),
    "substrate": (0.8, 1.9e6),
}


@dataclass(frozen=True)
class Layer:
    """One layer of the stack. ``inclusion``: material of the square inclusion centred in each block (bump, TSV),
    ``side``: its side as a fraction of the block pitch."""
    name: str
    thickness: float
    nz: int
    matrix: str
    inclusion: str | None = None
    side: float = 0.0


@dataclass(frozen=True)
class Variant:
    """What makes a block differ from the base block. ``inclusion_k``: ((layer, factor), ...) multiplies the
    inclusion conductivity of a layer (a void fraction lowers it); ``removed``: layers whose inclusion is missing
    (replaced by the matrix); ``contact``: ((lower, upper, factor), ...) multiplies a contact resistance."""
    inclusion_k: tuple[tuple[str, float], ...] = ()
    removed: tuple[str, ...] = ()
    contact: tuple[tuple[str, str, float], ...] = ()

    @property
    def key(self) -> tuple:
        return (tuple(sorted((n, round(float(f), 12)) for n, f in self.inclusion_k)), tuple(sorted(self.removed)),
                tuple(sorted((a, b, round(float(f), 12)) for a, b, f in self.contact)))

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, Variant) and self.key == other.key


BASE = Variant()


@dataclass(frozen=True)
class Stack:
    """The layers (bottom to top), the block pitch and lateral cells per block, contact resistances between layers
    (m^2 K / W), Robin coefficients on the top of the lid and the bottom of the substrate (W/m^2/K), and the layer
    whose bottom cell row is the active (heat-generating, junction) surface."""
    layers: tuple[Layer, ...]
    pitch: float = 40e-6
    n: int = 8
    contact: tuple[tuple[str, str, float], ...] = ()
    h_top: float = 2.0e4
    h_bottom: float = 1.0e3
    active: str = "die"

    @property
    def nz(self) -> int:
        return sum(L.nz for L in self.layers)

    @property
    def dx(self) -> float:
        return self.pitch / self.n

    def dz(self) -> np.ndarray:
        return np.concatenate([np.full(L.nz, L.thickness / L.nz) for L in self.layers])

    def layer_slices(self) -> dict[str, slice]:
        out, k = {}, 0
        for L in self.layers:
            out[L.name] = slice(k, k + L.nz)
            k += L.nz
        return out

    @property
    def active_k(self) -> int:
        """z index of the active cell row (bottom row of the active layer)."""
        return self.layer_slices()[self.active].start

    def refined(self, lateral: int = 1, vertical: int = 1) -> Stack:
        """The same stack with ``lateral`` times more cells per block side and ``vertical`` times more per layer."""
        return replace(self, n=self.n * lateral, layers=tuple(replace(L, nz=L.nz * vertical) for L in self.layers))

    def properties(self, variant: Variant = BASE) -> dict[str, np.ndarray]:
        """Cell properties of one block: ``k`` (n, n, nz) (isotropic here), ``rho_c``, ``dz`` (nz,), ``R`` (nz-1,)
        contact resistance of each horizontal face between z rows, and the per-layer ``inclusion`` masks."""
        n, nz = self.n, self.nz
        k = np.empty((n, n, nz))
        rc = np.empty((n, n, nz))
        centres = (np.arange(n) + 0.5) / n - 0.5
        incl_k = dict(variant.inclusion_k)
        masks = {}
        for L, sl in zip(self.layers, self.layer_slices().values()):
            km, cm = MATERIALS[L.matrix]
            k[:, :, sl], rc[:, :, sl] = km, cm
            if L.inclusion and L.name not in variant.removed:
                inside = np.abs(centres) < L.side / 2
                m2 = inside[:, None] & inside[None, :]
                ki, ci = MATERIALS[L.inclusion]
                k[m2, sl] = ki * incl_k.get(L.name, 1.0)
                rc[m2, sl] = ci
                masks[L.name] = m2
            else:
                masks[L.name] = np.zeros((n, n), bool)
        R = np.zeros(nz - 1)
        sls = self.layer_slices()
        scale = {(a, b): f for a, b, f in variant.contact}
        for a, b, r in self.contact:
            if sls[a].stop != sls[b].start:
                raise ValueError(f"contact {a}/{b}: layers are not adjacent")
            R[sls[a].stop - 1] = r * scale.get((a, b), 1.0)
        return {"k": k, "rho_c": rc, "dz": self.dz(), "R": R, "masks": masks}


def reference_stack(n: int = 8, vertical: int = 1) -> Stack:
    """The benchmark stack: 40 um block pitch (one microbump, one TSV and one C4 per block; a simplification: C4
    pitch is usually larger than microbump pitch), ~1 mm total height, 8 x 8 cells per block side by default."""
    layers = (
        Layer("substrate", 200e-6, 3, "substrate"),
        Layer("c4", 60e-6, 3, "underfill", "solder", 0.5),
        Layer("interposer", 100e-6, 4, "silicon", "copper", 0.25),
        Layer("ubump", 20e-6, 3, "underfill", "solder", 0.5),
        Layer("die", 100e-6, 4, "silicon"),
        Layer("tim", 30e-6, 2, "tim"),
        Layer("lid", 500e-6, 4, "copper"),
    )
    layers = tuple(replace(L, nz=L.nz * vertical) for L in layers)
    return Stack(layers=layers, n=n, contact=(("die", "tim", 5e-6), ("tim", "lid", 5e-6)))


@dataclass
class Package:
    """A lateral ``(nbx, nby)`` array of blocks over one stack. ``layout[I, J]`` indexes ``variants``; ``power`` is
    the heat (W) of every active cell, shape ``(nbx * n, nby * n)``."""
    stack: Stack
    layout: np.ndarray
    variants: list[Variant] = field(default_factory=lambda: [BASE])
    power: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.layout.shape)

    @property
    def n_blocks(self) -> int:
        return int(self.layout.size)

    @property
    def n_cells(self) -> int:
        return self.n_blocks * self.stack.n ** 2 * self.stack.nz

    def block_power(self, power: np.ndarray | None = None) -> np.ndarray:
        """Power of each block's active cells, shape ``(nbx, nby, n * n)`` (block-local C order)."""
        p = self.power if power is None else power
        n = self.stack.n
        nbx, nby = self.shape
        return p.reshape(nbx, n, nby, n).transpose(0, 2, 1, 3).reshape(nbx, nby, n * n)

    def global_properties(self) -> dict[str, np.ndarray]:
        """Cell properties of the whole package, shape ``(nbx * n, nby * n, nz)``; contact resistances are per
        block, shape ``(nbx * n, nby * n, nz - 1)``."""
        n, nz = self.stack.n, self.stack.nz
        nbx, nby = self.shape
        props = [self.stack.properties(v) for v in self.variants]
        k = np.empty((nbx, nby, n, n, nz))
        rc = np.empty_like(k)
        R = np.empty((nbx, nby, n, n, nz - 1))
        for t, p in enumerate(props):
            sel = self.layout == t
            k[sel], rc[sel], R[sel] = p["k"], p["rho_c"], p["R"]

        def g(a):
            return a.transpose(0, 2, 1, 3, 4).reshape(nbx * n, nby * n, a.shape[-1])
        return {"k": g(k), "rho_c": g(rc), "R": g(R), "dz": self.stack.dz()}


def hotspot_power(stack: Stack, shape: tuple[int, int], total: float, *, hotspots: int = 3, background: float = 0.4,
                  width: float = 0.08, seed: int = 0, centres: np.ndarray | None = None) -> np.ndarray:
    """A non-uniform power map on the active cells (W per cell): ``background`` of ``total`` spread uniformly, the
    rest in Gaussian hotspots (``width`` as a fraction of the die side) at random or given ``centres``."""
    rng = np.random.default_rng(seed)
    nx, ny = shape[0] * stack.n, shape[1] * stack.n
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny, indexing="ij")
    c = rng.uniform(0.15, 0.85, (hotspots, 2)) if centres is None else np.asarray(centres, float)
    amp = rng.uniform(0.5, 1.0, len(c))
    g = np.zeros_like(X)
    for (cx, cy), a in zip(c, amp):
        g += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
    if not len(c):
        background = 1.0
    q = background * total / X.size + (1 - background) * total * g / max(g.sum(), 1e-300)
    return q


def uniform_package(shape: tuple[int, int], stack: Stack | None = None, *, total_power: float | None = None,
                    power_density: float = 1.0e6, seed: int = 0, hotspots: int = 3) -> Package:
    """A package of identical blocks with a hotspot power map. ``power_density``: mean heat flux over the die
    (W/m^2; 1e6 W/m^2 = 100 W/cm^2) unless ``total_power`` is given. Change ``layout`` / ``variants`` afterwards
    for heterogeneous blocks."""
    stack = stack or reference_stack()
    area = shape[0] * shape[1] * stack.pitch ** 2
    total = power_density * area if total_power is None else total_power
    return Package(stack, np.zeros(shape, int), [BASE],
                   hotspot_power(stack, shape, total, seed=seed, hotspots=hotspots))
