"""A chip floorplan: a die divided into placement slots, blocks with power densities and leakage, and the nets
whose delay matters.

The default floorplan is illustrative (a 10 mm x 10 mm die, six blocks of a mobile SoC class with round-number
powers); it is not data for a real product. Units: SI unless the name says otherwise (``_mm``, ``_w``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Block", "Floorplan", "default_floorplan"]


@dataclass(frozen=True)
class Block:
    """``dynamic_w``: switching power; ``leakage_w``: leakage at ``t_ref_c``, scaling as
    ``exp(leak_beta (T - t_ref))`` (sub-threshold leakage roughly doubles every 30-40 K)."""
    name: str
    dynamic_w: float
    leakage_w: float = 0.0
    leak_beta: float = 0.02
    t_ref_c: float = 85.0


@dataclass
class Floorplan:
    """``slots``: (nx, ny) equal rectangles tiling the die; ``placement[i]`` is the slot of ``blocks[i]``
    (``slot = ix * ny + iy``). ``nets``: (block a, block b, weight) of the critical nets, routed centre to centre with
    Manhattan length."""
    die_w_mm: float
    die_h_mm: float
    blocks: list[Block]
    slots: tuple[int, int]
    placement: list[int]
    nets: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        n = self.slots[0] * self.slots[1]
        if len(self.blocks) > n:
            raise ValueError(f"{len(self.blocks)} blocks for {n} slots")
        if sorted(self.placement) != sorted(set(self.placement)) or len(self.placement) != len(self.blocks):
            raise ValueError("placement must give each block its own slot")
        names = {b.name for b in self.blocks}
        bad = [n for a, b in self.nets for n in (a, b) if n not in names]
        if bad:
            raise ValueError(f"nets name unknown blocks {bad}")

    @property
    def n_slots(self) -> int:
        return self.slots[0] * self.slots[1]

    @property
    def slot_size_mm(self) -> tuple[float, float]:
        return self.die_w_mm / self.slots[0], self.die_h_mm / self.slots[1]

    def slot_centre_mm(self, slot: int) -> tuple[float, float]:
        ix, iy = divmod(int(slot), self.slots[1])
        w, h = self.slot_size_mm
        return (ix + 0.5) * w, (iy + 0.5) * h

    def slot_mask(self, slot: int, nx: int, ny: int) -> np.ndarray:
        """Cells of an (nx, ny) die grid that fall in a slot (cell centres)."""
        ix, iy = divmod(int(slot), self.slots[1])
        xc = (np.arange(nx) + 0.5) / nx * self.slots[0]
        yc = (np.arange(ny) + 0.5) / ny * self.slots[1]
        return ((xc >= ix) & (xc < ix + 1))[:, None] & ((yc >= iy) & (yc < iy + 1))[None, :]

    def power_density_w_mm2(self) -> dict[str, float]:
        w, h = self.slot_size_mm
        return {b.name: (b.dynamic_w + b.leakage_w) / (w * h) for b in self.blocks}

    def net_lengths_mm(self, placement: list[int] | None = None) -> dict[str, float]:
        pl = self.placement if placement is None else placement
        where = {b.name: self.slot_centre_mm(s) for b, s in zip(self.blocks, pl, strict=True)}
        return {f"{a}-{b}": abs(where[a][0] - where[b][0]) + abs(where[a][1] - where[b][1]) for a, b in self.nets}

    def with_placement(self, placement: list[int]) -> Floorplan:
        return Floorplan(self.die_w_mm, self.die_h_mm, list(self.blocks), self.slots, list(placement),
                         list(self.nets))

    def to_dict(self) -> dict:
        return {"die_mm": [self.die_w_mm, self.die_h_mm], "slots": list(self.slots), "placement": list(self.placement),
                "blocks": [{"name": b.name, "dynamic_w": b.dynamic_w, "leakage_w": b.leakage_w,
                            "slot": s, "power_density_w_mm2": self.power_density_w_mm2()[b.name]}
                           for b, s in zip(self.blocks, self.placement, strict=True)],
                "nets": [list(n) for n in self.nets]}


def default_floorplan() -> Floorplan:
    """10 mm x 10 mm die, 3 x 2 slots: two CPU clusters, a GPU, an NPU, the L3 cache and the I/O ring segment;
    the critical nets join the CPUs to the cache and the GPU and NPU to the cache."""
    blocks = [Block("cpu0", 9.0, 2.0), Block("cpu1", 9.0, 2.0), Block("gpu", 12.0, 2.5), Block("npu", 6.0, 1.0),
              Block("l3", 2.0, 1.5), Block("io", 1.5, 0.3)]
    return Floorplan(10.0, 10.0, blocks, (3, 2), [0, 1, 2, 3, 4, 5],
                     [("cpu0", "l3"), ("cpu1", "l3"), ("gpu", "l3"), ("npu", "l3"), ("cpu0", "cpu1")])
