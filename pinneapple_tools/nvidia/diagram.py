"""Architecture diagrams of a PINNeAPPle + NVIDIA pipeline, drawn by code (matplotlib only, no graphviz).

    from pinneapple_tools.nvidia.diagram import Block, architecture_diagram

    blocks = [Block("data", "Darcy solver\\n(finite volumes)", "pinneapple", col=0),
              Block("fno", "FNO\\nphysicsnemo.models.fno", "physicsnemo", col=1),
              Block("usd", "USD stage\\n(OpenUSD, usd-core)", "omniverse", col=2),
              Block("kit", "Omniverse Kit / RTX", "omniverse", col=3, ran=False)]
    architecture_diagram("Cold plate", blocks, [("data", "fno", "a, T pairs"), ("fno", "usd", "T field")], "arch.png")

Each block has an owner (``pinneapple``, an NVIDIA product: ``physicsnemo``, ``omniverse``, ``brep``, ``isaac``,
``cosmos``, ``nim``, or a PINNeAPPle stack: ``robotsim``, ``services``, ``worldmodel``) that sets its colour, a column (left to right is the data flow) and an optional row. ``ran=False`` draws a
dashed, hatched block: the step needs a GPU, an API key or a download and was not executed where the figure was made.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Block", "OWNERS", "architecture_diagram"]

# owner -> (face colour, edge colour, legend label)
OWNERS = {
    "pinneapple": ("#ffe08a", "#b88a00", "PINNeAPPle (physics, checks, trust)"),
    "physicsnemo": ("#c9e8a0", "#4f7d00", "NVIDIA PhysicsNeMo"),
    "omniverse": ("#a9d8f0", "#1f6f99", "NVIDIA Omniverse / OpenUSD"),
    "brep": ("#f3d6b0", "#8a5a12", "NVIDIA USD BRep (B-rep kernel)"),
    "robotsim": ("#e2d4f5", "#5b3a8c", "PINNeAPPle Robot Sim (on NVIDIA Newton)"),
    "services": ("#c4ecd9", "#1d7a50", "PINNeAPPle Model Services (NIM-compatible)"),
    "worldmodel": ("#f6d0d0", "#9a2b2b", "PINNeAPPle World Model (Cosmos alternative)"),
    "isaac": ("#d7c4ef", "#5b3a8c", "NVIDIA Isaac Sim / Isaac Lab"),
    "cosmos": ("#f6c0c0", "#9a2b2b", "NVIDIA Cosmos"),
    "nim": ("#c4ecd9", "#1d7a50", "NVIDIA NIM (build.nvidia.com)"),
    "data": ("#eeeeee", "#666666", "data / files"),
}


@dataclass
class Block:
    key: str
    label: str
    owner: str = "pinneapple"
    col: int = 0
    row: float = 0.0
    ran: bool = True            # False: needs GPU / API key / download, not executed here


def architecture_diagram(title: str, blocks: Sequence[Block], edges: Sequence[tuple], path: str, *,
                         subtitle: str = "", width: float = 2.6, height: float = 1.0, dpi: int = 150) -> str:
    """Draw ``blocks`` and ``edges`` (``(src, dst)`` or ``(src, dst, label)``) and save a PNG at ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    by = {b.key: b for b in blocks}
    for e in edges:
        for k in e[:2]:
            if k not in by:
                raise KeyError(f"edge {e!r} names an unknown block {k!r}")
    for b in blocks:
        if b.owner not in OWNERS:
            raise ValueError(f"unknown owner {b.owner!r}; one of {sorted(OWNERS)}")
    ncol = max(b.col for b in blocks) + 1
    rows = [b.row for b in blocks]
    gx, gy = width + 0.9, height + 0.55
    W = ncol * gx + 0.4
    H = (max(rows) - min(rows) + 1) * gy + 1.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 1.1

    def centre(b):
        return 0.3 + b.col * gx + width / 2, top - (b.row - min(rows)) * gy - height / 2

    for b in blocks:
        fc, ec, _ = OWNERS[b.owner]
        x, y = centre(b)
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec,
                                    lw=1.6, ls="-" if b.ran else (0, (4, 3)), hatch=None if b.ran else "//",
                                    alpha=1.0 if b.ran else 0.75, zorder=2))
        ax.text(x, y, b.label + ("" if b.ran else "\n[not run here: GPU / key]"), ha="center", va="center",
                fontsize=8.2, zorder=3, wrap=True)
    for e in edges:
        s, d = by[e[0]], by[e[1]]
        (x0, y0), (x1, y1) = centre(s), centre(d)
        if s.col != d.col:
            sx = 1 if x1 > x0 else -1
            p0, p1 = (x0 + sx * width / 2, y0), (x1 - sx * width / 2, y1)
        else:
            sy = 1 if y1 > y0 else -1
            p0, p1 = (x0, y0 + sy * height / 2), (x1, y1 - sy * height / 2)
        dashed = not (s.ran and d.ran)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#333333",
                                     ls=(0, (4, 3)) if dashed else "-", zorder=1,
                                     connectionstyle="arc3,rad=0.0"))
        if len(e) > 2 and e[2]:
            vertical = s.col == d.col
            ax.text((p0[0] + p1[0]) / 2 + (0.08 if vertical else 0), (p0[1] + p1[1]) / 2 + (0 if vertical else 0.12),
                    e[2], ha="left" if vertical else "center", va="center" if vertical else "bottom", fontsize=7,
                    color="#222222", zorder=4, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.text(0.3, H - 0.35, title, fontsize=12, weight="bold", va="center")
    if subtitle:
        ax.text(0.3, H - 0.72, subtitle, fontsize=8.5, va="center", color="#444444")
    owners = list(dict.fromkeys(b.owner for b in blocks))
    handles = [Patch(fc=OWNERS[o][0], ec=OWNERS[o][1], label=OWNERS[o][2]) for o in owners]
    if any(not b.ran for b in blocks):
        handles.append(Patch(fc="white", ec="#666666", hatch="//", ls="--", label="not executed here (GPU / API key)"))
    ax.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=7.5, frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
