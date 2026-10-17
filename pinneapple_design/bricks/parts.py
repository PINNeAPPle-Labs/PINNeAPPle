"""Interlocking toy bricks: nominal dimensions, the part catalogue, masses and colours.

Dimensions (millimetres)
------------------------
The grid follows the LDraw unit system (1 LDU = 0.4 mm): a stud pitch of 20 LDU (8.0 mm), a plate 8 LDU high
(3.2 mm), a brick 24 LDU high (9.6 mm, three plates), studs 12 LDU in diameter (4.8 mm) and 4 LDU high (1.6 mm),
as drawn by the LDraw ``stud.dat`` primitive (LDraw File Format Specification and parts library, ldraw.org).
A part is 0.2 mm narrower than its grid cell (0.1 mm per side) so neighbours do not touch; an n-stud long part is
``8 n - 0.2`` mm long. StableLego uses the same 7.8 mm unit width (``Brick_Unit_Length = 0.0078`` m in the
public configuration of Liu et al., 2024).

Masses
------
Brick masses are the values of the StableLego brick library (``lego_library.json`` in
github.com/intelligent-control-lab/StableLego, MIT licence; Liu, Deng, Wang and Liu, "StableLego: Stability
Analysis of Block Stacking Assembly", IEEE Robotics and Automation Letters 9(11), 9383-9390, 2024,
doi:10.1109/LRA.2024.3414281). Parts without a published mass (plates, 2x3, 2x8 bricks) are estimated with a
two-term shell model fitted to those masses, ``m = a * studs + b * perimeter * height``
(:func:`fit_mass_model`); :func:`mass_model_residuals` reports how well it reproduces the published values.

Colours
-------
The colour codes are LDraw colour numbers (used only when writing .ldr files); the RGB values here are
approximate display colours chosen for rendering, not the LDraw ``LDConfig.ldr`` values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LDU_MM = 0.4                       # 1 LDraw unit in millimetres
PITCH_MM = 8.0                     # stud pitch = 20 LDU
PLATE_MM = 3.2                     # plate height = 8 LDU
BRICK_MM = 9.6                     # brick height = 24 LDU = 3 plates
STUD_DIAMETER_MM = 4.8             # 12 LDU (stud.dat radius 6 LDU)
STUD_HEIGHT_MM = 1.6               # 4 LDU
CLEARANCE_MM = 0.1                 # gap per side between a part and its grid cell (7.8 mm wide 1x1)
G = 9.81                           # m/s^2

MASS_SOURCE = ("StableLego brick library (lego_library.json), Liu, Deng, Wang, Liu, IEEE RA-L 9(11) 9383-9390, "
               "2024, doi:10.1109/LRA.2024.3414281")

# published masses [kg], keyed by (studs along x, studs along y, height in plates)
_PUBLISHED_MASS = {
    (1, 1, 3): 0.00043, (2, 1, 3): 0.00081, (4, 1, 3): 0.00157, (6, 1, 3): 0.00228, (8, 1, 3): 0.00303,
    (2, 2, 3): 0.00115, (4, 2, 3): 0.00216, (6, 2, 3): 0.00323,
}


@dataclass(frozen=True)
class PartType:
    """A rectangular brick or plate: ``nx`` x ``ny`` studs, ``h`` plates high (3 = brick, 1 = plate).

    ``ldraw`` is the LDraw part file of the same shape, whose long side lies along the LDraw x axis.
    ``smooth=True`` is a stud-less block of the same footprint (a tile, or a test body with no clutch).
    """

    name: str
    nx: int
    ny: int
    h: int
    ldraw: str = ""
    mass: float = 0.0              # [kg]
    mass_source: str = ""
    smooth: bool = False

    @property
    def size_mm(self) -> tuple[float, float, float]:
        """Outer size of the body (without studs) in millimetres."""
        return (PITCH_MM * self.nx - 2 * CLEARANCE_MM, PITCH_MM * self.ny - 2 * CLEARANCE_MM, PLATE_MM * self.h)

    @property
    def studs(self) -> int:
        return 0 if self.smooth else self.nx * self.ny

    @property
    def is_brick(self) -> bool:
        return self.h == 3


def _shell_terms(nx: int, ny: int, h: int) -> np.ndarray:
    perim = 2 * (nx + ny) * PITCH_MM
    return np.array([nx * ny, perim * h * PLATE_MM], float)


def fit_mass_model() -> np.ndarray:
    """Least-squares coefficients ``(a, b)`` of ``m = a * studs + b * perimeter_mm * height_mm`` [kg] fitted to the
    published brick masses."""
    A = np.array([_shell_terms(*k) for k in _PUBLISHED_MASS])
    y = np.array(list(_PUBLISHED_MASS.values()))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def estimate_mass(nx: int, ny: int, h: int) -> float:
    """Mass of an ``nx`` x ``ny`` part ``h`` plates high [kg] from the shell model of :func:`fit_mass_model`."""
    return float(_shell_terms(nx, ny, h) @ fit_mass_model())


def mass_model_residuals() -> dict[str, float]:
    """Relative error of the shell model on each published mass (leave-nothing-out fit)."""
    return {f"{a}x{b}x{h}": estimate_mass(a, b, h) / m - 1.0 for (a, b, h), m in _PUBLISHED_MASS.items()}


def _part(name, nx, ny, h, ldraw, smooth=False) -> PartType:
    key = (max(nx, ny), min(nx, ny), h)
    if key in _PUBLISHED_MASS and not smooth:
        m, src = _PUBLISHED_MASS[key], MASS_SOURCE
    else:
        m, src = estimate_mass(nx, ny, h), "shell model fitted to " + MASS_SOURCE
    return PartType(name, nx, ny, h, ldraw, m, src, smooth)


# LDraw part numbers of the common rectangular bricks and plates
CATALOG: dict[str, PartType] = {p.name: p for p in [
    _part("brick_1x1", 1, 1, 3, "3005.dat"), _part("brick_1x2", 2, 1, 3, "3004.dat"),
    _part("brick_1x3", 3, 1, 3, "3622.dat"), _part("brick_1x4", 4, 1, 3, "3010.dat"),
    _part("brick_1x6", 6, 1, 3, "3009.dat"), _part("brick_1x8", 8, 1, 3, "3008.dat"),
    _part("brick_2x2", 2, 2, 3, "3003.dat"), _part("brick_2x3", 3, 2, 3, "3002.dat"),
    _part("brick_2x4", 4, 2, 3, "3001.dat"), _part("brick_2x6", 6, 2, 3, "2456.dat"),
    _part("brick_2x8", 8, 2, 3, "3007.dat"),
    _part("plate_1x1", 1, 1, 1, "3024.dat"), _part("plate_1x2", 2, 1, 1, "3023.dat"),
    _part("plate_1x4", 4, 1, 1, "3710.dat"), _part("plate_1x6", 6, 1, 1, "3666.dat"),
    _part("plate_1x8", 8, 1, 1, "3460.dat"), _part("plate_2x2", 2, 2, 1, "3022.dat"),
    _part("plate_2x4", 4, 2, 1, "3020.dat"), _part("plate_2x6", 6, 2, 1, "3795.dat"),
    _part("plate_2x8", 8, 2, 1, "3034.dat"),
]}
BY_LDRAW = {p.ldraw: p for p in CATALOG.values()}


def part(name: str) -> PartType:
    """A catalogue part by name (``"brick_2x4"``) or LDraw file (``"3001.dat"``)."""
    if name in CATALOG:
        return CATALOG[name]
    key = name.lower() if name.lower().endswith(".dat") else name.lower() + ".dat"
    if key in BY_LDRAW:
        return BY_LDRAW[key]
    raise KeyError(f"unknown part {name!r}; known: {sorted(CATALOG)}")


def smooth_block(nx: int, ny: int, h: int, mass: float | None = None) -> PartType:
    """A stud-less block (no clutch with anything) of the given footprint, e.g. for unconnected stacking tests."""
    m = estimate_mass(nx, ny, h) if mass is None else float(mass)
    return PartType(f"block_{nx}x{ny}x{h}", nx, ny, h, "", m, "shell model" if mass is None else "given", True)


@dataclass(frozen=True)
class Colour:
    name: str
    ldraw: int
    rgb: tuple[float, float, float]          # sRGB, 0..1


PALETTE: dict[str, Colour] = {c.name: c for c in [
    Colour("red", 4, (0.79, 0.10, 0.09)), Colour("blue", 1, (0.00, 0.33, 0.75)),
    Colour("yellow", 14, (0.98, 0.78, 0.04)), Colour("green", 2, (0.14, 0.48, 0.25)),
    Colour("white", 15, (0.96, 0.96, 0.96)), Colour("black", 0, (0.11, 0.11, 0.12)),
    Colour("orange", 25, (0.95, 0.47, 0.08)), Colour("light_grey", 71, (0.63, 0.65, 0.65)),
    Colour("dark_grey", 72, (0.39, 0.40, 0.40)), Colour("tan", 19, (0.89, 0.80, 0.60)),
    Colour("lime", 27, (0.65, 0.79, 0.07)), Colour("azure", 322, (0.21, 0.68, 0.87)),
]}
BY_LDRAW_COLOUR = {c.ldraw: c for c in PALETTE.values()}


def colour(name_or_code) -> Colour:
    if isinstance(name_or_code, Colour):
        return name_or_code
    if isinstance(name_or_code, (int, np.integer)):
        return BY_LDRAW_COLOUR.get(int(name_or_code), Colour(f"ldraw_{int(name_or_code)}", int(name_or_code),
                                                             (0.6, 0.6, 0.6)))
    return PALETTE[name_or_code]


@dataclass
class JointModel:
    """Strength of a stud joint and contact friction used by the statics check.

    ``clutch_n``: tensile capacity of one stud-tube contact [N]. The default 0.98 N is the friction capacity per
    stud reported for the StableLego analysis (Liu et al., IEEE RA-L 2024) and reused by BrickGPT (Pun et al.,
    arXiv:2505.05469, 2025). Neither
    source publishes a measurement uncertainty, and the public StableLego code ships a configuration with a
    different value, so ``clutch_band`` is an *assumed* range for sensitivity studies, not a measured one.
    Real clutch varies with wear, mould, temperature and how far the parts were pressed together.

    ``mu``: Coulomb friction for smooth contacts (a table, stud-less blocks). 0.4 is an assumption for ABS on a
    hard surface; set it from your own measurement.

    ``shear_n``: cap on the horizontal force through one stud [N]. ``None`` (default) leaves it unbounded, as in
    the StableLego force model, where horizontal stud forces are free; no published value bounds it. Overlapping
    layers then act like a truss (horizontal forces at different heights make a couple), which can be
    optimistic for bridges and cantilevers: compare with a finite cap to see how much a result relies on it.

    ``edge_inset_mm``: contact (compression and friction) acts at least this far inside the edge of a contact
    patch. Real part and table edges are rounded, and a build balanced exactly on an edge is in neutral
    equilibrium (it tips in a dynamic simulation), so a small inset keeps the search away from knife-edge designs.
    0.5 mm is an assumption; set 0 to compare with sharp-edged closed forms.
    """

    clutch_n: float = 0.98
    clutch_band: tuple[float, float] = (0.5, 1.5)    # x clutch_n, assumed, see the docstring
    mu: float = 0.4
    shear_n: float | None = None                     # per-stud horizontal force cap; None = unbounded
    edge_inset_mm: float = 0.5                       # contact forces act this far inside a patch edge (assumed)
    source: str = ("0.98 N per stud: StableLego (Liu et al., IEEE RA-L 9(11), 2024, doi:10.1109/LRA.2024.3414281); "
                   "BrickGPT (Pun et al., arXiv:2505.05469, 2025)")
    notes: list[str] = field(default_factory=lambda: [
        "tension only: a stud joint fails when its pull-out force exceeds clutch_n",
        "shear and torsion at stud joints are carried by the stud geometry (unbounded unless shear_n is set)",
        "compression is unlimited (ABS crushing is far above these loads)",
        "rigid parts: no joint compliance, so the force distribution is the most favourable one (lower bound)",
        "a single stud is a pivot about the vertical (no torsion)",
        "contact acts at least edge_inset_mm inside a patch edge (rounded edges, no knife-edge balance)",
    ])
