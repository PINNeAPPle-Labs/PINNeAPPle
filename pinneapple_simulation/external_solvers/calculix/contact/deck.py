"""Writing the CalculiX deck — the file PrePoMax opens.

Everything here is a keyword CalculiX defines and PrePoMax writes and reads itself, so a deck assembled from these
blocks imports into PrePoMax as a model with its contact definitions intact (surface interaction, friction, contact
pairs, rigid bodies) and runs unchanged under ``ccx``. The two files agree by construction, which is the point: the
deck an engineer inspects in PrePoMax is the deck that produced the evidence.

Units are the engineer's, kept consistent: **mm, N, MPa (= N/mm²), s**. ``E`` in MPa, pressures in MPa, forces in N —
the same convention PrePoMax uses when its unit system is set to mm-N-s.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .mesh import Grid, nset

#: The unit system every deck written here uses; it is also what the closed-form references in :mod:`.cases` expect.
UNITS = "mm, N, MPa (N/mm2), s"


@dataclass
class Deck:
    """A deck under construction: a title, then keyword blocks in the order CalculiX requires them."""
    title: str
    units: str = UNITS
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.comment(self.title)
        self.comment(f"units: {self.units} — written by PINNeAPPle for PrePoMax / CalculiX")

    def comment(self, text: str) -> None:
        self.lines.extend(f"** {line}".rstrip() for line in str(text).splitlines() or [""])

    def block(self, keyword: str, *data: str) -> None:
        self.lines.append(keyword)
        self.lines.extend(str(d) for d in data if str(d) != "")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.text(), encoding="utf-8")
        return p


# ------------------------------------------------------------------------------------------------------- blocks
def geometry(deck: Deck, blocks: Sequence[tuple[str, Grid]]) -> None:
    """``*NODE`` once for every block, then one ``*ELEMENT`` per block, each in its own element set."""
    nodes: dict[int, tuple[float, float, float]] = {}
    for _, g in blocks:
        nodes.update(g.nodes)
    deck.block("*NODE, NSET=NALL",
               *[f"{n}, {p[0]:.8g}, {p[1]:.8g}, {p[2]:.8g}" for n, p in sorted(nodes.items())])
    for name, g in blocks:
        deck.block(f"*ELEMENT, TYPE=C3D8, ELSET={name}",
                   *[f"{eid}, " + ", ".join(str(n) for n in cell)
                     for eid, cell in zip(g.elem_ids, g.cells, strict=True)])


def node_set(deck: Deck, name: str, ids: Sequence[int]) -> None:
    if not ids:
        raise ValueError(f"node set {name!r} would be empty")
    deck.block(f"*NSET, NSET={name}", *nset(ids))


def element_set(deck: Deck, name: str, ids: Sequence[int]) -> None:
    if not ids:
        raise ValueError(f"element set {name!r} would be empty")
    deck.block(f"*ELSET, ELSET={name}", *nset(ids))


def surface(deck: Deck, name: str, faces: Iterable[tuple[int, int]]) -> None:
    """``*SURFACE, TYPE=ELEMENT`` from ``(element id, face number)`` pairs, as :meth:`mesh.Grid.faces` returns them."""
    entries = [f"{eid}, S{face}" for eid, face in faces]
    if not entries:
        raise ValueError(f"surface {name!r} would be empty: no face lies in the plane it was selected on")
    deck.block(f"*SURFACE, NAME={name}, TYPE=ELEMENT", *entries)


def interaction(deck: Deck, name: str, friction: float, k_normal: float = 1e5, c0: float = 0.0,
                stick_slope: float = 1e4) -> None:
    """``*SURFACE INTERACTION`` with a linear pressure-overclosure and Coulomb friction.

    ``PRESSURE-OVERCLOSURE=LINEAR`` and a finite ``k_normal`` are what PrePoMax and CalculiX both recommend for
    contact that starts closed: ``HARD`` is silently reduced to linear by ``ccx`` anyway (with a default stiffness
    of about 1e7 N/mm³), and a very stiff interface converges slowly. ``stick_slope`` is the tangential stiffness
    before slip; too small a value lets a "stuck" contact creep, too large a one slows every iteration down.
    """
    if not 0.0 <= friction < 10.0:
        raise ValueError(f"friction coefficient {friction!r} is not a Coulomb coefficient in [0, 10)")
    deck.block(f"*SURFACE INTERACTION, NAME={name}")
    deck.block("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR", f"{k_normal:.6g}, {c0:.6g}")
    deck.block("*FRICTION", f"{friction:.6g}, {stick_slope:.6g}")


def contact_pair(deck: Deck, name: str, slave: str, master: str, interaction_name: str,
                 kind: str = "SURFACE TO SURFACE") -> None:
    """A contact pair. The slave is the softer, finer surface — the one whose nodes report contact results."""
    deck.block(f"*CONTACT PAIR, INTERACTION={interaction_name}, TYPE={kind}", f"{slave}, {master}")


def material(deck: Deck, name: str, E: float, nu: float) -> None:
    if E <= 0:
        raise ValueError("the Young modulus must be positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("the Poisson ratio must be in (-1, 0.5)")
    deck.block(f"*MATERIAL, NAME={name}", "*ELASTIC", f"{E:.8g}, {nu:.6g}")


def section(deck: Deck, elset: str, material_name: str) -> None:
    deck.block(f"*SOLID SECTION, ELSET={elset}, MATERIAL={material_name}")


def plastic(deck: Deck, sigma_y: float, hardening: float = 0.0) -> None:
    """``*PLASTIC`` under the material just declared, as an elastic-**perfectly** plastic card.

    The curve is written with two points and no slope between them (``sigma_y`` at zero plastic strain and the same
    stress at 1.0), which is what "perfectly plastic" means: the material carries ``sigma_y`` and no more, however far
    it is pushed. It has to follow the ``*ELASTIC`` of the same material — a ``*PLASTIC`` card attaches to the
    preceding ``*MATERIAL``, and writing another material in between silently gives the yield to the wrong body.
    """
    if sigma_y <= 0:
        raise ValueError("the yield stress must be positive")
    points = [(sigma_y, 0.0), (sigma_y + hardening, 1.0)]
    deck.block("*PLASTIC, HARDENING=ISOTROPIC", *[f"{s:.8g}, {e:.6g}" for s, e in points])


def distributing_coupling(deck: Deck, elset: str, element: int, ref: int, weights: dict[int, float]) -> None:
    """``*DISTRIBUTING COUPLING``: a load put at ``ref`` spread over the listed nodes instead of entering at one.

    The keyword needs an element set holding exactly one ``DCOUP3D`` element, whose single node is the reference;
    what is applied there — a force by ``*CLOAD`` or a displacement by ``*BOUNDARY`` — is distributed over the nodes
    below it with the weights given. Area weights (each node's tributary area) make the traction uniform, which is
    the point when the coupling stands for a bolt head or a nut face: a single node dents the plate, a weighted
    footprint presses it.

    CalculiX builds one multiple point constraint per direction with every listed node in it, so the equation is
    fully populated and the manual recommends staying under about 50 nodes — a footprint, not a whole face.
    """
    if not weights:
        raise ValueError(f"the distributing coupling {elset!r} has no node to distribute the load over")
    if any(w <= 0 for w in weights.values()):
        raise ValueError("a distributing coupling weight has to be positive")
    deck.block(f"*ELEMENT, TYPE=DCOUP3D, ELSET={elset}", f"{element}, {ref}")
    deck.block(f"*DISTRIBUTING COUPLING, ELSET={elset}", *[f"{n}, {w:.8g}" for n, w in sorted(weights.items())])


def plane_strain(deck: Deck, faces: Sequence[int], dof: int = 3) -> None:
    """``*BOUNDARY`` inside the step that holds the out-of-plane displacement on both thickness faces.

    A one-element-thick block whose two faces cannot move out of plane behaves exactly as a plane-strain section, and
    that is what the two-dimensional closed-form references (Hertz line contact) describe. It is written as an
    ordinary boundary so it reads in PrePoMax like any other support — no equations to maintain.
    """
    if not faces:
        raise ValueError("plane strain needs the node sets of both thickness faces")
    for ids in faces:
        deck.block("*BOUNDARY", f"{ids}, {dof}, {dof}, 0.0")


def outputs(deck: Deck, totals_set: str, displacement_set: str | None = None, contact: bool = True,
            stress: bool = False) -> None:
    """The requests that make a run checkable: reaction totals in the ``.dat`` (equilibrium and friction force) and
    contact results in the ``.frd`` (pressure, shear, opening, slip per node), which is also what PrePoMax plots."""
    deck.block(f"*NODE PRINT, NSET={totals_set}, TOTALS=YES", "RF")
    if displacement_set:
        deck.block(f"*NODE PRINT, NSET={displacement_set}", "U")
    deck.block("*NODE FILE", "U")
    if stress:
        deck.block("*EL FILE", "S")
    if contact:
        deck.block("*CONTACT FILE", "CSTRESS, CDISP")
