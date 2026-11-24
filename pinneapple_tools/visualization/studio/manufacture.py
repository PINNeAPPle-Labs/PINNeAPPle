"""A body with a manufacturing history.

Everything else in the studio adds detail from the outside: a bolt, a weld, a chamfer. This is the other direction -- a
part that carries the record of how it was made, because that record *is* what a viewer reads as a real part. A
casting has a draft on its walls, a skin with sand grain and a parting line; a machined face on that casting is flat,
bright, cut in one direction, and its edge is a chamfer, not a corner; a bore has a chamfered mouth and a dark throat; a
weld has a bead and the straw-and-blue band beside it. Nothing in that list is visible as "detail" -- the whole surface
changes character.

    from pinneapple_tools.visualization.studio import manufacture as M
    body = M.cast_blank(radius=0.11, height=0.09, draft=2.0).machine_face([0.11, 0, 0], (1, 0, 0), radius=0.05) \
                                                     .bore([0.11, 0, 0], (1, 0, 0), diameter=0.03) \
                                                     .weld(path).finish()
    sc = body.scene
    body.describe()          # the history, in order, as a sentence a figure caption can carry

Each operation appends to :attr:`Body.history` and leaves the scene editable, so the same body can be rendered at any
point of its history -- the blank, three operations in, and finished -- and the pictures are a comparison of the part,
not of three different models.

What this is not: a mesh kernel. A bore here is drawn, not cut -- a dark throat under a chamfered mouth, which reads as
a hole at photographic distance because a hole is a shadow with a bright lip. A boolean would need a real modeller; the
day one is in the stack, ``bore`` is the operation that should call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .detail import bolt_circle, cone, pad, seam_ring, weld_bead
from .scene import Scene, disc_surface, rod_surface


@dataclass
class Operation:
    """One step of the history: what was done, and the surfaces it left behind."""

    name: str
    detail: str
    surfaces: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.detail


@dataclass
class Body:
    """A scene and the history that made it. Every method returns ``self``, so the operations read as a process."""

    scene: Scene
    history: List[Operation] = field(default_factory=list)
    material_of_skin: str = "as_cast"
    material_of_cut: str = "machined"

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, name: str, detail: str, before: int) -> None:
        self.history.append(Operation(name, detail, [s.name for s in self.scene.surfaces[before:]]))

    def describe(self, sep: str = "; ") -> str:
        """The history as one line, for a figure caption: ``cast blank; machined face 50 mm; bored 30 mm``."""
        return sep.join(op.detail for op in self.history)

    def materials(self) -> dict:
        """The body's own material overrides, ready for a render: the cast skin, the cut face, the oxide and the bore."""
        from .scene import MATERIALS
        return {k: dict(MATERIALS[k]) for k in (self.material_of_skin, self.material_of_cut, "heat_tint",
                                                "bore", "raw_weld") if k in MATERIALS}

    # ------------------------------------------------------------------ operations
    def machine_face(self, centre, normal, *, radius: float, height: float = 0.003, chamfer: float = 0.0012,
                     name: Optional[str] = None) -> "Body":
        """Cut a face: a pad standing proud of the casting with a chamfered rim, in the machined material. It is the
        contrast that does the work -- the same light that dulls the cast skin runs down the cut."""
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nb = nb / (np.linalg.norm(nb) or 1.0)
        c = np.asarray(centre, float).reshape(3) + nb * 0.0004          # a hair proud: no z-fighting with the skin
        nm = name or f"face{len(self.history):02d}"
        self.scene.surfaces.append(pad(c, nb, radius=radius, height=height, chamfer=chamfer,
                                       material=self.material_of_cut, name=nm))
        self._record("machine_face", f"face usinada Ø{2 * radius * 1000:.0f} mm", before)
        return self

    def bore(self, centre, normal, *, diameter: float, depth: float = 0.02, csk: float = 0.0015,
             name: Optional[str] = None) -> "Body":
        """Bore a hole: a dark throat under a chamfered mouth. Drawn, not cut -- see the module docstring -- because a
        hole at photographic distance is a shadow with a bright lip, and that is exactly what this builds: the throat is
        a dark disc on the face and the lip is a chamfer ring of the machined material. ``depth`` is recorded in the
        history and used for nothing else until there is a mesh kernel to cut with."""
        if diameter <= 0:
            raise ValueError("a bore needs a diameter greater than zero")
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nb = nb / (np.linalg.norm(nb) or 1.0)
        c = np.asarray(centre, float).reshape(3)
        nm = name or f"bore{len(self.history):02d}"
        r = 0.5 * float(diameter)
        self.scene.surfaces.append(disc_surface(c + nb * 0.0004, r, normal=nb, name=nm, material="bore",
                                                n=max(32, int(24 * max(1.0, r * 600)))))
        if csk > 0:                                                       # the chamfered mouth
            self.scene.surfaces.append(seam_ring(c + nb * 0.0006, nb, radius=r + 0.5 * csk, thickness=0.5 * csk,
                                                 material=self.material_of_cut, name=f"{nm}_mouth", n=40))
        self._record("bore", f"furado Ø{diameter * 1000:.0f} mm × {depth * 1000:.0f} mm", before)
        return self

    def weld(self, points, *, radius: float = 0.005, tint: float = 1.35, name: Optional[str] = None) -> "Body":
        """Weld along a path: the bead, and the heat-tinted band beside it. The band is what makes a weld look welded
        rather than glued -- the metal around the bead is not the metal of the plate."""
        before = len(self.scene.surfaces)
        P = np.asarray(points, float).reshape(-1, 3)
        nm = name or f"weld{len(self.history):02d}"
        self.scene.surfaces.append(weld_bead(P, radius=radius, material="raw_weld", name=nm))
        # the tinted band: a fatter, thinner bead in the same place, under the bead itself
        self.scene.surfaces.append(weld_bead(P, radius=radius * float(tint), material="heat_tint",
                                             name=f"{nm}_tint", segments=10))
        self._record("weld", f"solda Ø{2 * radius * 1000:.0f} mm em {len(P)} pontos", before)
        return self

    def bolt_pattern(self, centre, normal, *, radius: float, count: int, head_radius: float = 0.008,
                     name: Optional[str] = None) -> "Body":
        """Drill and fit a bolt pattern on a face: the machined face, the bolts and the chamfered mouths."""
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nm = name or f"bolts{len(self.history):02d}"
        self.scene.surfaces += bolt_circle(centre, nb, radius=radius, count=count, head_radius=head_radius,
                                           height=head_radius * 0.8, material=self.material_of_cut, name=nm)
        self._record("bolt_pattern", f"{count} parafusos Ø{2 * head_radius * 1000:.0f} mm a {radius * 1000:.0f} mm",
                     before)
        return self

    def finish(self, *, angle: float = 30.0, width: float = 0.0015, segments: int = 2, log=None) -> "Body":
        """Break every remaining sharp edge -- the last thing a shop does to a part -- by beveling the whole body.

        This is the one operation that rebuilds the mesh, so it returns the same :class:`Body` with a new scene and
        keeps the history; the surfaces are renamed by the round trip, which is why the history stores names as a
        record of what was there and not as a way to find it again."""
        from .bevel import bevel_scene

        self.scene = bevel_scene(self.scene, angle=angle, width=width, segments=segments, log=log)
        self.scene.materials = {**self.scene.materials, **self.materials()}
        before = 0
        self._record("finish", f"arestas quebradas {width * 1000:.1f} mm", before)
        return self


def cast_blank(*, radius: float = 0.11, height: float = 0.09, draft: float = 2.0, base: float = 1.16,
               material: Optional[str] = None, name: str = "blank") -> Body:
    """A housing as it leaves the mould: a drum with a drafted wall, a wider base and the as-cast skin.

    ``draft`` is in degrees and is not decoration -- a casting that has to come out of a mould cannot have a vertical
    wall, and the slight taper is one of the cues that says "cast" before any texture does."""
    sc = Scene()
    h, r = float(height), float(radius)
    r_top = r * (1.0 - np.tan(np.radians(float(draft))) * h / max(r, 1e-9))
    body_mat = material or "as_cast"
    sc.surfaces.append(cone([0, 0, 0], [0, 0, h], r, r_top, name=f"{name}_skin", material=body_mat, n=96))
    sc.surfaces.append(disc_surface([0, 0, h], r_top, normal=(0, 0, 1), name=f"{name}_top", material=body_mat,
                                    n=96))
    sc.surfaces.append(rod_surface([0, 0, 0], [0, 0, 0.012], r * float(base), name=f"{name}_base",
                                   material=body_mat, n=96))
    sc.surfaces.append(seam_ring([0, 0, h * 0.55], (0, 0, 1), radius=r * 0.997, thickness=0.0012,
                                 material=body_mat, name=f"{name}_parting", n=72))
    b = Body(sc, material_of_skin=body_mat)
    b.scene.materials = b.materials()
    b._record("cast", f"fundido bruto Ø{2 * r * 1000:.0f} × {h * 1000:.0f} mm, saída {draft:.1f}°", 0)
    return b


__all__ = ["Body", "Operation", "cast_blank"]
