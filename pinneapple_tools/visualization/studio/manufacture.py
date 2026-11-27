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

import numpy as np

from .detail import bolt_circle, pad, seam_ring, weld_bead
from .scene import Scene, disc_surface, rod_surface
from .solid import solid_drum


@dataclass
class Operation:
    """One step of the history: what was done, the surfaces it left behind, and the numbers it was done with -- the
    last one is what lets the history be written down as data and replayed."""

    name: str
    detail: str
    surfaces: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.detail

    def as_spec(self) -> dict:
        return {"op": self.name, "params": _jsonable(self.params)}


def _jsonable(value):
    """Numpy out, plain Python in: a history is data, and data has to survive a JSON file."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class Body:
    """A scene and the history that made it. Every method returns ``self``, so the operations read as a process."""

    scene: Scene
    history: list[Operation] = field(default_factory=list)
    material_of_skin: str = "as_cast"
    material_of_cut: str = "machined"
    blank_params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, name: str, detail: str, before: int, params: dict | None = None) -> None:
        self.history.append(Operation(name, detail, [s.name for s in self.scene.surfaces[before:]], params or {}))

    def spec(self, *, blank: dict | None = None) -> dict:
        """The body as data: the blank's parameters and every operation with its numbers, in order. ``build`` replays
        it, so a route sheet is a file instead of a script."""
        return {"blank": _jsonable(blank or self.blank_params), "operations": [op.as_spec() for op in self.history[1:]]}

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
                     name: str | None = None) -> Body:
        """Cut a face: a pad standing proud of the casting with a chamfered rim, in the machined material. It is the
        contrast that does the work -- the same light that dulls the cast skin runs down the cut."""
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nb = nb / (np.linalg.norm(nb) or 1.0)
        c = np.asarray(centre, float).reshape(3) + nb * 0.0004          # a hair proud: no z-fighting with the skin
        nm = name or f"face{len(self.history):02d}"
        self.scene.surfaces.append(pad(c, nb, radius=radius, height=height, chamfer=chamfer,
                                       material=self.material_of_cut, name=nm))
        self._record("machine_face", f"face usinada Ø{2 * radius * 1000:.0f} mm", before,
                     {"centre": centre, "normal": normal, "radius": radius, "height": height, "chamfer": chamfer})
        return self

    def bore(self, centre, normal, *, diameter: float, depth: float = 0.02, csk: float = 0.0015,
             cut: bool | None = None, name: str | None = None) -> Body:
        """Bore a hole. With a kernel it is **cut**: :mod:`.solid.boolean` subtracts a closed cylinder and leaves an
        opening with an interior wall and a silhouette that changes with the camera. Without one -- or on a body whose
        shells are open -- it is **drawn**: a dark throat under a chamfered mouth, which is what a hole looks like at
        photographic distance, because a hole is a shadow with a bright lip.

        ``cut=None`` decides by itself (a kernel here, a closed body there) and the history says which happened, which
        is the part that matters: a figure should not claim a cut it did not make."""
        if diameter <= 0:
            raise ValueError("a bore needs a diameter greater than zero")
        from .solid import boolean, cutter_cylinder, is_closed
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nb = nb / (np.linalg.norm(nb) or 1.0)
        c = np.asarray(centre, float).reshape(3)
        nm = name or f"bore{len(self.history):02d}"
        r = 0.5 * float(diameter)
        closed_body = all(is_closed(s) for s in self.scene.surfaces)
        want_cut = closed_body if cut is None else bool(cut)
        if want_cut and not closed_body:
            raise ValueError("a cut bore needs closed shells: this body has open ones (use cut=False to draw it)")
        if want_cut:
            tool = cutter_cylinder(c - nb * float(depth), c + nb * 0.002, r, n=max(24, int(400 * r)))
            if csk > 0:                                   # the mouth: a wider, shallower pass
                tool = cutter_cylinder(c - nb * float(depth), c + nb * 0.002, r + csk, n=max(24, int(400 * r)))
            self.scene = boolean(self.scene, [tool], op="difference")
            self.scene.materials = {**self.scene.materials, **self.materials()}
            self._record("bore", f"furado (cortado) Ø{diameter * 1000:.0f} mm × {depth * 1000:.0f} mm", 0,
                         {"centre": centre, "normal": normal, "diameter": diameter, "depth": depth, "csk": csk,
                          "cut": True})
            return self
        self.scene.surfaces.append(disc_surface(c + nb * 0.0004, r, normal=nb, name=nm, material="bore",
                                                n=max(32, int(24 * max(1.0, r * 600)))))
        if csk > 0:                                       # the chamfered mouth
            self.scene.surfaces.append(seam_ring(c + nb * 0.0006, nb, radius=r + 0.5 * csk, thickness=0.5 * csk,
                                                 material=self.material_of_cut, name=f"{nm}_mouth", n=40))
        self._record("bore", f"furado (desenhado) Ø{diameter * 1000:.0f} mm × {depth * 1000:.0f} mm", before,
                     {"centre": centre, "normal": normal, "diameter": diameter, "depth": depth, "csk": csk,
                      "cut": False})
        return self

    def shell(self, *, thickness: float = 0.010, floor: float = 0.010, name: str | None = None) -> Body:
        """Hollow the casting: subtract an inner volume so the body has a wall of ``thickness`` and a floor of
        ``floor``. This is the operation a boolean was needed for -- the geometry inside did not exist before, and no
        amount of surface detail can stand in for it."""
        from .solid import boolean, is_closed, solid_drum

        if thickness <= 0 or floor < 0:
            raise ValueError("a shell needs a wall thickness greater than zero and a floor that is not negative")
        if not all(is_closed(s) for s in self.scene.surfaces):
            raise ValueError("shelling needs a closed body: cast_blank(closed=True)")
        # the body's own extent, so the inner volume follows whatever blank it is: no hard-coded dimensions
        V = np.vstack([s.vertices for s in self.scene.surfaces])
        z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
        near_top = np.linalg.norm(V[V[:, 2] >= z1 - 0.05 * (z1 - z0), :2], axis=1)
        r = float(near_top.max())      # the wall at the mouth, not the base flange: a drafted casting narrows upwards
        # start a hair above the floor: a cutter whose base plane lies exactly on the base's top face makes the boolean
        # solve a coplanar pair, and the base comes back non-manifold
        lift = 5e-4
        inner = solid_drum([0, 0, z0 + float(floor) + lift], r - float(thickness), (z1 - z0 - float(floor)) + 0.02,
                           n=96, name=name or f"shell{len(self.history):02d}", material="bore")
        self.scene = boolean(self.scene, [inner], op="difference")
        self.scene.materials = {**self.scene.materials, **self.materials()}
        self._record("shell", f"vazado, parede {thickness * 1000:.0f} mm e fundo {floor * 1000:.0f} mm", 0,
                     {"thickness": thickness, "floor": floor})
        return self

    def weld(self, points, *, radius: float = 0.005, tint: float = 1.35, name: str | None = None) -> Body:
        """Weld along a path: the bead, and the heat-tinted band beside it. The band is what makes a weld look welded
        rather than glued -- the metal around the bead is not the metal of the plate."""
        before = len(self.scene.surfaces)
        P = np.asarray(points, float).reshape(-1, 3)
        nm = name or f"weld{len(self.history):02d}"
        self.scene.surfaces.append(weld_bead(P, radius=radius, material="raw_weld", name=nm))
        # the tinted band: a fatter, thinner bead in the same place, under the bead itself
        self.scene.surfaces.append(weld_bead(P, radius=radius * float(tint), material="heat_tint",
                                             name=f"{nm}_tint", segments=10))
        self._record("weld", f"solda Ø{2 * radius * 1000:.0f} mm em {len(P)} pontos", before,
                     {"points": P, "radius": radius, "tint": tint})
        return self

    def bolt_pattern(self, centre, normal, *, radius: float, count: int, head_radius: float = 0.008,
                     name: str | None = None) -> Body:
        """Drill and fit a bolt pattern on a face: the machined face, the bolts and the chamfered mouths."""
        before = len(self.scene.surfaces)
        nb = np.asarray(normal, float).reshape(3)
        nm = name or f"bolts{len(self.history):02d}"
        self.scene.surfaces += bolt_circle(centre, nb, radius=radius, count=count, head_radius=head_radius,
                                           height=head_radius * 0.8, material=self.material_of_cut, name=nm)
        self._record("bolt_pattern", f"{count} parafusos Ø{2 * head_radius * 1000:.0f} mm a {radius * 1000:.0f} mm",
                     before, {"centre": centre, "normal": normal, "radius": radius, "count": count,
                              "head_radius": head_radius})
        return self

    def finish(self, *, angle: float = 30.0, width: float = 0.0015, segments: int = 2, log=None) -> Body:
        """Break every remaining sharp edge -- the last thing a shop does to a part -- by beveling the whole body.

        This is the one operation that rebuilds the mesh, so it returns the same :class:`Body` with a new scene and
        keeps the history; the surfaces are renamed by the round trip, which is why the history stores names as a
        record of what was there and not as a way to find it again."""
        from .bevel import bevel_scene

        self.scene = bevel_scene(self.scene, angle=angle, width=width, segments=segments, log=log)
        self.scene.materials = {**self.scene.materials, **self.materials()}
        before = 0
        self._record("finish", f"arestas quebradas {width * 1000:.1f} mm", before,
                     {"angle": angle, "width": width, "segments": segments})
        return self


def cast_blank(*, radius: float = 0.11, height: float = 0.09, draft: float = 2.0, base: float = 1.16,
               running: float = 0.012, material: str | None = None, name: str = "blank",
               closed: bool = True) -> Body:
    """A housing as it leaves the mould: a drum with a drafted wall, a wider base and the as-cast skin.

    ``draft`` is in degrees and is not decoration -- a casting that has to come out of a mould cannot have a vertical
    wall, and the slight taper is one of the cues that says "cast" before any texture does.

    ``closed`` builds the shell as a solid (a closed drum), which is what a real :meth:`Body.bore` needs; the drafted
    wall then lives on the solid's own rings, and there is no visible difference from the shell at render time.
    """
    sc = Scene()
    h, r = float(height), float(radius)
    r_top = r * (1.0 - np.tan(np.radians(float(draft))) * h / max(r, 1e-9))
    body_mat = material or "as_cast"
    if closed:
        sc.surfaces.append(solid_drum([0, 0, 0], r, h, radius_top=r_top, n=96, name=f"{name}_skin",
                                      material=body_mat))
        sc.surfaces.append(solid_drum([0, 0, 0], r * float(base), float(running), n=96, name=f"{name}_base",
                                      material=body_mat))
    else:
        from .detail import cone
        sc.surfaces.append(cone([0, 0, 0], [0, 0, h], r, r_top, name=f"{name}_skin", material=body_mat, n=96))
        sc.surfaces.append(disc_surface([0, 0, h], r_top, normal=(0, 0, 1), name=f"{name}_top", material=body_mat,
                                        n=96))
        sc.surfaces.append(rod_surface([0, 0, 0], [0, 0, float(running)], r * float(base), name=f"{name}_base",
                                       material=body_mat, n=96))
    sc.surfaces.append(seam_ring([0, 0, h * 0.55], (0, 0, 1), radius=r * 0.997, thickness=0.0012,
                                 material=body_mat, name=f"{name}_parting", n=72))
    b = Body(sc, material_of_skin=body_mat)
    b.scene.materials = b.materials()
    b.blank_params = {"radius": radius, "height": height, "draft": draft, "base": base, "running": running,
                      "material": material, "name": name, "closed": closed}
    b._record("cast", f"fundido bruto Ø{2 * r * 1000:.0f} × {h * 1000:.0f} mm, saída {draft:.1f}°", 0,
              dict(b.blank_params))
    return b


def build(spec: dict, *, log=None) -> Body:
    """A body from its route sheet: ``{"blank": {...}, "operations": [{"op": "bore", "params": {...}}, ...]}``.

    The operations are the methods of :class:`Body`, called in the order they appear -- so a history is data a shop can
    write down, diff and keep next to the drawing, and the same file renders the same part anywhere."""
    blank = {k: v for k, v in dict(spec.get("blank") or {}).items() if v is not None}
    body = cast_blank(**blank)
    for step in spec.get("operations") or []:
        op = step.get("op")
        fn = getattr(body, op or "", None)
        if not callable(fn):
            raise ValueError(f"{op!r} is not an operation of a body: {[o.name for o in body.history]}")
        params = dict(step.get("params") or {})
        fn(**params)
    if log:
        log(body.describe(" | "))
    return body


__all__ = ["Body", "Operation", "cast_blank", "build"]
