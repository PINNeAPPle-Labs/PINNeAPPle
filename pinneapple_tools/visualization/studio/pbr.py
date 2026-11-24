"""Procedural PBR shaders for the materials a flat factor table cannot describe on its own.

An entry of ``MATERIALS`` (see :mod:`.scene`) carries the numbers every consumer can use — glTF, USD, the three.js
viewer — and optionally ``"shader": "<name>"``, naming one of the builders registered here that Blender alone runs
at render time. An entry with a shader still exports, everywhere else, as the flat material it declares; the shader
only adds what the export formats cannot carry: micro-bump, wear, spatter and streaks.

    "brushed_steel": {"color": [0.72, 0.73, 0.74], "metallic": 1.0, "roughness": 0.22,
                      "shader": "brushed_metal"}

The recipes are the ones the industrial pump-station example
(``examples/use_cases/physics_data_factory/blender_factory_render.py``) proved out — painted steel that chips to
bare metal, concrete with aggregate, fine manufacturing bump — rewritten against the table's parameters so that any
colour gets the same treatment, with the procedural detail driven by object coordinates (so it stays put when the
body moves, and always has a physical scale in metres).
"""
from __future__ import annotations

from typing import Callable, Dict

#: name -> builder(``bpy``, material, entry of ``MATERIALS``); each builder augments the Principled node tree the
#: flat material already has and returns the material
SHADERS: Dict[str, Callable] = {}


def shader(name: str):
    """Register a builder under ``name`` (used by ``"shader"`` in a ``MATERIALS`` entry)."""

    def register(fn):
        SHADERS[name] = fn
        return fn

    return register


# ------------------------------------------------------------------ the little the builders share
def _lin(c) -> tuple:
    """sRGB (the tables) -> linear (Blender sockets)."""
    return tuple(float(v) ** 2.2 for v in c)


def _tree(mat):
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    return nt, out, bsdf


def _socket(node, *names):
    """The first of ``names`` that the node has (Noise "Factor" in 5.x, "Fac" in 4.x)."""
    for n in names:
        if n in node.outputs:
            return node.outputs[n]
    raise KeyError(f"{node.bl_idname} has none of {names}")


def _coords(nt, kind: str = "Object", scale=None):
    """Texture coordinates of the object, optionally scaled (which is what makes streaks directional)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vec = tc.outputs[kind]
    if scale is None:
        return vec
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = tuple(float(v) for v in scale)
    nt.links.new(vec, mp.inputs["Vector"])
    return mp.outputs["Vector"]


def _noise(nt, vec, scale: float, detail: float = 6.0, rough: float = 0.6, distortion: float = 0.0):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = float(scale)
    n.inputs["Detail"].default_value = float(detail)
    n.inputs["Roughness"].default_value = float(rough)
    if "Distortion" in n.inputs:
        n.inputs["Distortion"].default_value = float(distortion)
    nt.links.new(vec, n.inputs["Vector"])
    return n


def _bump(nt, height, strength: float, distance: float | None = None):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = float(strength)
    if distance is not None and "Distance" in b.inputs:
        b.inputs["Distance"].default_value = float(distance)
    nt.links.new(height, b.inputs["Height"])
    return b.outputs["Normal"]


def _roughness_by(nt, bsdf, height, factor_lo: float, factor_hi: float, base: float):
    """Vary the roughness of ``bsdf`` with ``height`` between ``base*factor_lo`` and ``base*factor_hi``: a worn
    surface has both polished and dull patches, which is what reads as use rather than as a rendering artefact."""
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = float(base * factor_lo)
    mr.inputs["To Max"].default_value = float(base * factor_hi)
    nt.links.new(height, mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])


def _mix_over(nt, out, base_bsdf, layer_bsdf, mask, soft: float = 0.0):
    """Put ``layer_bsdf`` over ``base_bsdf`` where ``mask`` is bright, replacing the surface output. ``soft`` widens
    the transition (0 = the mask as it is: a chip with a hard edge)."""
    fed = mask
    if soft:
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = float(0.5 - soft / 2)
        mr.inputs["From Max"].default_value = float(0.5 + soft / 2)
        nt.links.new(mask, mr.inputs["Value"])
        fed = mr.outputs["Result"]
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fed, mix.inputs[0])
    nt.links.new(base_bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(layer_bsdf.outputs["BSDF"], mix.inputs[2])
    for link in list(nt.links):
        if link.to_node is out and link.to_socket.name == "Surface":
            nt.links.remove(link)
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mix


def _threshold(nt, height, position: float, width: float = 0.05, hard: bool = True):
    """A colour-ramp mask: black below ``position``, white above (``hard``) or fading over ``width``."""
    rp = nt.nodes.new("ShaderNodeValToRGB")
    el = rp.color_ramp.elements
    el[0].position = max(0.0, min(1.0, position))
    el[0].color = (0.0, 0.0, 0.0, 1.0)
    el[1].position = max(0.0, min(1.0, position + max(1e-3, width)))
    el[1].color = (1.0, 1.0, 1.0, 1.0)
    if hard:
        rp.color_ramp.interpolation = "CONSTANT"
    nt.links.new(height, rp.inputs[0])
    return _socket(rp, "Color")


# ------------------------------------------------------------------ the materials
@shader("painted_metal")
def painted_metal(bpy, mat, st):
    """Industrial paint with fine manufacturing bump, chipped through to bare metal where it wears (``wear``: the
    fraction of the surface, 0..0.4; ``metal_color``: what shows through). The pump casing of the factory example."""
    nt, out, paint = _tree(mat)
    vec = _coords(nt)
    fine = _noise(nt, vec, float(st.get("grain", 40.0)), detail=5.0, rough=0.6)
    normal = _bump(nt, _socket(fine, "Factor", "Fac"), 0.12, 0.003)
    nt.links.new(normal, paint.inputs["Normal"])
    wear = max(0.0, min(0.4, float(st.get("wear", 0.12))))
    loop = _noise(nt, vec, 8.0, detail=4.0, rough=0.8, distortion=0.5)
    mask = _threshold(nt, _socket(loop, "Factor", "Fac"), 1.0 - 2.0 * wear, 0.05, hard=True)
    bare = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bare.inputs["Base Color"].default_value = (*_lin(st.get("metal_color", (0.55, 0.56, 0.58))), 1.0)
    bare.inputs["Metallic"].default_value = 0.95
    bare.inputs["Roughness"].default_value = 0.22
    nt.links.new(normal, bare.inputs["Normal"])
    _mix_over(nt, out, paint, bare, mask)
    return mat


@shader("brushed_metal")
def brushed_metal(bpy, mat, st):
    """Brushed stainless: streaks along the object's x, a roughness that runs from polished to dull across the
    brush, and a bump too fine to see as bumps. ``streak`` stretches the pattern (higher = finer lines)."""
    nt, out, b = _tree(mat)
    streak = float(st.get("streak", 90.0))
    vec = _coords(nt, scale=(1.0, streak, streak))
    n = _noise(nt, vec, float(st.get("grain", 8.0)), detail=6.0, rough=0.55)
    nt.links.new(_bump(nt, _socket(n, "Factor", "Fac"), 0.10), b.inputs["Normal"])
    _roughness_by(nt, b, _socket(n, "Factor", "Fac"), 0.7, 1.5, float(st.get("roughness", 0.3)))
    return mat


@shader("cast_metal")
def cast_metal(bpy, mat, st):
    """Cast iron or a sand-cast housing: a pitted surface with dull and polished patches, no paint."""
    nt, out, b = _tree(mat)
    vec = _coords(nt)
    n = _noise(nt, vec, float(st.get("grain", 55.0)), detail=6.0, rough=0.75)
    nt.links.new(_bump(nt, _socket(n, "Factor", "Fac"), 0.30, 0.004), b.inputs["Normal"])
    _roughness_by(nt, b, _socket(n, "Factor", "Fac"), 0.85, 1.2, float(st.get("roughness", 0.6)))
    return mat


@shader("oxidized_metal")
def oxidized_metal(bpy, mat, st):
    """Bare metal under rust: the entry's colour is the metal, ``rust_color`` the oxide that covers it in patches,
    with a soft edge (rust does not chip like paint). ``rust``: 0..1, how much of it there is."""
    nt, out, metal = _tree(mat)
    vec = _coords(nt)
    pit = _noise(nt, vec, float(st.get("grain", 70.0)), detail=6.0, rough=0.7)
    nt.links.new(_bump(nt, _socket(pit, "Factor", "Fac"), 0.22, 0.004), metal.inputs["Normal"])
    amount = max(0.05, min(0.95, float(st.get("rust", 0.4))))
    patch = _noise(nt, vec, 6.0, detail=6.0, rough=0.7, distortion=0.4)
    mask = _threshold(nt, _socket(patch, "Factor", "Fac"), 1.0 - amount, 0.25, hard=False)
    oxide = nt.nodes.new("ShaderNodeBsdfPrincipled")
    oxide.inputs["Base Color"].default_value = (*_lin(st.get("rust_color", (0.34, 0.17, 0.08))), 1.0)
    oxide.inputs["Metallic"].default_value = 0.0
    oxide.inputs["Roughness"].default_value = 0.88
    nt.links.new(_bump(nt, _socket(pit, "Factor", "Fac"), 0.35, 0.006), oxide.inputs["Normal"])
    _mix_over(nt, out, metal, oxide, mask)
    return mat


@shader("concrete")
def concrete(bpy, mat, st):
    """Cast concrete: aggregate in two shades of the entry's colour, a coarse pit, and the dull roughness a floor or
    a wall has. ``aggregate``: the grain size in metres^-1 (higher = finer)."""
    nt, out, b = _tree(mat)
    vec = _coords(nt, kind="Object")
    agg = _noise(nt, vec, float(st.get("aggregate", 12.0)), detail=6.0, rough=0.7)
    rp = nt.nodes.new("ShaderNodeValToRGB")
    base = _lin(st.get("color", (0.62, 0.62, 0.60)))
    rp.color_ramp.elements[0].color = (*[c * 0.72 for c in base], 1.0)
    rp.color_ramp.elements[1].color = (*[c * 1.20 for c in base], 1.0)
    nt.links.new(_socket(agg, "Factor", "Fac"), rp.inputs[0])
    nt.links.new(_socket(rp, "Color"), b.inputs["Base Color"])
    nt.links.new(_bump(nt, _socket(agg, "Factor", "Fac"), 0.25), b.inputs["Normal"])
    _roughness_by(nt, b, _socket(agg, "Factor", "Fac"), 0.92, 1.05, float(st.get("roughness", 0.9)))
    return mat


@shader("galvanized")
def galvanized(bpy, mat, st):
    """Hot-dip galvanising: the spangle of zinc crystals, which is a voronoi cell pattern, over the entry's colour."""
    nt, out, b = _tree(mat)
    vec = _coords(nt)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = float(st.get("spangle", 22.0))
    nt.links.new(vec, vor.inputs["Vector"])
    nt.links.new(_bump(nt, vor.outputs["Distance"], 0.18), b.inputs["Normal"])
    _roughness_by(nt, b, vor.outputs["Distance"], 0.75, 1.3, float(st.get("roughness", 0.42)))
    return mat


@shader("rubber")
def rubber(bpy, mat, st):
    """Rubber, a matte seal or a belt: a fine pebbled bump and a roughness that barely varies."""
    nt, out, b = _tree(mat)
    vec = _coords(nt)
    n = _noise(nt, vec, float(st.get("grain", 120.0)), detail=6.0, rough=0.6)
    nt.links.new(_bump(nt, _socket(n, "Factor", "Fac"), 0.18, 0.001), b.inputs["Normal"])
    _roughness_by(nt, b, _socket(n, "Factor", "Fac"), 0.95, 1.08, float(st.get("roughness", 0.85)))
    return mat


@shader("as_cast")
def as_cast(bpy, mat, st):
    """The skin a part has the moment it leaves the mould: a sand grain too fine to see as bumps, a dull roughness that
    varies in patches, and no highlight sharp enough to read as machined. ``grain`` is the sand; ``mottle`` how much the
    roughness wanders (a casting is never uniform)."""
    nt, out, b = _tree(mat)
    vec = _coords(nt)
    sand = _noise(nt, vec, float(st.get("grain", 260.0)), detail=8.0, rough=0.7)
    nt.links.new(_bump(nt, _socket(sand, "Factor", "Fac"), 0.22, 0.0006), b.inputs["Normal"])
    patch = _noise(nt, vec, float(st.get("mottle", 14.0)), detail=4.0, rough=0.6)
    _roughness_by(nt, b, _socket(patch, "Factor", "Fac"), 0.9, 1.25, float(st.get("roughness", 0.72)))
    return mat


@shader("machined")
def machined(bpy, mat, st):
    """A machined face: tool marks in one direction, a roughness that is low and even, and the fine chatter of a
    cutter that is never perfectly still. The contrast with :func:`as_cast` on the same part is the point -- it is how
    a viewer reads which faces were cut and which were merely cast."""
    nt, out, b = _tree(mat)
    step = float(st.get("step", 420.0))          # lines per metre: a face mill's step-over
    vec = _coords(nt, scale=(1.0, step, step))
    marks = _noise(nt, vec, float(st.get("grain", 12.0)), detail=3.0, rough=0.5)
    nt.links.new(_bump(nt, _socket(marks, "Factor", "Fac"), 0.10, 0.0004), b.inputs["Normal"])
    chatter = _noise(nt, _coords(nt, scale=(1.0, 1.0, 1.0)), float(st.get("chatter", 6.0)), detail=2.0, rough=0.4)
    _roughness_by(nt, b, _socket(chatter, "Factor", "Fac"), 0.85, 1.3, float(st.get("roughness", 0.16)))
    return mat


@shader("heat_tint")
def heat_tint(bpy, mat, st):
    """The straw and blue a weld leaves beside the bead: the entry's colour is the metal, ``tint`` the oxide, and the
    band follows a stretched noise so it is never a painted stripe."""
    nt, out, metal = _tree(mat)
    vec = _coords(nt, scale=(1.0, float(st.get("band", 40.0)), float(st.get("band", 40.0))))
    n = _noise(nt, vec, 30.0, detail=5.0, rough=0.6)
    mask = _threshold(nt, _socket(n, "Factor", "Fac"), float(st.get("amount", 0.45)), 0.35, hard=False)
    oxide = nt.nodes.new("ShaderNodeBsdfPrincipled")
    oxide.inputs["Base Color"].default_value = (*_lin(st.get("tint", (0.36, 0.28, 0.42))), 1.0)
    oxide.inputs["Metallic"].default_value = 0.85
    oxide.inputs["Roughness"].default_value = 0.45
    _mix_over(nt, out, metal, oxide, mask)
    return mat


__all__ = ["SHADERS", "shader", "painted_metal", "brushed_metal", "cast_metal", "oxidized_metal", "concrete",
           "galvanized", "rubber", "as_cast", "machined", "heat_tint"]
