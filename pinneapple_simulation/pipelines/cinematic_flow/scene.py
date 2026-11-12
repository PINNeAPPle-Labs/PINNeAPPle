"""The render half of a film. Runs inside Blender (``blender --background --python scene.py -- cfg.json``).

It is deliberately the only file that touches ``bpy``: the solve side writes NPZ and returns, so the physics
can be re-run, checked and re-used without a render, and the look can be re-lit without re-solving.

What is built, in the order the light needs: a dark studio (a floor that reflects, two strip lights overhead,
a key and a rim), the body with a paint that holds a highlight rather than a colour map, and the smoke as an
OpenVDB grid written from the simulated field — vorticity, not decoration.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np


# ---------------------------------------------------------------------------------------- helpers
def _set(obj, attr, value):
    try:
        setattr(obj, attr, value)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def _mat(name, base=(0.05, 0.05, 0.06), rough=0.3, metallic=0.0, coat=0.0, coat_rough=0.05, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if b is None:
        b = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        out = m.node_tree.nodes.get("Material Output") or m.node_tree.nodes.new("ShaderNodeOutputMaterial")
        m.node_tree.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    _set(b.inputs.get("Coat Weight", b.inputs.get("Clearcoat")), "default_value", coat)
    _set(b.inputs.get("Coat Roughness", b.inputs.get("Clearcoat Roughness")), "default_value", coat_rough)
    if alpha < 1.0:
        _set(b.inputs.get("Alpha"), "default_value", alpha)
    return m


def _emissive(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------------------- smoke
def _upsample(a: np.ndarray, up: int) -> np.ndarray:
    """Trilinear refinement by ``up``, done separably so it stays numpy — Blender's Python has no scipy."""
    for axis in range(3):
        n = a.shape[axis]
        m = n * up
        t = np.clip((np.arange(m) + 0.5) / up - 0.5, 0.0, n - 1.0)
        i0 = t.astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        w = (t - i0).astype(np.float32)
        shape = [1, 1, 1]
        shape[axis] = m
        w = w.reshape(shape)
        a = (np.take(a, i0, axis=axis) * (1.0 - w) + np.take(a, i1, axis=axis) * w).astype(np.float32)
    return a


def _taper(d: np.ndarray, frac: float) -> np.ndarray:
    """Fade the field into the tunnel walls.

    The volume is a box, and a box has edges: where the wake still has density at the boundary the render shows
    a straight cut, which reads as a box rather than as air. A raised-cosine over the outer ``frac`` of each axis
    takes the field to zero before it reaches the wall, so the smoke dissolves instead of ending.
    """
    if frac <= 0.0:
        return d
    for axis in range(3):
        n = d.shape[axis]
        edge = np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(1.0, frac * n)
        w = 0.5 - 0.5 * np.cos(np.pi * np.clip(edge, 0.0, 1.0))
        shape = [1, 1, 1]
        shape[axis] = n
        d = d * w.reshape(shape).astype(np.float32)
    return d


def write_vdb(npz_path: str, vdb_path: str, *, voxel: float, smoke: dict) -> str:
    """Turn one simulated field into an OpenVDB density grid.

    The stored fields are the two the picture needs — speed deficit and vorticity — and the scalar is built here,
    so the threshold that decides what reads as smoke can be tuned against the same solve instead of forcing a
    re-run. Everything below the cut is dropped: the grid stays sparse and the empty air of the tunnel costs
    nothing.
    """
    import openvdb as vdb

    z = np.load(npz_path)
    speed, q = z["speed"].astype(np.float32), z["q"].astype(np.float32)
    fluid = speed > 1e-6
    pos = np.maximum(q, 0.0)
    ref = max(float(np.percentile(pos[fluid], smoke.get("percentile", 99.0))) if fluid.any() else 1.0, 1e-12)
    on = np.clip(pos / ref, 0.0, 1.0) ** float(smoke.get("power", 1.0))
    deficit = np.clip(1.0 - speed, 0.0, 1.0)
    d = float(smoke.get("vort_weight", 1.0)) * on + float(smoke.get("deficit_weight", 0.35)) * deficit
    floor = float(smoke.get("floor", 0.18))
    d = np.clip((d - floor) / max(1.0 - floor, 1e-9), 0.0, 1.0) * fluid
    d *= float(smoke.get("gain", 1.0))
    d[d < 0.01] = 0.0
    d = _taper(d, float(smoke.get("taper", 0.0)))
    # The picture is only as smooth as the stored field, and the stored field is subsampled to keep the files
    # small. Interpolating to a finer grid before writing does not invent detail, but it removes the stair-step
    # of a 8 cm voxel along a curved wing mirror, which is what makes a volume read as blocks instead of smoke.
    up = int(smoke.get("resample", 1) or 1)
    if up > 1:
        d = _upsample(d, up)
        voxel = voxel / up
    g = vdb.FloatGrid()
    g.copyFromArray(np.ascontiguousarray(d.astype(np.float32)))
    g.name = "density"
    g.transform = vdb.createLinearTransform(voxelSize=float(voxel))
    vdb.write(vdb_path, g)
    return vdb_path


def _volume_material(name, *, color, gain, emission, noise_scale, fiber, hot, floor_mix=0.0):
    """The smoke shader, built by hand around the grid.

    ``Principled Volume``'s own emission is emitted throughout the volume's bounds whatever the density is, so a
    sparse field would light up as a solid slab. Here the emission is a separate node multiplied by the density
    itself: the glow appears exactly where the flow is, and nowhere else. The colour walks from the tint of the
    cold wake to white where the field is dense, which is what reads as depth in a cloud.
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    info = nt.nodes.new("ShaderNodeVolumeInfo")          # the "density" grid of the imported VDB
    dens = nt.nodes.new("ShaderNodeMath")
    dens.operation = "MULTIPLY"
    dens.inputs[1].default_value = float(gain)
    nt.links.new(info.outputs["Density"], dens.inputs[0])
    if fiber > 0.0:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = noise_scale
        tex.inputs["Detail"].default_value = 6.0
        tex.inputs["Roughness"].default_value = 0.6
        nramp = nt.nodes.new("ShaderNodeValToRGB")
        nramp.color_ramp.elements[0].position = 0.30
        nramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        nramp.color_ramp.elements[1].position = 0.80
        nramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        nmix = nt.nodes.new("ShaderNodeMix")
        nmix.data_type = "FLOAT"
        nmix.inputs["Factor"].default_value = float(fiber)
        # the noise multiplies the field down to zero, not down to a fraction of itself: mixing with 1.0 leaves
        # every voxel at least a fifth of the grid's density, and a fifth of a dense field over the whole tunnel
        # is the milky haze that hides the body
        nmix.inputs[2].default_value = float(floor_mix)
        nmod = nt.nodes.new("ShaderNodeMath")
        nmod.operation = "MULTIPLY"
        nt.links.new(tex.outputs["Fac"], nramp.inputs["Fac"])
        nt.links.new(nramp.outputs["Color"], nmix.inputs[3])
        nt.links.new(nmix.outputs[0], nmod.inputs[0])
        nt.links.new(dens.outputs[0], nmod.inputs[1])
        density_out = nmod.outputs[0]
    else:
        density_out = dens.outputs[0]
    # colour as a function of the field itself: thin wake is the tint, dense cores go to white
    cramp = nt.nodes.new("ShaderNodeValToRGB")
    cramp.color_ramp.elements[0].position = 0.0
    cramp.color_ramp.elements[0].color = (*color, 1.0)
    cramp.color_ramp.elements[1].position = 1.0
    cramp.color_ramp.elements[1].color = (*hot, 1.0)
    nt.links.new(dens.outputs[0], cramp.inputs["Fac"])
    scatter = nt.nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Anisotropy"].default_value = 0.35
    nt.links.new(cramp.outputs["Color"], scatter.inputs["Color"])
    nt.links.new(density_out, scatter.inputs["Density"])
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(cramp.outputs["Color"], emit.inputs["Color"])
    estr = nt.nodes.new("ShaderNodeMath")
    estr.operation = "MULTIPLY"
    estr.inputs[1].default_value = float(emission)
    nt.links.new(density_out, estr.inputs[0])
    nt.links.new(estr.outputs[0], emit.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(scatter.outputs["Volume"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Volume"])
    return m, dens


def build_smoke(cfg) -> list:
    """Import the density grids and crossfade them over the frame range.

    The windows overlap by half a fade on each side, so the outgoing grid and the incoming one add to a constant
    density across the hand-off. Keyed end to end instead, both would be at zero on the same frame and the smoke
    would blink out once per snapshot — a fault that only exists in the animation, never in a still.
    """
    frames = int(cfg["render"]["frames"])
    snaps = cfg["snapshots"]
    n = len(snaps)
    span0 = (frames - 1.0) / max(n - 1, 1)
    half = max(1.0, span0 * 0.2)
    span = (frames - 1.0 - half) / max(n - 1, 1)       # the last grid is fully on by the final frame
    voxel = float(cfg["voxel_size"])
    origin = cfg["origin"]
    # linear keys: an eased hand-off makes the sum of the two ramps dip below one in the middle of the fade
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    objs = []
    for k, npz in enumerate(snaps):
        vdb_path = os.path.join(cfg["out_dir"], "vdb", f"smoke_{k:03d}.vdb")
        os.makedirs(os.path.dirname(vdb_path), exist_ok=True)
        write_vdb(npz, vdb_path, voxel=voxel, smoke=cfg["style"].get("smoke", {}))
        bpy.ops.object.volume_import(filepath=vdb_path)
        ob = bpy.context.object
        ob.name = f"smoke_{k:03d}"
        ob.location = origin
        mat, dens = _volume_material(f"smoke_mat_{k}", color=cfg["style"]["tint"], gain=1.0,
                                     emission=cfg["style"]["emission"],
                                     noise_scale=cfg["style"]["noise_scale"], fiber=cfg["style"]["fiber"],
                                     floor_mix=cfg["style"].get("fiber_floor", 0.0),
                                     hot=cfg["style"].get("hot", [0.86, 0.94, 1.0]))
        ob.data.materials.append(mat)
        t = 1.0 + k * span
        keys = [(t - half, 0.0), (t + half, 1.0)]
        if k < n - 1:
            keys += [(t + span - half, 1.0), (t + span + half, 0.0)]
        if k == 0:
            keys[0] = (-1.0, 1.0)                   # the film opens on the first grid, already at full density
        key = dens.inputs[1]
        for f, v in keys:
            key.default_value = v
            key.keyframe_insert("default_value", frame=f)
        objs.append(ob)
    return objs


# ---------------------------------------------------------------------------------------- geometry
def build_bodies(cfg):
    parts = []
    for entry in cfg["bodies"]:
        bpy.ops.wm.ply_import(filepath=entry["path"])
        ob = bpy.context.object
        ob.name = entry["name"]
        for p in ob.data.polygons:
            p.use_smooth = True
        kind = entry.get("material", "paint")
        if kind == "tyre":
            m = _mat("tyre", base=(0.016, 0.016, 0.018), rough=0.62, metallic=0.0)
        elif kind == "rim":
            # a mirror here reads as a white disc: the rim is the face of the wheel and it reflects the strip
            # light straight into the lens, which is what blew the wheels out to flat grey. A dark satin alloy
            # keeps the highlight in the spokes instead of across the whole face.
            m = _mat("rim", base=(0.055, 0.060, 0.070), rough=0.42, metallic=0.85)
        elif kind == "glass":
            m = _mat("glass", base=(0.006, 0.010, 0.018), rough=0.04, metallic=0.0, coat=1.0)
        elif kind == "lamp":
            m = _emissive("lamp", (0.85, 0.92, 1.0), 12.0)
        elif kind == "tail":
            m = _emissive("tail", (1.0, 0.10, 0.06), 6.0)
        else:
            # a clearcoat at roughness 0.03 is a mirror, and the strip light lands on it as a clipped white
            # sheet across the roof. The coat stays (that is what makes paint read as paint) but softens, so the
            # highlight is a broad gradient the way the reference's is.
            paint = cfg["style"]
            m = _mat("paint", base=paint["paint"], rough=float(paint.get("paint_rough", 0.26)),
                     metallic=float(paint.get("paint_metallic", 0.55)), coat=1.0,
                     coat_rough=float(paint.get("coat_rough", 0.10)))
        ob.data.materials.append(m)
        parts.append(ob)
    return parts


def build_studio(cfg):
    sc = bpy.context.scene
    frame = cfg["frame"]
    center = np.asarray(frame["center"], float)
    radius = float(frame["radius"])
    ground = cfg.get("ground")
    # world: a dark room with a cold gradient — a flat black world leaves the metallic paint with nothing to
    # reflect, and a car lit only by lamps reads as a silhouette
    sc.world = bpy.data.worlds.new("studio")
    sc.world.use_nodes = True
    wnt = sc.world.node_tree
    wnt.nodes.clear()
    wout = wnt.nodes.new("ShaderNodeOutputWorld")
    bg = wnt.nodes.new("ShaderNodeBackground")
    tc = wnt.nodes.new("ShaderNodeTexCoord")
    sep = wnt.nodes.new("ShaderNodeSeparateXYZ")
    mr = wnt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.4
    mr.inputs["From Max"].default_value = 0.9
    ramp = wnt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.004, 0.007, 0.013, 1.0)
    ramp.color_ramp.elements[1].color = (0.055, 0.085, 0.145, 1.0)
    wnt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    wnt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    wnt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    wnt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    wnt.links.new(bg.outputs["Background"], wout.inputs["Surface"])
    # floor
    z_floor = ground if ground is not None else center[2] - 2.4 * radius
    bpy.ops.mesh.primitive_plane_add(size=30 * radius, location=(center[0], center[1], z_floor))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(_mat("floor", base=(0.010, 0.013, 0.021), rough=0.16, metallic=0.35))
    # two strip lights overhead, the long white lines of the reference, angled across the body so they reflect
    light = cfg["style"].get("light", {})
    strip_gain = float(light.get("strip", 45.0))
    for dy, ang, strength in ((-0.9 * radius, 6.0, strip_gain), (1.5 * radius, -5.0, strip_gain * 0.78)):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(center[0] + 0.2 * radius, center[1] + dy,
                                                          z_floor + 3.4 * radius))
        strip = bpy.context.object
        strip.name = "strip"
        strip.scale = (3.4 * radius, 0.035 * radius, 1)
        strip.rotation_euler = (math.radians(ang), 0, math.radians(12))
        strip.data.materials.append(_emissive("strip", (1.0, 0.98, 0.95), strength))
    # a soft box above and a key/rim pair, all scaled with the subject
    gain = float(light.get("area", 520.0)) * radius * radius
    for loc, energy, size, color in (
            ((0.2 * radius, 0.1 * radius, 2.6 * radius), 0.55 * gain, 2.4 * radius, (0.80, 0.90, 1.0)),
            ((2.2 * radius, -2.4 * radius, 2.0 * radius), gain, 1.5 * radius, (0.72, 0.84, 1.0)),
            ((-2.6 * radius, 1.6 * radius, 1.6 * radius), 1.7 * gain, 1.1 * radius, (0.55, 0.74, 1.0))):
        bpy.ops.object.light_add(type="AREA", location=(center[0] + loc[0], center[1] + loc[1], center[2] + loc[2]))
        lamp = bpy.context.object
        lamp.data.energy = energy
        lamp.data.size = size
        lamp.data.color = color
        lamp.rotation_euler = (0.0 if loc[2] > 2.0 * radius else math.radians(58), 0,
                               math.radians(38) if loc[0] > 0 else math.radians(-38))


def build_camera(cfg):
    sc = bpy.context.scene
    frame = cfg["frame"]
    center = np.asarray(frame["center"], float)
    radius = float(frame["radius"])
    cam_cfg = cfg["camera"]
    frames = int(cfg["render"]["frames"])
    bpy.ops.object.empty_add(location=center)
    pivot = bpy.context.object
    pivot.name = "pivot"
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    sc.camera = cam
    cam.data.lens = float(cam_cfg["lens"])
    cam.parent = pivot
    d0, d1 = float(cam_cfg["dist0"]) * radius, float(cam_cfg["dist1"]) * radius
    e0, e1 = math.radians(cam_cfg["elev0"]), math.radians(cam_cfg["elev1"])
    a0, a1 = math.radians(cam_cfg["az0"]), math.radians(cam_cfg["az1"])
    for f, t in ((1.0, 0.0), (frames, 1.0)):
        d = d0 + (d1 - d0) * t
        e = e0 + (e1 - e0) * t
        a = a0 + (a1 - a0) * t
        cam.location = (d * math.cos(e) * math.cos(a), d * math.cos(e) * math.sin(a), d * math.sin(e))
        cam.keyframe_insert("location", frame=f)
    # the camera keeps the subject in frame while it travels: a track-to on the pivot, so the dolly and the
    # orbit never need the rotation to be recomputed by hand
    track = cam.constraints.new("TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = d0 * 0.98
    cam.data.dof.aperture_fstop = float(cam_cfg.get("fstop", 2.4))
    return cam


def _volume_range(cfg) -> tuple[float, float]:
    """Where the volume ray starts and ends, in metres from the camera.

    EEVEE marches the volume a fixed number of samples between these two distances, so the step is the range
    divided by the sample count. Left at the old default of 0.05 m to 400 m, 96 samples gives a 4 m step over a
    field whose voxels are 8 cm: every wisp is smeared into one fog and the picture is a milky blanket no matter
    how sparse the grid is. The range is the real distance from the camera to the tunnel's far corner, so the
    step follows the subject instead of the scene's clip distance.
    """
    sc = bpy.context.scene
    origin = np.asarray(cfg["origin"], float)
    nx, ny, nz = cfg.get("shape") or (1, 1, 1)
    voxel = float(cfg["voxel_size"])
    far = origin + np.asarray([nx, ny, nz], float) * voxel
    corners = np.array([[x, y, z] for x in (origin[0], far[0]) for y in (origin[1], far[1])
                        for z in (origin[2], far[2])])
    eye = sc.camera.matrix_world.translation if sc.camera else np.zeros(3)
    d = np.linalg.norm(corners - np.asarray(eye, float), axis=1)
    return float(cfg["render"].get("volumetric_start", 0.05)), float(d.max() * 1.02)


def setup_render(cfg):
    sc = bpy.context.scene
    r = cfg["render"]
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = int(r["res_x"])
    sc.render.resolution_y = int(r["res_y"])
    sc.render.fps = int(r.get("fps", 24))
    sc.frame_start = int(r.get("frame_start", 1))
    sc.frame_end = int(r.get("frame_end", r["frames"]))
    _set(sc.eevee, "taa_render_samples", int(r.get("samples", 64)))
    _set(sc.eevee, "use_raytracing", True)
    _set(sc.eevee, "use_shadows", True)
    # the comb of dots along every silhouette is the body shadow-casting into the smoke shell that touches it;
    # the smoke is lit by the field's own emission anyway, so the shadow map only costs a frame and an artefact
    _set(sc.eevee, "use_volumetric_shadows", bool(r.get("volumetric_shadows", False)))
    _set(sc.eevee, "volumetric_tile_size", r.get("volumetric_tile", "4"))
    _set(sc.eevee, "volumetric_samples", int(r.get("volumetric_samples", 192)))
    v_start, v_end = _volume_range(cfg)
    _set(sc.eevee, "volumetric_start", v_start)
    _set(sc.eevee, "volumetric_end", v_end)
    _set(sc.eevee, "use_volume_custom_range", True)
    _set(sc.render, "use_motion_blur", bool(r.get("motion_blur", True)))
    _set(sc.render, "motion_blur_shutter", 0.45)
    sc.view_settings.view_transform = "AgX"
    _set(sc.view_settings, "look", r.get("look", "AgX - Medium High Contrast"))
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    cfg = json.load(open(argv[0], encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_studio(cfg)
    build_bodies(cfg)
    build_smoke(cfg)
    build_camera(cfg)
    setup_render(cfg)
    out = cfg["out_dir"]
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    only = int(cfg["render"].get("only_frame", 0) or 0)
    if only:
        bpy.context.scene.frame_set(only)
        bpy.context.scene.render.filepath = os.path.join(frames_dir, f"frame_{only:04d}.png")
        bpy.ops.render.render(write_still=True)
        print("STILL", bpy.context.scene.render.filepath, flush=True)
        return
    # the animation writes one file per frame by appending the number to this prefix; left unset it is /tmp and
    # the frames land outside the film's folder without anything failing
    bpy.context.scene.render.filepath = os.path.join(frames_dir, "frame_")
    bpy.ops.render.render(animation=True)
    print("FRAMES_DONE", frames_dir, flush=True)

if __name__ == "__main__":
    main()
