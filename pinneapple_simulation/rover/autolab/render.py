"""Blender Cycles renders of a rover mission (``pip install bpy``): the VIPER model of the repository
(``data/geometries/viper_rover/viper_rover.light.glb``) on the site, the terrain with the ruts the multibody runs
left, coloured by the learned traverse risk or by soil, and both routes.

* :func:`render_overview` -- the whole site from above at an angle, terrain coloured by the learned risk blended over
  the soil albedo, the failed and the replanned routes as tubes, the rover at the stall point and at the goal.
* :func:`render_hero` -- close-up at the stall: light studio background, hard Sun shadow, granular sand thrown into
  the berms of the ruts (instanced grains placed on the berms the terrain model computed; visual only).
* :func:`render_traverse` -- frames of both runs with a chase camera, ruts growing as the rover drives, encoded to
  MP4 and GIF.

Poses come from the multibody runs: the chassis rotation and the mean of the four wheel contact points.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

VIPER_GLB = Path(__file__).resolve().parents[3] / "data" / "geometries" / "viper_rover" / "viper_rover.light.glb"
RESTYLE = {"white_paint": ((0.92, 0.92, 0.90), 0.0, 0.45), "solar_cell": ((0.05, 0.16, 0.48), 0.3, 0.25),
           "aluminium": ((0.80, 0.81, 0.83), 0.9, 0.3)}


def _lin(c):
    return tuple(float(v) ** 2.2 for v in c)


def available() -> bool:
    from pinneapple_blender.cycles import available as ok
    return ok()


def _scene(resolution, samples, background=(0.95, 0.95, 0.94), ambient=0.45, backdrop=(0.97, 0.97, 0.96)):
    """Cycles scene whose world lights the scene at ``ambient`` while the camera sees a flat ``backdrop``."""
    from pinneapple_blender.cycles import CyclesScene
    sc = CyclesScene(resolution=resolution, samples=samples, background=_lin(background), ambient=ambient)
    nt = sc.scene.world.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
    light_bg = nt.nodes["Background"]
    cam_bg = nt.nodes.new("ShaderNodeBackground")
    cam_bg.inputs["Color"].default_value = (*_lin(backdrop), 1.0)
    cam_bg.inputs["Strength"].default_value = 1.0
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(light_bg.outputs[0], mix.inputs[1])
    nt.links.new(cam_bg.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    sc.scene.view_settings.view_transform = "Standard"
    return sc


def import_viper(sc, path=VIPER_GLB):
    """Import the VIPER glb under an empty (origin on the ground under the chassis, x forward, z up)."""
    bpy = sc.bpy
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found: the VIPER glb ships with the repository (data/geometries), not "
                                "with the wheel; pass path= or run from a checkout")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new("viper_root", None)
    sc.scene.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    for m in bpy.data.materials:
        if m.name in RESTYLE and m.use_nodes:
            col, met, rough = RESTYLE[m.name]
            p = next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
            if p is not None:
                for lk in list(p.inputs["Base Color"].links):
                    m.node_tree.links.remove(lk)
                p.inputs["Base Color"].default_value = (*_lin(col), 1.0)
                p.inputs["Metallic"].default_value = met
                p.inputs["Roughness"].default_value = rough
    return root


def pose_at(run, k):
    """(location, rotation matrix) of the rover at output sample ``k``: contact-point centroid and chassis axes."""
    from pinneapple_simulation.multibody.tree import quat_to_matrix
    R = quat_to_matrix(run.q[k, 3:7])
    P = run.wheel["contact_point"][k]
    z = np.clip(run.wheel["sinkage"][k], 0.0, None)
    loc = P.mean(0) - float(np.mean(z)) * R[:, 2]             # contact points lie on the undisturbed surface
    return loc, R


def place(root, run, k):
    from mathutils import Matrix
    loc, R = pose_at(run, k)
    M = Matrix(((R[0, 0], R[0, 1], R[0, 2], loc[0]), (R[1, 0], R[1, 1], R[1, 2], loc[1]),
                (R[2, 0], R[2, 1], R[2, 2], loc[2]), (0, 0, 0, 1)))
    root.matrix_world = M


def terrain_grid(ter, x_range, y_range, step):
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, ter.surface(X, Y)


def _surface_mesh(sc, X, Y, Z, colors, roughness=0.9, name="terrain"):
    """Height grid -> mesh with per-vertex colours (fast bulk assignment)."""
    bpy = sc.bpy
    ny, nx = Z.shape
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    j, i = np.mgrid[0:ny - 1, 0:nx - 1]
    a = (j * nx + i).ravel()
    F = np.column_stack([a, a + 1, a + nx + 1, a + nx]).astype(np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 4, dtype=np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me.update(calc_edges=True)
    col = me.color_attributes.new(name="col", type="FLOAT_COLOR", domain="POINT")
    C = np.ones((len(V), 4), np.float32)
    C[:, :3] = np.asarray(colors, float).reshape(-1, 3) ** 2.2
    col.data.foreach_set("color", C.ravel())
    obj = bpy.data.objects.new(name, me)
    mat = sc.material(roughness=roughness, vertex_colors="col")
    return sc._link(obj, mat)


def _update_heights(obj, Z, X, Y):
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    obj.data.vertices.foreach_set("co", V.ravel())
    obj.data.update()


def terrain_colors(site, X, Y, planner=None, mode="risk", shade_with=None):
    """Soil albedo (``mode="soil"``) or the learned worst-heading risk blended over it (``"risk"``)."""
    U = site.unit_at(X, Y)
    alb = np.array([u.color for u in site.units])[U]
    if mode == "soil" or planner is None:
        return alb
    import matplotlib
    cm = matplotlib.colormaps["YlOrRd"]
    Xr, Yr, R = planner.risk_map(0.25)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((Yr[:, 0], Xr[0]), R, bounds_error=False, fill_value=0.0)
    r = np.clip(f(np.column_stack([Y.ravel(), X.ravel()])).reshape(X.shape), 0.0, 1.0)
    rc = cm(0.35 + 0.6 * r)[..., :3]
    w = (0.85 * np.sqrt(r))[..., None]
    return (1 - w) * (0.7 * alb + 0.3 * 0.9) + w * rc


def _route_tube(sc, site, P, color, lift=0.05, radius=0.06):
    P = np.asarray(P, float)
    z = site.height(P[:, 0], P[:, 1]) + lift
    return sc.add_tube(np.column_stack([P, z]), radius=radius, color=_lin(color), emission=0.4)


def _sun(sc, elevation=38.0, azimuth=-40.0, energy=4.0, angle_deg=0.6):
    o = sc.light("SUN", rotation=(90 - elevation, 0, azimuth + 90), energy=energy)
    o.data.angle = math.radians(angle_deg)
    return o


def build_terrain(site, runs, cell=0.03, upto=None):
    """A fresh deformable terrain with the ruts of ``runs`` replayed (all of each run, or up to ``upto[i]`` s)."""
    ter = site.deformable_terrain(cell=cell)
    for i, r in enumerate(runs):
        r.replay_terrain(ter, float(r.t[-1] + 10.0) if upto is None else upto[i])
    return ter


def _grains(sc, site, ter, x_range, y_range, unit, n=4000, seed=0):
    """Instanced grains on the berms of ruts in ``unit`` (visual only)."""
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.arange(*x_range, ter.cell), np.arange(*y_range, ter.cell))
    B = ter._bilinear(ter.berm_h, X, Y)
    m = (B > 0.002) & (site.unit_at(X, Y) == unit)
    if not m.any():
        return []
    pts = np.column_stack([X[m], Y[m]])
    w = B[m] / B[m].sum()
    k = rng.choice(len(pts), size=min(n, 4 * len(pts)), p=w)
    P = pts[k] + rng.uniform(-0.5, 0.5, (len(k), 2)) * ter.cell
    Z = ter.surface(P[:, 0], P[:, 1])
    objs = []
    for j, r in enumerate((0.003, 0.005, 0.008)):
        sl = slice(j, None, 3)
        Q = np.column_stack([P[sl], Z[sl] + 0.5 * r])
        objs.append(sc.add_points(Q, radius=r, color=_lin((0.80, 0.70, 0.52) if unit else (0.6, 0.6, 0.58))))
    return objs


def _annotate(path, lines):
    from pinneapple_tools.visualization.studio.blender import _hud_overlay
    _hud_overlay(path, list(lines))


def route_lines(runs, routes, learned_planner=None):
    """Caption lines of a render: what each route was planned with and what the multibody run measured."""
    out = ["The rover that learns the planet (multibody VIPER, synthetic site)"]
    for i, (r, run) in enumerate(zip(routes, runs, strict=True)):
        s = run.summary()
        risk = learned_planner.evaluate(r.waypoints).max_risk if learned_planner is not None else r.max_risk
        out.append(f"route {chr(65 + i)} ({'red' if i == 0 else 'green'}, {r.belief_label}): "
                   f"{run.meta['ended'].split(':')[0]} after {s['distance_m']:.1f} m, {s['energy_J']:.0f} J, "
                   f"learned risk {risk:.2f}")
    out.append("terrain: yellow -> red = P(slip > limit) under the learned soil belief")
    return out


def render_overview(site, runs, routes, planner, out_png, *, size=(1600, 900), samples=48, step=0.06, lines=None):
    sc = _scene(size, samples)
    ter = build_terrain(site, runs)
    X, Y, Z = terrain_grid(ter, site.x_range, site.y_range, step)
    _surface_mesh(sc, X, Y, Z, terrain_colors(site, X, Y, planner, "risk"))
    cols = ((0.75, 0.15, 0.12), (0.10, 0.55, 0.25))
    for i, run in enumerate(runs):
        _route_tube(sc, site, run.q[::10, :2], cols[i % 2])
    for k, run in ((len(runs[0].t) - 1, runs[0]), (len(runs[-1].t) - 1, runs[-1])):
        root = import_viper(sc)
        place(root, run, k)
    _sun(sc, elevation=35, azimuth=-60, energy=4.0, angle_deg=1.0)
    c = np.array([21.0, 10.5, 1.2])
    sc.camera(c + np.array([-6.0, -21.0, 15.0]), c, lens=30)
    sc.render(out_png)
    if lines:
        _annotate(out_png, lines)
    return out_png


def render_hero(site, runs, out_png, *, size=(1600, 900), samples=96, window=6.0, cell=0.02, grains=20000,
                lines=None):
    """Close-up of the stall of ``runs[0]`` (studio background, hard shadow, sand grains on the berms)."""
    run = runs[0]
    k = len(run.t) - 1
    loc, R = pose_at(run, k)
    xr = (loc[0] - window * 0.75, loc[0] + window * 0.5)
    yr = (loc[1] - window * 0.45, loc[1] + window * 0.45)
    ter = build_terrain(site, runs[:1], cell=cell)
    sc = _scene(size, samples, background=(0.97, 0.97, 0.96), ambient=0.55)
    X, Y, Z = terrain_grid(ter, (xr[0] - 2, xr[1] + 2), (yr[0] - 2, yr[1] + 2), cell)
    U = site.unit_at(X, Y)
    base = np.where((U == 2)[..., None], np.array([0.86, 0.82, 0.74]), np.array([0.80, 0.80, 0.79]))
    rut = np.clip(ter.rut_depth(X, Y) / 0.08, 0, 1)[..., None]
    _surface_mesh(sc, X, Y, Z, base * (1 - 0.18 * rut), roughness=0.95)
    sand = [i for i in site.mislabelled_units()] or [2]
    _grains(sc, site, ter, xr, yr, sand[0], n=grains)
    root = import_viper(sc)
    place(root, run, k)
    _sun(sc, elevation=30, azimuth=-150, energy=5.0, angle_deg=0.4)
    fwd = np.array([R[0, 0], R[1, 0], 0.0])
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0], 0.0])
    cam = loc - 5.8 * fwd - 3.4 * left + np.array([0, 0, 2.3])
    sc.camera(cam, loc + 0.4 * fwd + np.array([0, 0, 0.45]), lens=38)
    sc.render(out_png)
    if lines:
        _annotate(out_png, lines)
    return out_png


def render_traverse(site, runs, out_dir, *, size=(800, 450), samples=16, every_s=2.5, step=0.06, fps=12,
                    gif_width=480, max_frames=None, progress=None, labels=None):
    """Frames of ``runs`` one after the other (chase camera, ruts growing), then ``traverse.mp4`` and
    ``traverse.gif``. Returns (frame paths, mp4, gif)."""
    from pinneapple_tools.visualization.studio.blender import _encode
    os.makedirs(out_dir, exist_ok=True)
    sc = _scene(size, samples)
    ter = site.deformable_terrain(cell=0.03)
    X, Y, Z0 = terrain_grid(ter, site.x_range, site.y_range, step)
    obj = _surface_mesh(sc, X, Y, Z0, terrain_colors(site, X, Y, None, "soil"))
    cols = ((0.75, 0.15, 0.12), (0.10, 0.55, 0.25))
    tubes = [_route_tube(sc, site, run.q[::10, :2], cols[i % 2], radius=0.035) for i, run in enumerate(runs)]
    from .verify import monitor_slip
    front = [i for i, c in enumerate(runs[0].model.spec.corners) if c.front > 0]
    mon = [monitor_slip(r, front) for r in runs]
    labels = labels or [f"route {chr(65 + i)}" for i in range(len(runs))]
    root = import_viper(sc)
    _sun(sc, elevation=35, azimuth=-60, energy=4.0, angle_deg=1.0)
    cam = None
    frames = []
    starts = [0] * len(runs)
    plan = [(i, k) for i, run in enumerate(runs)
            for k in np.unique(np.r_[np.arange(0, len(run.t), max(1, int(round(every_s / 0.05)))), len(run.t) - 1])]
    if max_frames:
        plan = plan[:: max(1, int(math.ceil(len(plan) / max_frames)))]
    for n, (i, k) in enumerate(plan):
        run = runs[i]
        for j, tb in enumerate(tubes):
            tb.hide_render = j > i
        starts[i] = run.replay_terrain(ter, float(run.t[k]), start=starts[i])
        _update_heights(obj, ter.surface(X, Y), X, Y)
        place(root, run, k)
        loc, R = pose_at(run, k)
        yaw = math.atan2(R[1, 0], R[0, 0])
        eye = loc + np.array([-6.0 * math.cos(yaw) + 3.0 * math.sin(yaw), -6.0 * math.sin(yaw) - 3.0 * math.cos(yaw),
                              3.6])
        if cam is not None:
            sc.bpy.data.objects.remove(cam)
        cam = sc.camera(eye, loc + np.array([0, 0, 0.6]), lens=30)
        frames.append(sc.render(os.path.join(out_dir, f"frame_{n:04d}.png")))
        ms = mon[i][k]
        _annotate(frames[-1], [f"{labels[i]}   t = {run.t[k]:5.1f} s   {run.distance()[k]:4.1f} m",
                               f"slip (2-s mean, front) {'  -  ' if not np.isfinite(ms) else f'{ms:5.2f}'}   "
                               f"sinkage {1e3 * run.wheel['sinkage'][k].max():4.0f} mm   "
                               f"drive energy {run.energy()[k]:5.0f} J"]
                  + ([run.meta["ended"].split(":")[0]] if k == len(run.t) - 1 else []))
        if progress:
            progress(n, len(plan))
    mp4 = _encode(frames, os.path.join(out_dir, "..", "traverse.mp4"), fps=fps)
    gif = _encode(frames, os.path.join(out_dir, "..", "traverse.gif"), fps=fps, gif=True, gif_width=gif_width)
    return frames, os.path.abspath(mp4), os.path.abspath(gif)


__all__ = ["available", "import_viper", "place", "pose_at", "render_overview", "render_hero", "render_traverse",
           "build_terrain", "terrain_colors", "VIPER_GLB"]
