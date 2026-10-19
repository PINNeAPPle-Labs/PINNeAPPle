"""Software renderer for thumbnails, the contact sheet and the turntable GIF (NumPy + Pillow, no GPU, no Blender).

Deferred rasteriser: every triangle is scan-converted into a z-buffer that keeps the triangle id and barycentric
coordinates per pixel, then all pixels are shaded at once (perspective-correct normals with sharp creases,
physically-flavoured PBR: Lambert + Blinn-Phong with roughness, metallic reflections of a studio sky, Fresnel rim,
emissive), a shadow map from the key light, a soft contact shadow on a ground disc, screen-space ambient occlusion
and 2x supersampling. Deterministic: the same geometry renders the same image.

For photoreal renders use Blender through ``pinneapple_tools.visualization.studio.render(geom.scene(), ...)``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .meshops import crease_normals


def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, (0, 1, 0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f])          # rows: camera x, y, z (camera looks along -z)


def rasterize(sx, sy, depth, F, W, H, persp=True, max_frag=3_000_000):
    """Scan-convert triangles. ``sx, sy``: pixel coordinates of the vertices, ``depth``: positive distance.
    Returns (zbuf, tri id (-1 = empty), b0, b1) of shape (H, W)."""
    zbuf = np.full(H * W, np.inf)
    ibuf = np.full(H * W, -1, np.int64)
    b0buf = np.zeros(H * W)
    b1buf = np.zeros(H * W)
    X, Y, Z = sx[F], sy[F], depth[F]
    ok = np.all(Z > 1e-9, 1) & np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1)
    i0 = np.ceil(X.min(1) - 0.5).astype(np.int64)
    i1 = np.floor(X.max(1) - 0.5).astype(np.int64)
    j0 = np.ceil(Y.min(1) - 0.5).astype(np.int64)
    j1 = np.floor(Y.max(1) - 0.5).astype(np.int64)
    i0, j0 = np.maximum(i0, 0), np.maximum(j0, 0)
    i1, j1 = np.minimum(i1, W - 1), np.minimum(j1, H - 1)
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    ok &= (i1 >= i0) & (j1 >= j0) & (np.abs(area) > 1e-12)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return zbuf.reshape(H, W), ibuf.reshape(H, W), b0buf.reshape(H, W), b1buf.reshape(H, W)
    bw = 1 << np.ceil(np.log2(i1[idx] - i0[idx] + 1)).astype(np.int64)
    bh = 1 << np.ceil(np.log2(j1[idx] - j0[idx] + 1)).astype(np.int64)
    key = bw * 65536 + bh
    order = np.argsort(key, kind="stable")
    idx, key = idx[order], key[order]
    starts = np.r_[0, np.where(np.diff(key))[0] + 1, len(key)]
    invd = 1.0 / Z if persp else Z
    for s, e in zip(starts[:-1], starts[1:], strict=False):
        w, h = int(key[s] // 65536), int(key[s] % 65536)
        ox, oy = np.meshgrid(np.arange(w), np.arange(h))
        ox, oy = ox.ravel(), oy.ravel()
        step = max(1, max_frag // (w * h))
        for c in range(s, e, step):
            t = idx[c:min(c + step, e)]
            px = i0[t, None] + ox[None]
            py = j0[t, None] + oy[None]
            inside = (px <= i1[t, None]) & (py <= j1[t, None])
            cx, cy = px + 0.5, py + 0.5
            x0, y0 = X[t, 0, None], Y[t, 0, None]
            x1, y1 = X[t, 1, None], Y[t, 1, None]
            x2, y2 = X[t, 2, None], Y[t, 2, None]
            a = area[t, None]
            l0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / a
            l1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / a
            l2 = 1.0 - l0 - l1
            m = inside & (l0 >= -1e-7) & (l1 >= -1e-7) & (l2 >= -1e-7)
            if not m.any():
                continue
            rows, cols = np.nonzero(m)
            tri = t[rows]
            L0, L1, L2 = l0[rows, cols], l1[rows, cols], l2[rows, cols]
            q = L0 * invd[tri, 0] + L1 * invd[tri, 1] + L2 * invd[tri, 2]
            d = 1.0 / q if persp else q
            pix = py[rows, cols] * W + px[rows, cols]
            np.minimum.at(zbuf, pix, d)
            win = d <= zbuf[pix]
            ibuf[pix[win]] = tri[win]
            b0buf[pix[win]] = L0[win]
            b1buf[pix[win]] = L1[win]
    return zbuf.reshape(H, W), ibuf.reshape(H, W), b0buf.reshape(H, W), b1buf.reshape(H, W)


def _blur(a, r):
    """Separable box blur (radius r pixels), edge-clamped."""
    if r < 1:
        return a
    k = 2 * r + 1
    for ax in (0, 1):
        p = np.pad(a, [(r, r) if i == ax else (0, 0) for i in range(2)], mode="edge")
        c = np.cumsum(p, axis=ax)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=ax)), c], axis=ax)
        a = (np.take(c, np.arange(k, c.shape[ax]), axis=ax) - np.take(c, np.arange(0, c.shape[ax] - k), axis=ax)) / k
    return a


def _materials(geom) -> dict[str, dict]:
    from pinneapple_tools.visualization.studio.scene import MATERIALS
    return {**MATERIALS, **geom.spec.material_table()}


def _gather(geom):
    """Vertices, faces, per-face material properties (linear base colour, metallic, roughness, emissive, alpha)."""
    mats = _materials(geom)
    Vs, Fs, cols, met, rough, emis, cnorm = [], [], [], [], [], [], []
    off = 0
    for b in geom.bodies:
        fm = b.face_materials()
        Vs.append(b.vertices)
        Fs.append(b.faces + off)
        off += len(b.vertices)
        cnorm.append(crease_normals(b.vertices, b.faces))
        uniq = list(dict.fromkeys(fm))
        lut = {u: mats.get(u, {"color": [0.6, 0.62, 0.65], "metallic": 0.0, "roughness": 0.5}) for u in uniq}
        cols.append(np.array([_srgb_to_lin(lut[u]["color"]) for u in fm]).reshape(-1, 3))
        met.append(np.array([lut[u].get("metallic", 0.0) for u in fm], float))
        rough.append(np.array([lut[u].get("roughness", 0.5) for u in fm], float))
        emis.append(np.array([_srgb_to_lin(lut[u].get("emissive", [0, 0, 0])) for u in fm]).reshape(-1, 3))
    return (np.vstack(Vs), np.vstack(Fs), np.vstack(cnorm), np.vstack(cols), np.concatenate(met),
            np.concatenate(rough), np.vstack(emis))


def _sky(d):
    """Studio environment radiance along unit directions d (z up): bright soft box overhead, warm horizon, grey floor."""
    z = d[..., 2]
    zen = np.array([0.60, 0.68, 0.80])
    hor = np.array([0.95, 0.93, 0.90])
    gnd = np.array([0.22, 0.21, 0.20])
    t = np.clip(z, 0, 1)[..., None] ** 0.6
    up = hor * (1 - t) + zen * t
    soft = np.exp(-((d[..., 0] + 0.35) ** 2 + (d[..., 1] - 0.25) ** 2) / 0.25)[..., None] * (z > 0)[..., None] * 1.8
    down = gnd + (hor - gnd) * np.clip(1 + z, 0, 1)[..., None] ** 6
    return np.where((z >= 0)[..., None], up + soft, down)


def render_geometry(geom, path: str | None = None, size: tuple[int, int] = (960, 640), azimuth: float | None = None,
                    elevation: float | None = None, ssaa: int = 2, background: str = "studio", fov_deg: float = 28.0,
                    ground: bool | None = None, margin: float = 0.90, label: str | None = None,
                    frame: tuple[Sequence[float], Sequence[float]] | None = None):
    """Render ``geom`` (a library :class:`Geometry`) and return a PIL image (also saved to ``path`` if given).

    ``azimuth`` / ``elevation`` (degrees) default to the entry's camera; azimuth 0 looks from +x (behind for
    vehicles, whose x points aft), measured toward +y. ``frame``: (lo, hi) corners of a box to frame instead of
    the geometry's own bounds, so the frames of an animation share one camera."""
    from PIL import Image, ImageDraw

    cam = geom.spec.camera
    az = math.radians(cam.get("azimuth", 215.0) if azimuth is None else azimuth)
    el = math.radians(cam.get("elevation", 18.0) if elevation is None else elevation)
    W, H = size[0] * ssaa, size[1] * ssaa
    V, F, CN, base, metal, rough, emis = _gather(geom)
    if frame is None:
        fit = V
    else:
        (x0, y0, z0), (x1, y1, z1) = frame
        fit = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    lo, hi = fit.min(0), fit.max(0)
    ctr = 0.5 * (lo + hi)
    rad = 0.5 * float(np.linalg.norm(hi - lo))
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    dist = rad / math.sin(math.radians(fov_deg) / 2) * 1.05
    eye = ctr + d * dist
    R = _look_at(eye, ctr)
    Pc = (V - eye) @ R.T
    depth = -Pc[:, 2]
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    nx, ny = f * Pc[:, 0] / depth, f * Pc[:, 1] / depth
    asp = W / H
    # fit: zoom and centre so the projected model (or the given frame box) fills `margin` of the frame
    Pf = (fit - eye) @ R.T
    fx, fy = f * Pf[:, 0] / -Pf[:, 2], f * Pf[:, 1] / -Pf[:, 2]
    cx, cy = 0.5 * (fx.min() + fx.max()), 0.5 * (fy.min() + fy.max())
    zoom = margin * min(2 * asp / max(np.ptp(fx), 1e-9), 2 / max(np.ptp(fy), 1e-9))
    sx = ((nx - cx) * zoom / asp * 0.5 + 0.5) * W
    sy = (0.5 - (ny - cy) * zoom * 0.5) * H
    zb, ib, b0, b1 = rasterize(sx, sy, depth, F, W, H, persp=True)
    hit = ib >= 0
    t = ib[hit]
    B0, B1 = b0[hit], b1[hit]
    B2 = 1 - B0 - B1
    zt = depth[F[t]]
    w = np.stack([B0 / zt[:, 0], B1 / zt[:, 1], B2 / zt[:, 2]], 1)
    w /= w.sum(1, keepdims=True)
    P = np.einsum("ik,ikj->ij", w, V[F[t]])
    N = np.einsum("ik,ikj->ij", w, CN[t])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    view = eye - P
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", N, view) < 0
    N[flip] *= -1                                                     # double-sided

    # key light (sun) with a shadow map, fill light opposite, sky ambient
    key_dir = np.array([math.cos(math.radians(40)) * math.cos(az + 0.9), math.cos(math.radians(40)) * math.sin(az + 0.9),
                        math.sin(math.radians(52))])
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-key_dir[0], -key_dir[1], 0.35])
    fill_dir /= np.linalg.norm(fill_dir)
    zg = lo[2] - 1e-4 * rad
    use_ground = (el > 0.02) if ground is None else ground
    # shadow map: orthographic along the key light
    Rl = _look_at(ctr + key_dir * 4 * rad, ctr)
    Pl = (V - (ctr + key_dir * 4 * rad)) @ Rl.T
    SM = 1400
    ext = 1.6 * rad
    lx = (Pl[:, 0] / ext * 0.5 + 0.5) * SM
    ly = (0.5 - Pl[:, 1] / ext * 0.5) * SM
    smz, _, _, _ = rasterize(lx, ly, -Pl[:, 2], F, SM, SM, persp=False)

    def shadow(Pw, bias):
        q = (Pw - (ctr + key_dir * 4 * rad)) @ Rl.T
        u = (q[:, 0] / ext * 0.5 + 0.5) * SM
        v = (0.5 - q[:, 1] / ext * 0.5) * SM
        dd = -q[:, 2]
        lit = np.zeros(len(Pw))
        for ox in (-1.0, 0.0, 1.0):                                  # 3x3 percentage-closer filter
            for oy in (-1.0, 0.0, 1.0):
                ui = np.clip((u + ox * 1.2).astype(np.int64), 0, SM - 1)
                vi = np.clip((v + oy * 1.2).astype(np.int64), 0, SM - 1)
                lit += (dd <= smz[vi, ui] + bias)
        return lit / 9.0

    bias = 0.004 * rad
    sh = shadow(P + N * 0.002 * rad, bias)
    alb, mt, rg, em = base[t], metal[t][:, None], np.clip(rough[t], 0.04, 1.0)[:, None], emis[t]
    ndl = np.clip(np.einsum("ij,j->i", N, key_dir), 0, None)[:, None]
    ndf = np.clip(np.einsum("ij,j->i", N, fill_dir), 0, None)[:, None]
    hvec = key_dir[None] + view
    hvec /= np.linalg.norm(hvec, axis=1, keepdims=True)
    ndh = np.clip(np.einsum("ij,ij->i", N, hvec), 0, None)[:, None]
    ndv = np.clip(np.einsum("ij,ij->i", N, view), 0, 1)[:, None]
    shin = 2.0 / (rg ** 4) + 2.0
    f0 = 0.04 * (1 - mt) + alb * mt
    fres = f0 + (1 - f0) * (1 - ndv) ** 5
    spec = fres * (shin + 8) / (8 * math.pi) * ndh ** shin * ndl
    refl = view * -1
    refl = refl - 2 * np.einsum("ij,ij->i", refl, N)[:, None] * N
    env = _sky(refl)
    hemi = 0.5 * (1 + N[:, 2:3])
    amb = np.array([0.30, 0.31, 0.33]) * (1 - hemi) * 0.55 + np.array([0.62, 0.66, 0.72]) * hemi
    diff = alb * (1 - mt) * (1.55 * ndl * sh[:, None] + 0.30 * ndf + 0.50 * amb)
    gloss = (1 - rg) ** 2
    col = diff + 2.0 * spec * sh[:, None] + fres * env * (0.25 + 0.75 * gloss) * (0.35 + 0.65 * mt) + em * 1.4
    # screen-space ambient occlusion from the depth buffer
    zfill = np.where(hit, zb, np.nan)
    zz = np.where(hit, zb, np.nanmax(zfill) if hit.any() else 0)
    r = max(2, int(0.012 * max(W, H)))
    occ = np.clip((zz - _blur(zz, r)) / (0.03 * rad), 0, 1)
    occ2 = np.clip((zz - _blur(zz, 3 * r)) / (0.08 * rad), 0, 1)
    ao = 1 - 0.45 * occ[hit] - 0.25 * occ2[hit]
    col *= ao[:, None]

    # background and ground
    yy, xx = np.mgrid[0:H, 0:W]
    g = yy / H
    if background == "studio":
        bg = (np.array([0.93, 0.94, 0.95])[None, None] * (1 - g[..., None]) +
              np.array([0.72, 0.74, 0.77])[None, None] * g[..., None])
        bg = _srgb_to_lin(bg)
    else:
        bg = np.zeros((H, W, 3))
    img = bg.copy()
    if use_ground:
        # rays through every background pixel, intersected with the ground plane z = zg
        px_nx = ((xx + 0.5) / W - 0.5) * 2 * asp / zoom + cx
        px_ny = (0.5 - (yy + 0.5) / H) * 2 / zoom + cy
        dirc = np.stack([px_nx / f, px_ny / f, -np.ones_like(px_nx)], -1)
        dw = dirc @ R
        dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
        tt = (zg - eye[2]) / np.where(np.abs(dw[..., 2]) > 1e-9, dw[..., 2], -1e-9)
        gp = eye + tt[..., None] * dw
        rr = np.hypot(gp[..., 0] - ctr[0], gp[..., 1] - ctr[1]) / (1.25 * rad)
        gm = (tt > 0) & (~hit) & (rr < 1.6)
        if gm.any():
            gpts = gp[gm]
            lit = shadow(gpts, bias)
            # contact darkening: distance to the model footprint in the shadow-map sense (soft)
            fade = np.clip(1 - rr[gm] / 1.6, 0, 1) ** 1.5
            gcol = _srgb_to_lin(np.array([0.80, 0.81, 0.83]))
            shade = (0.55 + 0.45 * lit) * 1.0
            gc = gcol[None] * shade[:, None]
            img[gm] = img[gm] * (1 - fade[:, None]) + gc * fade[:, None]
    img[hit] = col
    # tone map (soft shoulder), sRGB, downsample
    img = img / (1 + 0.30 * img)
    img = (_lin_to_srgb(img * 0.98) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(img, "RGB")
    if ssaa > 1:
        im = im.resize(size, Image.LANCZOS)
    if label:
        dr = ImageDraw.Draw(im)
        font = _font(max(12, size[1] // 22))
        dr.text((int(0.03 * size[0]), int(0.03 * size[1])), label, fill=(40, 44, 52), font=font)
    if path:
        im.save(path, optimize=True)
    return im


def _font(px: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def contact_sheet(geoms: Sequence, path: str, cols: int = 4, tile: tuple[int, int] = (480, 330), title: str = "") -> str:
    """Grid of thumbnails with names (PNG)."""
    from PIL import Image, ImageDraw
    rows = math.ceil(len(geoms) / cols)
    head = 70 if title else 0
    sheet = Image.new("RGB", (cols * tile[0], rows * tile[1] + head), (236, 238, 241))
    dr = ImageDraw.Draw(sheet)
    if title:
        dr.text((20, 18), title, fill=(30, 34, 40), font=_font(34))
    for k, g in enumerate(geoms):
        im = g.render(None, size=tile, label=g.spec.title)
        sheet.paste(im, ((k % cols) * tile[0], head + (k // cols) * tile[1]))
    sheet.save(path, optimize=True)
    return path


def turntable(geoms: Sequence, path: str, frames: int = 24, cols: int = 4, tile: tuple[int, int] = (240, 165),
              duration_ms: int = 90) -> str:
    """Animated GIF: every geometry spinning once, in a grid."""
    from PIL import Image
    rows = math.ceil(len(geoms) / cols)
    out = []
    for fr in range(frames):
        sheet = Image.new("RGB", (cols * tile[0], rows * tile[1]), (236, 238, 241))
        for k, g in enumerate(geoms):
            az0 = g.spec.camera.get("azimuth", 215.0)
            im = g.render(None, size=tile, azimuth=az0 + 360.0 * fr / frames, ssaa=1)
            sheet.paste(im, ((k % cols) * tile[0], (k // cols) * tile[1]))
        out.append(sheet.convert("P", palette=Image.ADAPTIVE, colors=192))
    out[0].save(path, save_all=True, append_images=out[1:], duration=duration_ms, loop=0, optimize=True)
    return path


__all__ = ["render_geometry", "contact_sheet", "turntable", "rasterize"]
