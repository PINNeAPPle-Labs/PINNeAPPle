"""General-relativistic ray tracing of an accretion flow around a Schwarzschild black hole ("Interstellar" view).

What the camera of *Interstellar* showed (James, von Tunzelmann, Franklin & Thorne 2015, Class. Quantum Grav. 32,
065001): light bent by the hole, so the far side of the disc appears both above and below the shadow, a thin photon
ring at the edge of the shadow, and the background stars lensed around it. Here that picture is computed from the
simulated (or forecast) flow:

* null geodesics of the Schwarzschild metric (G = M = c = 1): each ray stays in a plane through the hole, and
  u = 1/r obeys u'' = 3u^2 - u along the azimuth in that plane (RK4); rays with impact parameter b < 3 sqrt(3) fall
  in (the shadow);
* optically thin emission with optional grey absorption (kappa rho), integrated along each ray through the
  axisymmetric (r, theta) fields (bilinear); emissivity rho^q sqrt(T), q = 2 is bremsstrahlung-like, a larger q is
  a visual choice that brings out the dense mid-plane;
* redshift g = sqrt(1 - 2/r) / (gamma (1 - v . k)) from gravity and the gas motion (Doppler beaming, I ~ g^4) and a
  colour temperature that follows g -- the film switched beaming off for the audience; ``doppler=False`` does too;
* escaped rays sample a procedural star field, so the background is lensed as well.

The geodesics do not depend on time, so ``Camera.trace`` runs once and stores, for every ray, the points where it
crosses the flow; ``render`` then only gathers the fields at those points, which makes rendering a time series cheap.

    cam = Camera(width=480, height=270, r_cam=120.0, inclination_deg=84.0, fov_deg=32.0)
    paths = cam.trace(grid, r_max=60.0)
    img = render(paths, rho, temperature, velocity)        # (H, W, 3) float in [0, 1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Camera", "RayPaths", "render", "bloom", "star_field", "blackbody_rgb", "SHADOW_B"]

SHADOW_B = 3.0 * math.sqrt(3.0)          # critical impact parameter of a Schwarzschild hole (M = 1)


@dataclass
class RayPaths:
    """Where each ray crosses the emitting region, in camera-to-far order (flattened over rays)."""
    ray: np.ndarray            # (S,) int32 ray index of each sample
    ir: np.ndarray             # (S,) float32 fractional radial cell-centre coordinate
    ith: np.ndarray            # (S,) float32 fractional theta cell-centre coordinate
    ds: np.ndarray             # (S,) float32 path length
    kdotv: np.ndarray          # (S, 3) float32 photon direction (towards the camera) in (r, theta, phi) unit vectors
    lapse: np.ndarray          # (S,) float32 sqrt(1 - 2/r)
    captured: np.ndarray       # (H*W,) bool
    escape_dir: np.ndarray     # (H*W, 3) float32 final direction of escaped rays
    shape: tuple[int, int]     # (H, W)


@dataclass
class Camera:
    width: int = 480
    height: int = 270
    r_cam: float = 120.0
    inclination_deg: float = 84.0        # angle from the spin axis (90 = edge on)
    fov_deg: float = 32.0                # horizontal field of view
    roll_deg: float = 0.0
    dphi: float = 0.01                   # largest RK4 step in the orbital-plane azimuth
    ds_in: float = 1.0                   # largest path length per step inside the emitting region (GM/c^2)
    ds_out: float = 6.0                  # ... and outside it
    max_phi: float = 4.0 * math.pi

    def basis(self):
        i = math.radians(self.inclination_deg)
        pos = self.r_cam * np.array([math.sin(i), 0.0, math.cos(i)])
        fwd = -pos / np.linalg.norm(pos)
        up0 = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up0)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        if self.roll_deg:
            c, s = math.cos(math.radians(self.roll_deg)), math.sin(math.radians(self.roll_deg))
            right, up = c * right + s * up, -s * right + c * up
        return pos, fwd, right, up

    def directions(self) -> np.ndarray:
        """(H*W, 3) unit directions of the pixels (local static frame at the camera)."""
        _, fwd, right, up = self.basis()
        W, H = self.width, self.height
        tx = math.tan(math.radians(self.fov_deg) / 2)
        xs = (np.arange(W) + 0.5) / W * 2 - 1
        ys = 1 - (np.arange(H) + 0.5) / H * 2
        X, Y = np.meshgrid(xs * tx, ys * tx * H / W)
        d = fwd[None] + X.reshape(-1, 1) * right[None] + Y.reshape(-1, 1) * up[None]
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def trace(self, grid: dict[str, np.ndarray], r_max: float, *, stride: int = 1) -> RayPaths:
        """Integrate every pixel's geodesic; keep the samples with r_in < r < r_max (every ``stride`` steps)."""
        pos, _, _, _ = self.basis()
        d = self.directions()
        n = len(d)
        rc = self.r_cam
        e1 = pos / rc
        dpar = d @ e1
        perp = d - dpar[:, None] * e1[None]
        pn = np.linalg.norm(perp, axis=1)
        tiny = pn < 1e-9
        # rays aimed exactly at the hole: any perpendicular will do (b = 0)
        alt = np.cross(e1, [0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.cross(e1, [1.0, 0.0, 0.0])
        perp[tiny] = alt / np.linalg.norm(alt)
        pn[tiny] = 1.0
        e2 = perp / pn[:, None]
        sin_a = np.clip(np.where(tiny, 0.0, pn), 0, 1)                # angle from the radial direction
        b = rc * sin_a / math.sqrt(1 - 2 / rc)                         # impact parameter
        u = np.full(n, 1.0 / rc)
        inward = dpar < 0
        w2 = np.clip(1 / np.maximum(b, 1e-12) ** 2 - u ** 2 * (1 - 2 * u), 0, None)
        w = np.where(inward, 1.0, -1.0) * np.sqrt(w2)                 # du/dphi
        w[tiny] = 1e6                                                 # straight in: handled by capture below
        phi = np.zeros(n)
        active = np.ones(n, bool)
        captured = np.zeros(n, bool)
        esc_dir = np.zeros((n, 3))
        h = self.dphi
        r_in = float(grid["r_faces"][0])
        lr0 = math.log(grid["r_faces"][0])
        dlr = math.log(grid["r_faces"][-1] / grid["r_faces"][0]) / (len(grid["r_faces"]) - 1)
        nth = len(grid["theta_faces"]) - 1
        dth = math.pi / nth
        out = {k: [] for k in ("ray", "ir", "ith", "ds", "k", "lapse")}
        captured[tiny & inward] = True
        active[tiny] = False

        def f(uu, ww):
            return ww, 3 * uu * uu - uu

        idx = np.nonzero(active)[0]
        uu, ww, pp = u[idx], w[idx], phi[idx]
        a1, a2 = e1[None].repeat(len(idx), 0), e2[idx]
        step = 0
        while len(idx) and step < 200000:
            # step in phi, limited so that no step is longer than ds_in (ds_out) along the ray: near-radial rays
            # cover a lot of path in little azimuth
            speed = np.sqrt(1.0 + (ww / uu) ** 2) / uu                  # |dx/dphi|
            hh = np.minimum(h, np.where(1.0 / uu < r_max * 1.05, self.ds_in, self.ds_out) / speed)
            k1u, k1w = f(uu, ww)
            k2u, k2w = f(uu + 0.5 * hh * k1u, ww + 0.5 * hh * k1w)
            k3u, k3w = f(uu + 0.5 * hh * k2u, ww + 0.5 * hh * k2w)
            k4u, k4w = f(uu + hh * k3u, ww + hh * k3w)
            u_new = uu + hh / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            w_new = ww + hh / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
            pp = pp + hh
            step += 1
            cap = u_new >= 0.5
            esc = (u_new <= 0) | ((u_new < 1.0 / (1.5 * rc)) & (w_new < 0)) | (pp > self.max_phi)
            r = 1.0 / np.maximum(u_new, 1e-12)
            if step % stride == 0:
                inside = (~cap) & (~esc) & (r < r_max) & (r > r_in)
                if inside.any():
                    j = np.nonzero(inside)[0]
                    rr = r[j]
                    c, s = np.cos(pp[j]), np.sin(pp[j])
                    x = rr[:, None] * (c[:, None] * a1[j] + s[:, None] * a2[j])
                    drdphi = -w_new[j] / u_new[j] ** 2
                    tang = drdphi[:, None] * (c[:, None] * a1[j] + s[:, None] * a2[j]) + \
                        rr[:, None] * (-s[:, None] * a1[j] + c[:, None] * a2[j])
                    dsj = np.linalg.norm(tang, axis=1) * hh[j] * stride
                    kph = -tang / np.linalg.norm(tang, axis=1, keepdims=True)       # towards the camera
                    th = np.arccos(np.clip(x[:, 2] / rr, -1, 1))
                    R = np.hypot(x[:, 0], x[:, 1])
                    rhat = x / rr[:, None]
                    phat = np.stack([-x[:, 1], x[:, 0], np.zeros_like(R)], 1) / np.maximum(R, 1e-9)[:, None]
                    thhat = np.cross(phat, rhat)
                    out["ray"].append(idx[j].astype(np.int32))
                    out["ir"].append(np.clip((np.log(rr) - lr0) / dlr - 0.5, 0, len(grid["r_faces"]) - 2).astype(np.float32))
                    out["ith"].append(np.clip(th / dth - 0.5, 0, nth - 1).astype(np.float32))
                    out["ds"].append(dsj.astype(np.float32))
                    out["k"].append(np.stack([(kph * rhat).sum(1), (kph * thhat).sum(1), (kph * phat).sum(1)],
                                             1).astype(np.float32))
                    out["lapse"].append(np.sqrt(1 - 2 / rr).astype(np.float32))
            done = cap | esc
            if done.any():
                jd = np.nonzero(done)[0]
                captured[idx[jd[cap[jd]]]] = True
                je = jd[esc[jd]]
                if len(je):
                    c, s = np.cos(pp[je]), np.sin(pp[je])
                    # asymptotic direction: the tangent of the outgoing ray
                    uj = np.maximum(u_new[je], 1e-12)
                    tg = (-w_new[je] / uj ** 2)[:, None] * (c[:, None] * a1[je] + s[:, None] * a2[je]) + \
                        (1 / uj)[:, None] * (-s[:, None] * a1[je] + c[:, None] * a2[je])
                    esc_dir[idx[je]] = tg / np.linalg.norm(tg, axis=1, keepdims=True)
                keep = ~done
                idx, uu, ww, pp, a1, a2 = idx[keep], u_new[keep], w_new[keep], pp[keep], a1[keep], a2[keep]
            else:
                uu, ww = u_new, w_new
        def cat(k, dt_):
            return np.concatenate(out[k]).astype(dt_) if out[k] else np.zeros((0,) + ((3,) if k == "k" else ()), dt_)
        ray = cat("ray", np.int32)
        order = np.argsort(ray, kind="stable")                       # ray-major, camera-to-far within a ray
        return RayPaths(ray[order], cat("ir", np.float32)[order], cat("ith", np.float32)[order], cat("ds", np.float32)[order],
                        cat("k", np.float32)[order], cat("lapse", np.float32)[order], captured,
                        esc_dir.astype(np.float32), (self.height, self.width))


# ---------------------------------------------------------------------- colour
def blackbody_rgb(T: np.ndarray) -> np.ndarray:
    """Approximate sRGB colour (0..1) of a blackbody at temperature T [K] (Tanner Helland fit), (..., 3)."""
    t = np.clip(np.asarray(T, float), 1000, 40000) / 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return _bb(t)


def _bb(t):
    r = np.where(t <= 66, 255.0, 329.698727446 * (t - 60) ** -0.1332047592)
    g = np.where(t <= 66, 99.4708025861 * np.log(t) - 161.1195681661, 288.1221695283 * (t - 60) ** -0.0755148492)
    b = np.where(t >= 66, 255.0, np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.maximum(t - 10, 1e-9)) - 305.0447927307))
    return np.clip(np.stack([r, g, b], -1) / 255.0, 0, 1)


def star_field(width: int = 4096, height: int = 2048, n: int = 9000, seed: int = 3) -> np.ndarray:
    """Equirectangular star map (H, W, 3) with a faint Milky-Way-like band, deterministic."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.float32)
    lon = np.linspace(-math.pi, math.pi, width, endpoint=False)
    lat = np.linspace(math.pi / 2, -math.pi / 2, height)
    LON, LAT = np.meshgrid(lon, lat)
    # band along a tilted great circle
    tilt = math.radians(28)
    z = np.sin(LAT) * math.cos(tilt) - np.cos(LAT) * np.sin(LON) * math.sin(tilt)
    noise = rng.normal(size=(height // 16, width // 16))
    from scipy.ndimage import gaussian_filter, zoom
    noise = zoom(gaussian_filter(noise, 2.0), (height / noise.shape[0], width / noise.shape[1]), order=1)
    band = np.exp(-(z / 0.16) ** 2) * (0.5 + 0.5 * np.tanh(noise))
    img += (0.05 * band)[..., None] * np.array([0.85, 0.82, 1.0])
    # stars, area-uniform on the sphere
    sl = np.arcsin(rng.uniform(-1, 1, n))
    sL = rng.uniform(-math.pi, math.pi, n)
    mag = rng.pareto(1.6, n) * 0.25 + 0.15
    temp = rng.choice([3500, 4500, 5800, 7500, 10000, 15000], n, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
    col = blackbody_rgb(temp)
    yy = ((math.pi / 2 - sl) / math.pi * (height - 1)).astype(int)
    xx = ((sL + math.pi) / (2 * math.pi) * width).astype(int) % width
    for dy, dx, wgt in [(0, 0, 1.0), (0, 1, 0.35), (0, -1, 0.35), (1, 0, 0.35), (-1, 0, 0.35)]:
        np.add.at(img, ((yy + dy) % height, (xx + dx) % width), (mag * wgt)[:, None] * col)
    return np.clip(img, 0, 1)


def _sample_sky(sky: np.ndarray, d: np.ndarray) -> np.ndarray:
    H, W = sky.shape[:2]
    lon = np.arctan2(d[:, 1], d[:, 0])
    lat = np.arcsin(np.clip(d[:, 2], -1, 1))
    x = ((lon + math.pi) / (2 * math.pi) * W).astype(int) % W
    y = np.clip(((math.pi / 2 - lat) / math.pi * (H - 1)).astype(int), 0, H - 1)
    return sky[y, x]


# ---------------------------------------------------------------------- rendering
def _bilinear(field: np.ndarray, fr: np.ndarray, ft: np.ndarray) -> np.ndarray:
    i0 = np.minimum(fr.astype(np.int64), field.shape[0] - 2)
    j0 = np.minimum(ft.astype(np.int64), field.shape[1] - 2)
    a, b = fr - i0, ft - j0
    return ((1 - a) * (1 - b) * field[i0, j0] + a * (1 - b) * field[i0 + 1, j0] +
            (1 - a) * b * field[i0, j0 + 1] + a * b * field[i0 + 1, j0 + 1])


def bloom(img: np.ndarray, strength: float = 0.35, sigma: float = 6.0) -> np.ndarray:
    """Cinematic glow: add a blurred copy of the bright parts."""
    from scipy.ndimage import gaussian_filter
    bright = np.clip(img - 0.55, 0, None)
    return np.clip(img + strength * gaussian_filter(bright, (sigma, sigma, 0)) / 0.45, 0, 1)


def render(paths: RayPaths, rho: np.ndarray, temperature: np.ndarray, velocity: np.ndarray | None = None, *,
           doppler: bool = True, kappa: float = 0.0, exposure: float = 1.0, scale: float | None = None,
           emissivity_power: float = 2.0, tone: str = "film", r_min: float = 0.0,
           t_color: float = 6500.0, gamma: float = 1.0, sky: np.ndarray | None = None, sky_gain: float = 1.0,
           return_intensity: bool = False):
    """Render one frame.

    rho, temperature: (nr, ntheta) fields (temperature ~ p / rho); velocity: (3, nr, ntheta) = (v_r, v_theta, v_phi)
    in units of c, or None (no Doppler). ``scale``: intensity mapped to white (pass the same value for every frame
    of an animation; default: the 99.7th percentile of this frame). Returns (H, W, 3) floats in [0, 1].
    """
    H, W = paths.shape
    n = H * W
    fr, ft = paths.ir.astype(np.float64), paths.ith.astype(np.float64)
    rr = np.maximum(_bilinear(rho, fr, ft), 0.0)
    TT = np.maximum(_bilinear(temperature, fr, ft), 1e-12)
    j = rr ** emissivity_power * np.sqrt(TT)
    if r_min > 0:                                              # optional: leave out the plunging region
        j = j * (paths.lapse >= math.sqrt(1 - 2 / r_min))
    g = paths.lapse.astype(np.float64)
    if doppler and velocity is not None:
        v = np.stack([_bilinear(velocity[k], fr, ft) for k in range(3)], 1)
        v2 = np.clip((v * v).sum(1), 0, 0.98)
        vk = (v * paths.kdotv).sum(1)
        g = g * np.sqrt(1 - v2) / (1 - vk)
    contrib = j * g ** 4 * paths.ds
    if kappa > 0:                                              # absorption, accumulated from the camera side
        dtau = kappa * rr * paths.ds
        cs = np.cumsum(dtau)
        first = np.r_[0, np.nonzero(np.diff(paths.ray))[0] + 1] if len(paths.ray) else np.zeros(0, np.int64)
        start_cs = np.zeros(len(paths.ray))
        if len(paths.ray):
            seg_start = np.repeat(first, np.diff(np.r_[first, len(paths.ray)]))
            start_cs = np.where(seg_start > 0, cs[seg_start - 1], 0.0)
        tau = cs - dtau - start_cs
        contrib = contrib * np.exp(-tau)
    inten = np.bincount(paths.ray, weights=contrib, minlength=n)
    gw = np.bincount(paths.ray, weights=contrib * g, minlength=n) / np.maximum(inten, 1e-300)
    if return_intensity:
        return inten.reshape(H, W)
    s = scale if scale is not None else (np.percentile(inten[inten > 0], 99.7) if (inten > 0).any() else 1.0)
    x = exposure * (inten / max(s, 1e-300)) ** gamma
    lum = 1 - np.exp(-x) if tone == "film" else np.log1p(30 * x) / np.log1p(30)   # "log": shows faint structure
    col = blackbody_rgb(t_color * np.where(inten > 0, gw, 1.0))
    img = lum[:, None] * col
    if sky is not None:
        escaped = (~paths.captured) & (np.linalg.norm(paths.escape_dir, axis=1) > 0)
        bg = np.zeros((n, 3))
        bg[escaped] = _sample_sky(sky, paths.escape_dir[escaped].astype(np.float64)) * sky_gain
        img = img + bg * (1 - lum[:, None])
    return np.clip(img.reshape(H, W, 3), 0, 1)
