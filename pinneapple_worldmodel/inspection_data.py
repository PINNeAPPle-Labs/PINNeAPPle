"""Synthetic thermal-inspection data with labels from physics, scene variations written as OpenUSD stages, and the
check that an augmentation kept the label true.

``thermal_plate(defect, n)``
    a heated plate (uniform volumetric heat, edges at ambient) with a subsurface void (a disk of low conductivity):
    steady conduction by PINNeAPPle's finite-volume solver. The label is physical: where the void makes the surface
    hotter than the same plate without it (the argmax of the temperature difference).
``Variation`` / ``sample_variation(rng)``
    what the synthetic camera changes: emissivity gain and reflected-ambient offset, sensor noise and blur, camera
    zoom, rotation and offset, and optionally a specular glint (a reflection that looks like a hot spot).
``write_variation_stage(path, T, var)``
    the plate as a USD stage: temperature primvar, displayColor of the thermal camera, a ``UsdGeom.Camera`` with the
    variation's pose, the variation parameters and the physical label as attributes.
``thermal_image(stage, var, H, W)``
    the camera image: the stage rasterized from the top (``rasterize_topdown``), then the variation's camera
    transform, gain, offset, blur and noise.
``label_check(image, var, label)``
    the physics consistency of a sample: locate the hot spot in the image (the brightest blob after smoothing), map it
    back through the known camera transform, and compare with the physical label. A variation that moves the apparent
    hot spot (a glint, saturation) fails and is dropped from the dataset.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["thermal_plate", "Variation", "sample_variation", "write_variation_stage", "thermal_image", "label_check",
           "camera_transform"]


def thermal_plate(defect: tuple[float, float, float], n: int = 48, *, k: float = 1.0, k_void: float = 0.15,
                  q: float = 1.0) -> dict:
    """Steady conduction on the unit square with a low-conductivity disk ``defect = (x, y, r)``. Returns the field,
    the defect-free field, the coefficient and the label (the cell where the void raises the temperature most)."""
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d
    c = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    a = np.full((n, n), k)
    a[(X - defect[0]) ** 2 + (Y - defect[1]) ** 2 <= defect[2] ** 2] = k_void
    T = darcy_2d(a, q)
    T0 = darcy_2d(np.full((n, n), k), q)
    d = T - T0
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return {"T": T, "T0": T0, "a": a, "x": c, "label_xy": (float(c[i]), float(c[j])), "defect": defect,
            "rise": float(d.max() / T0.max())}


@dataclass
class Variation:
    gain: float = 1.0          # emissivity x camera gain
    offset: float = 0.0        # reflected ambient
    noise: float = 0.0
    blur: float = 0.0          # px
    zoom: float = 1.0
    angle_deg: float = 0.0
    shift: tuple = (0.0, 0.0)  # px
    glint: tuple | None = None  # (u, v, radius, strength) in the plate's unit square, a specular reflection

    def to_dict(self) -> dict:
        return asdict(self)


def sample_variation(rng, *, glint_p: float = 0.2) -> Variation:
    return Variation(gain=float(rng.uniform(0.7, 1.2)), offset=float(rng.uniform(0.0, 0.15)),
                     noise=float(rng.uniform(0.0, 0.03)), blur=float(rng.uniform(0.0, 1.2)),
                     zoom=float(rng.uniform(0.85, 1.1)), angle_deg=float(rng.uniform(-20, 20)),
                     shift=(float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4))),
                     glint=(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.04, 0.08)),
                            float(rng.uniform(0.6, 1.0))) if rng.random() < glint_p else None)


def camera_transform(var: Variation, H: int, W: int):
    """Plate unit square (u, v) -> image pixel (col, row) and its inverse, for the variation's camera."""
    th = np.radians(var.angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s = var.zoom * min(H, W) * 0.9
    c = np.array([W / 2 + var.shift[0], H / 2 + var.shift[1]])

    def fwd(uv):
        p = (np.atleast_2d(uv) - 0.5) * [1, -1]
        return p @ R.T * s + c

    def inv(px):
        p = (np.atleast_2d(px) - c) / s @ R
        return p * [1, -1] + 0.5
    return fwd, inv


def write_variation_stage(path: str, plate: dict, var: Variation, *, cmap: str = "gray") -> str:
    from pinneapple_tools.nvidia.omniverse import FieldStage, grid_surface
    from pxr import Gf, Sdf, UsdGeom
    c = plate["x"]
    X, Y = np.meshgrid(c, c, indexing="ij")
    T = plate["T"]
    radiance = np.clip(var.gain * T / plate["T0"].max() * 0.85 + var.offset, 0, 1)
    if var.glint is not None:
        gu, gv, gr, gs = var.glint
        radiance = np.clip(radiance + gs * np.exp(-((X - gu) ** 2 + (Y - gv) ** 2) / (2 * gr ** 2)), 0, 1)
    V, F = grid_surface(X, Y)
    st = FieldStage(path, "Thermal inspection sample (PINNeAPPle World Model)")
    m = st.add_mesh("Plate", V, F, fields={"temperature": T.ravel(), "radiance": radiance.ravel(),
                                          "conductivity": plate["a"].ravel()},
                    units={"temperature": "K above ambient (scaled)", "radiance": "camera units"},
                    color_by="radiance", cmap=cmap, vrange=(0.0, 1.0))
    cam = UsdGeom.Camera.Define(st.stage, "/World/ThermalCamera")
    cam.AddTranslateOp().Set(Gf.Vec3d(0.5 + var.shift[0] / 100, 0.5 + var.shift[1] / 100, 2.0 / var.zoom))
    cam.AddRotateZOp().Set(float(var.angle_deg))
    cam.CreateProjectionAttr("orthographic")
    prim = st.stage.GetPrimAtPath(m)
    prim.CreateAttribute("pinneapple:label:hotspot_xy", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*plate["label_xy"]))
    prim.CreateAttribute("pinneapple:label:defect_xyr", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*plate["defect"]))
    for k, v in var.to_dict().items():
        prim.CreateAttribute(f"pinneapple:variation:{k}", Sdf.ValueTypeNames.String).Set(str(v))
    return st.save()


def thermal_image(stage_path: str, var: Variation, H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    """The camera image ``(H, W)`` of a variation stage."""
    from scipy import ndimage

    from pinneapple_tools.nvidia.omniverse import rasterize_topdown
    top = rasterize_topdown(stage_path, 96, 96, bounds=(1 / 96, 1 - 1 / 96, 1 / 96, 1 - 1 / 96)).mean(-1)
    fwd, inv = camera_transform(var, H, W)
    rr, cc = np.mgrid[0:H, 0:W]
    uv = inv(np.c_[cc.ravel(), rr.ravel()])
    src_r = (1 - uv[:, 1]) * (top.shape[0] - 1)
    src_c = uv[:, 0] * (top.shape[1] - 1)
    img = ndimage.map_coordinates(top, [src_r, src_c], order=1, cval=0.0).reshape(H, W)
    if var.blur > 0:
        img = ndimage.gaussian_filter(img, var.blur)
    img = img + np.random.default_rng(seed).normal(0, var.noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def label_check(image: np.ndarray, var: Variation, label_xy, *, tol: float = 0.08) -> dict:
    """Does the image still show the physical hot spot where the label says? Brightest point after a 2-px Gaussian,
    mapped back to the plate; ``tol`` in plate units (0.08 = about 4 cells)."""
    from scipy import ndimage
    sm = ndimage.gaussian_filter(image, 2.0)
    r, c = np.unravel_index(int(np.argmax(sm)), sm.shape)
    _, inv = camera_transform(var, *image.shape)
    u, v = inv(np.array([[c, r]], float))[0]
    err = float(np.hypot(u - label_xy[0], v - label_xy[1]))
    return {"found_xy": (float(u), float(v)), "error": err, "passed": err <= tol}


def cleanup(paths) -> None:
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
