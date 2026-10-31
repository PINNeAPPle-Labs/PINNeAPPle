"""Molecular visualisation: proteins and molecules, alone or adsorbed on metal surfaces, rendered as stills, MP4
trajectories, 3-D scenes (glTF for the PINNeAPPle web viewer) and MD dashboards.

The visual side of a protein-on-metal MD pipeline (the kind RoyalMD Gold runs with OpenMM), with nothing of an MD
engine in it: anything that gives atom positions can be shown (a PDB file, an OpenMM or mdtraj trajectory, the
torch MD of ``pinneapple_simulation``).

* :class:`Molecule`: atoms read from PDB (multi-MODEL files give the frames), element and residue chemistry,
  CA backbone trace, per-residue class (hydrophobic, polar, positive, negative, special);
* :func:`fcc_slab`: an fcc metal slab (Au, Ag, Cu, Pt, Pd, Ni, Al, Pb) with a (111), (100) or (110) top facet, sized
  in whole lattice repeats so a periodic box does not strain it; :func:`slab_for_footprint` sizes it from a
  molecule's footprint plus padding;
* :func:`orient_on_surface`: principal axes of the molecule, the thinnest direction along the surface normal (the
  broadest face down), an optional active site turned away from the surface, and a set gap above the slab;
* :func:`render`: a still with pyvista (off-screen): atoms as spheres coloured by residue class, element, chain or
  height, a backbone tube, the slab with a metallic material, three-point lighting;
* :func:`render_frames` / :func:`encode_mp4`: a trajectory as PNG frames and an H.264 MP4 (imageio-ffmpeg), and
  :func:`pinneapple_tools.visualization.studio.particles.compose_video` adds live charts next to the render;
* :func:`md_dashboard`: energy, temperature, RMSD, height above the surface and contacts over time;
* :func:`to_glb`: the scene as glTF (one mesh per material) for the three.js viewer;
* :class:`StepLog`: readable, optionally coloured step-by-step console log of a pipeline.
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Molecule", "LATTICE_CONSTANTS", "fcc_slab", "slab_for_footprint", "orient_on_surface", "residue_class",
           "rmsd", "height_above", "contacts", "render", "render_frames", "encode_mp4", "md_dashboard", "to_glb",
           "StepLog", "COLORS"]

# fcc lattice constants (Angstrom, room temperature)
LATTICE_CONSTANTS = {"Au": 4.078, "Ag": 4.086, "Cu": 3.615, "Pt": 3.924, "Pd": 3.891, "Ni": 3.524, "Al": 4.046,
                     "Pb": 4.950}
METAL_COLORS = {"Au": "#D4AF37", "Ag": "#C9CDD2", "Cu": "#C47A4A", "Pt": "#C9C6BE", "Pd": "#BDB7AB", "Ni": "#A8A39A",
                "Al": "#D6D9DD", "Pb": "#7D8086"}
VDW = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "ZN": 1.39, "FE": 1.40, "MG": 1.73,
       "CA": 1.97, "NA": 2.27, "CL": 1.75}
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974, "ZN": 65.38, "FE": 55.845}
ELEMENT_COLORS = {"H": "#E8E8E8", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "S": "#FFC832", "P": "#FF8000",
                  "ZN": "#7D80B0", "FE": "#E06633"}
COLORS = {"hydrophobic": "#E8C468", "polar": "#7FB8A4", "positive": "#4C78D8", "negative": "#D9534F",
          "special": "#B38BD9", "other": "#9A9A9A"}
_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
_POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
_POSITIVE = {"LYS", "ARG", "HIS", "HID", "HIE", "HIP"}
_NEGATIVE = {"ASP", "GLU"}
_SPECIAL = {"GLY"}


def residue_class(resname: str) -> str:
    r = resname.upper()
    for name, s in (("hydrophobic", _HYDROPHOBIC), ("polar", _POLAR), ("positive", _POSITIVE),
                    ("negative", _NEGATIVE), ("special", _SPECIAL)):
        if r in s:
            return name
    return "other"


@dataclass
class Molecule:
    xyz: np.ndarray                      # (n, 3) Angstrom
    elements: list[str]
    names: list[str] = field(default_factory=list)
    resnames: list[str] = field(default_factory=list)
    resids: list[int] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)   # extra MODEL frames of a PDB, (n, 3) each

    @classmethod
    def from_pdb(cls, path: str, keep_water: bool = False, keep_hetero: bool = True) -> Molecule:
        models: list[list] = [[]]
        for line in open(path, encoding="utf-8", errors="replace"):
            rec = line[:6].strip()
            if rec == "ENDMDL":
                models.append([])
            elif rec in ("ATOM", "HETATM"):
                res = line[17:20].strip()
                if res in ("HOH", "WAT", "SOL") and not keep_water:
                    continue
                if rec == "HETATM" and not keep_hetero and res not in ("HOH", "WAT", "SOL"):
                    continue
                if line[16] not in (" ", "A"):                         # alternate locations: keep the first
                    continue
                name = line[12:16].strip()
                el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:1].upper()
                models[-1].append((float(line[30:38]), float(line[38:46]), float(line[46:54]), el, name, res,
                                   int(line[22:26]), line[21].strip() or "A"))
        models = [m for m in models if m]
        if not models:
            raise ValueError(f"no atoms in {path}")
        first = models[0]
        mol = cls(np.array([a[:3] for a in first], dtype=float), [a[3] for a in first], [a[4] for a in first],
                  [a[5] for a in first], [a[6] for a in first], [a[7] for a in first])
        mol.frames = [np.array([a[:3] for a in m], dtype=float) for m in models[1:] if len(m) == len(first)]
        return mol

    @classmethod
    def from_mdtraj(cls, traj, frame: int = 0) -> Molecule:
        top = traj.topology
        atoms = list(top.atoms)
        return cls(traj.xyz[frame] * 10.0, [a.element.symbol.upper() if a.element else "X" for a in atoms],
                   [a.name for a in atoms], [a.residue.name for a in atoms], [a.residue.resSeq for a in atoms],
                   [a.residue.chain.chain_id or "A" for a in atoms],
                   [traj.xyz[k] * 10.0 for k in range(traj.n_frames) if k != frame])

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def masses(self) -> np.ndarray:
        return np.array([MASS.get(e, 12.0) for e in self.elements])

    @property
    def radii(self) -> np.ndarray:
        return np.array([VDW.get(e, 1.7) for e in self.elements])

    def classes(self) -> list[str]:
        return [residue_class(r) for r in self.resnames]

    def ca_trace(self, xyz: np.ndarray | None = None) -> list[np.ndarray]:
        """CA positions per chain, in residue order (broken where consecutive CAs are > 4.2 A apart)."""
        X = self.xyz if xyz is None else xyz
        out: list[np.ndarray] = []
        for ch in dict.fromkeys(self.chains):
            idx = [i for i, (n, c) in enumerate(zip(self.names, self.chains, strict=True)) if n == "CA" and c == ch]
            if len(idx) < 2:
                continue
            P = X[idx]
            cut = np.where(np.linalg.norm(np.diff(P, axis=0), axis=1) > 4.2)[0] + 1
            out += [seg for seg in np.split(P, cut) if len(seg) >= 2]
        return out


# ------------------------------------------------------------------------------------------------ surfaces
def fcc_slab(metal: str = "Au", facet: str = "111", nx: int = 10, ny: int = 10, layers: int = 4,
             a: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Atom positions (n, 3) of an fcc slab with the given top facet, top layer at z = 0, and the in-plane cell
    vectors (2, 3) of the periodic box it fills exactly (so the lattice is not strained by periodicity)."""
    a = a or LATTICE_CONSTANTS[metal]
    if facet == "111":
        d = a / math.sqrt(2)                                          # nearest-neighbour distance
        a1, a2 = np.array([d, 0, 0]), np.array([d / 2, d * math.sqrt(3) / 2, 0])
        dz = a / math.sqrt(3)
        shifts = [np.zeros(3), (a1 + a2) / 3, 2 * (a1 + a2) / 3]      # ABC stacking
        cell = np.array([nx * a1, ny * a2])
    elif facet == "100":
        d = a / math.sqrt(2)
        a1, a2 = np.array([d, 0, 0]), np.array([0, d, 0])
        dz = a / 2
        shifts = [np.zeros(3), (a1 + a2) / 2]
        cell = np.array([nx * a1, ny * a2])
    elif facet == "110":
        a1, a2 = np.array([a, 0, 0]), np.array([0, a / math.sqrt(2), 0])
        dz = a / (2 * math.sqrt(2))
        shifts = [np.zeros(3), (a1 + a2) / 2]
        cell = np.array([nx * a1, ny * a2])
    else:
        raise ValueError("facet must be '111', '100' or '110'")
    pts = []
    for layer in range(layers):
        s = shifts[layer % len(shifts)]
        for i in range(nx):
            for j in range(ny):
                pts.append(i * a1 + j * a2 + s - np.array([0, 0, layer * dz]))
    P = np.array(pts)
    if facet == "111":                                                # rhombic cell -> rectangular-looking slab
        P[:, 0] = np.mod(P[:, 0], cell[0, 0])
    return P, cell


def slab_for_footprint(xyz: np.ndarray, metal: str = "Au", facet: str = "111", padding: float = 8.0,
                       layers: int = 4) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """A slab under a molecule: whole lattice repeats covering its x-y footprint plus ``padding`` on each side,
    centred under it. Returns (positions, cell, (nx, ny))."""
    a = LATTICE_CONSTANTS[metal]
    span = xyz[:, :2].max(0) - xyz[:, :2].min(0) + 2 * padding
    d = a / math.sqrt(2) if facet in ("111", "100") else a
    dy = d * (math.sqrt(3) / 2 if facet == "111" else 1.0) if facet != "110" else a / math.sqrt(2)
    nx, ny = int(math.ceil(span[0] / d)), int(math.ceil(span[1] / dy))
    P, cell = fcc_slab(metal, facet, nx, ny, layers)
    mid = lambda A: 0.5 * (A[:, :2].min(0) + A[:, :2].max(0))     # noqa: E731 - centre of the bounding box
    shift = mid(xyz) - mid(P)
    P[:, :2] += shift
    return P, cell, (nx, ny)


def orient_on_surface(xyz: np.ndarray, masses: np.ndarray | None = None, active_site: Sequence[int] | None = None,
                      gap: float = 3.0, surface_z: float = 0.0) -> np.ndarray:
    """Rotate a molecule so its thinnest principal direction is the surface normal (the broadest face lies on the
    surface), turn ``active_site`` atoms (indices) away from the surface (else the flatter side faces down), and
    place its lowest atom ``gap`` above ``surface_z``."""
    m = np.ones(len(xyz)) if masses is None else np.asarray(masses, float)
    c = (xyz * m[:, None]).sum(0) / m.sum()
    X = xyz - c
    cov = (X * m[:, None]).T @ X / m.sum()
    w, V = np.linalg.eigh(cov)                                        # ascending: V[:, 0] = thinnest direction
    R = np.stack([V[:, 2], V[:, 1], V[:, 0]], axis=1)                 # long axis -> x, thinnest -> z
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    Y = X @ R
    if active_site is not None and len(active_site):
        flip = Y[list(active_site), 2].mean() < 0
    else:
        lo, hi = Y[:, 2] < np.quantile(Y[:, 2], 0.1), Y[:, 2] > np.quantile(Y[:, 2], 0.9)
        flip = np.ptp(Y[lo][:, :2], axis=0).prod() < np.ptp(Y[hi][:, :2], axis=0).prod()   # broader side down
    if flip:
        Y = Y * np.array([1, -1, -1])
    Y[:, 2] += surface_z + gap - Y[:, 2].min()
    Y[:, :2] -= Y[:, :2].mean(0)
    return Y


# ------------------------------------------------------------------------------------------------ analysis
def rmsd(frames: Sequence[np.ndarray], ref: np.ndarray, align: bool = True) -> np.ndarray:
    """RMSD of each frame against ``ref`` (Kabsch-aligned when ``align``)."""
    out = []
    r0 = ref - ref.mean(0)
    for f in frames:
        x = f - f.mean(0) if align else f - ref.mean(0) * 0
        if align:
            U, _, Vt = np.linalg.svd(x.T @ r0)
            dd = np.sign(np.linalg.det(U @ Vt))
            x = x @ (U @ np.diag([1, 1, dd]) @ Vt)
            out.append(float(np.sqrt(np.mean(np.sum((x - r0) ** 2, 1)))))
        else:
            out.append(float(np.sqrt(np.mean(np.sum((f - ref) ** 2, 1)))))
    return np.array(out)


def height_above(frames: Sequence[np.ndarray], surface_z: float, masses: np.ndarray | None = None) -> np.ndarray:
    """Height of the centre of mass above the surface plane, per frame."""
    m = None if masses is None else np.asarray(masses, float)
    return np.array([(np.average(f[:, 2], weights=m) if m is not None else f[:, 2].mean()) - surface_z
                     for f in frames])


def contacts(frames: Sequence[np.ndarray], surface_z: float, cutoff: float = 3.5) -> np.ndarray:
    """Number of atoms within ``cutoff`` (A) of the surface plane, per frame."""
    return np.array([int(np.sum(f[:, 2] - surface_z < cutoff)) for f in frames])


# ------------------------------------------------------------------------------------------------ rendering
def _colors(mol: Molecule, xyz: np.ndarray, color_by: str, surface_z: float) -> np.ndarray:
    from matplotlib import colormaps
    from matplotlib import colors as mcolors

    if color_by == "residue":
        return np.array([mcolors.to_rgb(COLORS[c]) for c in mol.classes()])
    if color_by == "element":
        return np.array([mcolors.to_rgb(ELEMENT_COLORS.get(e, "#C0C0C0")) for e in mol.elements])
    if color_by == "chain":
        ch = list(dict.fromkeys(mol.chains))
        pal = colormaps["tab10"]
        return np.array([pal(ch.index(c) % 10)[:3] for c in mol.chains])
    if color_by == "height":
        h = xyz[:, 2] - surface_z
        return colormaps["viridis"](np.clip(h / max(np.percentile(h, 98), 1e-6), 0, 1))[:, :3]
    raise ValueError("color_by: residue, element, chain or height")


def _plotter(size, background):
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=list(size), lighting="none")
    p.set_background(background, top="#2A2F3A" if background.lower() in ("#0b0e14", "black", "#000000") else None)
    for pos, inten in (((1, -1, 1.6), 0.9), ((-1.2, -0.4, 0.8), 0.45), ((0.2, 1.4, 1.0), 0.6)):
        p.add_light(pv.Light(position=pos, focal_point=(0, 0, 0), intensity=inten, light_type="camera light"))
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:  # noqa: BLE001 - older pyvista or no support: render without
        pass
    return p


def _add_atoms(p, xyz, radii, rgb, scale, name, pbr=False, metallic=0.0, roughness=0.5, resolution=14,
               specular=0.35, specular_power=30):
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(xyz, float))
    cloud["r"] = np.asarray(radii, float) * scale
    cloud["rgb"] = (np.asarray(rgb) * 255).astype(np.uint8)
    glyphs = cloud.glyph(scale="r", geom=pv.Sphere(radius=1.0, theta_resolution=resolution,
                                                   phi_resolution=resolution), orient=False)
    kw = dict(scalars="rgb", rgb=True, smooth_shading=True, name=name, specular=specular,
              specular_power=specular_power)
    if pbr:
        kw.update(pbr=True, metallic=metallic, roughness=roughness)
        kw.pop("specular")
        kw.pop("specular_power")
    p.add_mesh(glyphs, **kw)


def _scene(p, mol, xyz, slab, metal, color_by, style, surface_z):
    import pyvista as pv
    from matplotlib import colors as mcolors

    rgb = _colors(mol, xyz, color_by, surface_z)
    scale = 0.55 if style == "cartoon" else 0.9
    if style == "cartoon":
        for seg in mol.ca_trace(xyz):
            spline = pv.Spline(seg, max(len(seg) * 6, 10))
            p.add_mesh(spline.tube(radius=1.1, n_sides=14), color="#F2E6C9", smooth_shading=True,
                       specular=0.4, name=f"trace{id(seg)}")
        side = np.array([n not in ("N", "C", "O", "CA") and e != "H" for n, e in zip(mol.names, mol.elements, strict=True)])
        _add_atoms(p, xyz[side], mol.radii[side], rgb[side], 0.45, "atoms", resolution=10)
    else:
        _add_atoms(p, xyz, mol.radii, rgb, scale, "atoms")
    if slab is not None and len(slab):
        c = np.array(mcolors.to_rgb(METAL_COLORS.get(metal, "#D4AF37")))
        top = slab[:, 2] > slab[:, 2].max() - 0.5
        shade = np.where(top[:, None], np.minimum(c * 1.15, 1.0), c * 0.75)        # top layer brighter
        _add_atoms(p, slab, np.full(len(slab), 1.44), shade, 1.0, "slab", resolution=12, specular=0.9,
                   specular_power=45)


def render(mol: Molecule, path: str, *, xyz: np.ndarray | None = None, slab: np.ndarray | None = None,
           metal: str = "Au", color_by: str = "residue", style: str = "spheres", size=(1600, 900),
           background: str = "#0B0E14", view: tuple[float, float, float] = (1.0, -1.6, 0.75), zoom: float = 1.15,
           surface_z: float | None = None, title: str = "") -> str:
    """A still of ``mol`` (positions ``xyz`` or its own) on ``slab``. ``style``: "spheres" (space-filling) or
    "cartoon" (backbone tube and side-chain atoms); ``color_by``: residue class, element, chain or height."""
    X = mol.xyz if xyz is None else np.asarray(xyz, float)
    sz = (slab[:, 2].max() if slab is not None and len(slab) else X[:, 2].min()) if surface_z is None else surface_z
    p = _plotter(size, background)
    _scene(p, mol, X, slab, metal, color_by, style, sz)
    c = X.mean(0)
    dist = 2.6 * max(np.ptp(X, 0).max(), 30.0)
    v = np.asarray(view, float) / np.linalg.norm(view)
    p.camera.position = tuple(c + v * dist)
    p.camera.focal_point = tuple(c - np.array([0, 0, 0.15 * np.ptp(X[:, 2])]))
    p.camera.up = (0, 0, 1)
    p.camera.zoom(zoom)
    if title:
        p.add_text(title, position="upper_left", font_size=12, color="#E8E8E8")
    p.screenshot(path)
    p.close()
    return path


def render_frames(mol: Molecule, frames: Sequence[np.ndarray], out_dir: str, *, slab=None, metal: str = "Au",
                  color_by: str = "residue", style: str = "spheres", size=(1280, 720), background: str = "#0B0E14",
                  view=(1.0, -1.6, 0.75), zoom: float = 1.15, orbit_deg: float = 0.0,
                  labels: Sequence[str] | None = None) -> list[str]:
    """One PNG per frame, with a fixed camera (or a slow orbit of ``orbit_deg`` over the whole run) so motion is
    readable. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    allx = np.concatenate(frames)
    c = allx.mean(0)
    dist = 2.6 * max(np.ptp(allx, 0).max(), 30.0)
    paths = []
    for k, X in enumerate(frames):
        ang = math.radians(orbit_deg * k / max(len(frames) - 1, 1))
        v = np.asarray(view, float)
        v = np.array([v[0] * math.cos(ang) - v[1] * math.sin(ang), v[0] * math.sin(ang) + v[1] * math.cos(ang), v[2]])
        v /= np.linalg.norm(v)
        p = _plotter(size, background)
        _scene(p, mol, np.asarray(X, float), slab, metal, color_by, style,
               slab[:, 2].max() if slab is not None and len(slab) else allx[:, 2].min())
        p.camera.position = tuple(c + v * dist)
        p.camera.focal_point = tuple(c - np.array([0, 0, 0.15 * np.ptp(allx[:, 2])]))
        p.camera.up = (0, 0, 1)
        p.camera.zoom(zoom)
        if labels is not None:
            p.add_text(labels[k], position="lower_right", font_size=11, color="#E8E8E8")
        path = os.path.join(out_dir, f"frame_{k:04d}.png")
        p.screenshot(path)
        p.close()
        paths.append(path)
    return paths


def encode_mp4(pngs: Sequence[str], out: str, fps: int = 12, crf: int = 22) -> str:
    """H.264 MP4 from PNG frames with the ffmpeg binary bundled by ``imageio-ffmpeg``."""
    import subprocess

    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    d = os.path.dirname(os.path.abspath(pngs[0]))
    pattern = os.path.join(d, os.path.basename(pngs[0]).rsplit("_", 1)[0] + "_%04d.png")
    subprocess.run([ff, "-y", "-v", "error", "-framerate", str(fps), "-i", pattern, "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", str(crf),
                    "-movflags", "+faststart", out], check=True)
    return out


def md_dashboard(path: str, t_ps: Sequence[float], *, potential: Sequence[float] | None = None,
                 temperature: Sequence[float] | None = None, rmsd_A: Sequence[float] | None = None,
                 height_A: Sequence[float] | None = None, contacts_n: Sequence[float] | None = None,
                 target_temperature: float | None = None, title: str = "") -> str:
    """A panel per observable given, over time: potential energy (kJ/mol), temperature (K, with the target),
    backbone RMSD (A), height of the centre of mass above the surface (A), atoms in contact with the surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(potential, "potential energy (kJ/mol)", "#4C78D8"), (temperature, "temperature (K)", "#D9534F"),
              (rmsd_A, "backbone RMSD (Å)", "#7FB8A4"), (height_A, "height above surface (Å)", "#E8C468"),
              (contacts_n, "atoms within 3.5 Å of the surface", "#B38BD9")]
    panels = [p for p in panels if p[0] is not None]
    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 1.9 * len(panels) + 0.6), sharex=True, squeeze=False)
    for ax, (y, lab, col) in zip(axs[:, 0], panels, strict=True):
        ax.plot(t_ps, y, color=col, lw=1.6)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.25)
        if lab.startswith("temperature") and target_temperature:
            ax.axhline(target_temperature, color="0.4", ls="--", lw=0.8)
    axs[-1, 0].set_xlabel("time (ps)")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def to_glb(mol: Molecule, path: str, *, xyz: np.ndarray | None = None, slab: np.ndarray | None = None,
           metal: str = "Au", max_atoms: int = 20000) -> str:
    """The scene as glTF binary: one low-poly sphere per atom, merged per residue class, and the slab as metal;
    opens in the PINNeAPPle web viewer and any glTF viewer."""
    from pinneapple_tools.visualization.studio.scene import Scene, Surface

    X = mol.xyz if xyz is None else np.asarray(xyz, float)
    ico_v, ico_f = _icosphere()
    surfaces = []
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(mol.classes()[:max_atoms]):
        groups.setdefault(c, []).append(i)
    for c, idx in groups.items():
        V, F = _instances(ico_v, ico_f, X[idx], mol.radii[idx] * 0.9)
        surfaces.append(Surface(c, V, F, material=c))
    if slab is not None and len(slab):
        V, F = _instances(ico_v, ico_f, slab, np.full(len(slab), 1.44))
        surfaces.append(Surface(metal, V, F, material="metal"))
    sc = Scene(surfaces)
    from matplotlib import colors as mcolors

    mats = {c: {"color": list(mcolors.to_rgb(COLORS[c])), "metallic": 0.0, "roughness": 0.6} for c in groups}
    mats["metal"] = {"color": list(mcolors.to_rgb(METAL_COLORS.get(metal, "#D4AF37"))), "metallic": 1.0,
                     "roughness": 0.3}
    with open(path, "wb") as fh:
        fh.write(sc.to_glb(mats))
    return path


def _icosphere():
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                  [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return V, F


def _instances(V, F, centers, radii):
    n = len(centers)
    VV = (V[None] * np.asarray(radii)[:, None, None] + np.asarray(centers)[:, None, :]).reshape(-1, 3)
    FF = (F[None] + (np.arange(n) * len(V))[:, None, None]).reshape(-1, 3)
    return VV, FF


# ------------------------------------------------------------------------------------------------ console
class StepLog:
    """Step-by-step console log of a pipeline: numbered steps, details indented under them, elapsed time, colour on
    a terminal (plain text otherwise or with ``color=False``); every line is also kept in ``lines``."""

    def __init__(self, title: str = "", color: bool | None = None, stream=None):
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty() if color is None else color
        self.lines: list[str] = []
        self.n = 0
        self.t0 = time.perf_counter()
        if title:
            self._emit(title, "\033[1;38;5;220m")

    def _emit(self, text: str, style: str = "") -> None:
        self.lines.append(text)
        print(f"{style}{text}\033[0m" if self.color and style else text, file=self.stream, flush=True)

    def step(self, text: str) -> None:
        self.n += 1
        self._emit(f"[{self.n}] {text}  ({time.perf_counter() - self.t0:.1f} s)", "\033[1;38;5;117m")

    def info(self, text: str) -> None:
        self._emit(f"    {text}")

    def ok(self, text: str) -> None:
        self._emit(f"    ok: {text}", "\033[38;5;114m")

    def warn(self, text: str) -> None:
        self._emit(f"    warning: {text}", "\033[38;5;215m")
