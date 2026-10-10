"""3-D linear elasticity with 8-node hexahedra and incompatible modes (Wilson-Taylor; the element Abaqus calls
C3D8I), so slender parts in bending do not lock. Structured box meshes, clamps, supports, nodal and pressure loads,
nodal stresses (extrapolated from the Gauss points and averaged) and von Mises; an optional cross-check of the same
mesh with CalculiX (``ccx``) and post-processor style figures (deformed mesh, discrete colour bands, legend).

    from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh, SolidFEM
    mesh = box_mesh(2.0, 0.1, 0.2, 80, 5, 10)                  # x along the beam, z up
    fem = SolidFEM(mesh, E=210e9, nu=0.3)
    fem.fix(mesh.nodes_on(x=0.0))                              # clamp the root
    fem.load_face(mesh.face_nodes("x+"), (0, 0, -1e4))         # 10 kN shared on the tip face
    res = fem.solve()
    res.u, res.von_mises, res.max_displacement
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

_G = 1 / math.sqrt(3)
_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)     # Abaqus / VTK hexahedron order


@dataclass
class HexMesh:
    nodes: np.ndarray                    # (n, 3)
    elements: np.ndarray                 # (m, 8)
    shape: tuple[int, int, int] = (0, 0, 0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def nodes_on(self, x: float | None = None, y: float | None = None, z: float | None = None,
                 tol: float = 1e-9) -> np.ndarray:
        m = np.ones(len(self.nodes), bool)
        for k, v in enumerate((x, y, z)):
            if v is not None:
                m &= np.abs(self.nodes[:, k] - v) < tol * max(1.0, max(self.size))
        return np.nonzero(m)[0]

    def boundary_faces(self) -> np.ndarray:
        """Quadrilateral faces on the boundary (each once), outward ordering."""
        loc = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        F = self.elements[:, loc].reshape(-1, 4)
        key = np.sort(F, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return F[cnt[inv.ravel()] == 1]

    def face_nodes(self, side: str) -> np.ndarray:
        """Nodes on one side of the box: "x-", "x+", "y-", "y+", "z-", "z+"."""
        k = "xyz".index(side[0])
        v = self.nodes[:, k].min() if side[1] == "-" else self.nodes[:, k].max()
        return self.nodes_on(**{side[0]: v})


def box_mesh(L: float, W: float, H: float, nx: int, ny: int, nz: int, origin=(0.0, 0.0, 0.0)) -> HexMesh:
    """Structured hexahedral mesh of a box [0, L] x [-W/2, W/2] x [-H/2, H/2] (shifted by ``origin``)."""
    xs = np.linspace(0, L, nx + 1) + origin[0]
    ys = np.linspace(-W / 2, W / 2, ny + 1) + origin[1]
    zs = np.linspace(-H / 2, H / 2, nz + 1) + origin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k          # noqa: E731
    ii, jj, kk = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    el = np.stack([nid(ii, jj, kk), nid(ii + 1, jj, kk), nid(ii + 1, jj + 1, kk), nid(ii, jj + 1, kk),
                   nid(ii, jj, kk + 1), nid(ii + 1, jj, kk + 1), nid(ii + 1, jj + 1, kk + 1), nid(ii, jj + 1, kk + 1)], 1)
    return HexMesh(nodes, el, (nx, ny, nz), (L, W, H))


def _dN(xi):
    """Shape-function derivatives d N_a / d xi_k at a natural point: (8, 3)."""
    s = _XI
    d = np.empty((8, 3))
    d[:, 0] = s[:, 0] * (1 + s[:, 1] * xi[1]) * (1 + s[:, 2] * xi[2]) / 8
    d[:, 1] = s[:, 1] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 2] * xi[2]) / 8
    d[:, 2] = s[:, 2] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 1] * xi[1]) / 8
    return d


def _N(xi):
    return np.prod(1 + _XI * np.asarray(xi), axis=1) / 8


def _B(dNdx):
    """Strain-displacement matrices (..., 6, 3 n) from shape-function gradients (..., n, 3); Voigt
    (xx, yy, zz, xy, yz, zx) with engineering shear."""
    sh = dNdx.shape[:-2]
    n = dNdx.shape[-2]
    B = np.zeros(sh + (6, 3 * n))
    gx, gy, gz = dNdx[..., 0], dNdx[..., 1], dNdx[..., 2]
    B[..., 0, 0::3], B[..., 1, 1::3], B[..., 2, 2::3] = gx, gy, gz
    B[..., 3, 0::3], B[..., 3, 1::3] = gy, gx
    B[..., 4, 1::3], B[..., 4, 2::3] = gz, gy
    B[..., 5, 0::3], B[..., 5, 2::3] = gz, gx
    return B


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D


def von_mises(s: np.ndarray) -> np.ndarray:
    """von Mises stress of Voigt stresses (..., 6)."""
    xx, yy, zz, xy, yz, zx = (s[..., k] for k in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))


@dataclass
class FEMResult:
    mesh: HexMesh
    u: np.ndarray                         # (n, 3) nodal displacements
    stress: np.ndarray                    # (n, 6) nodal stresses (extrapolated, averaged)
    reactions: np.ndarray                 # (n, 3)
    info: dict = field(default_factory=dict)

    @property
    def von_mises(self) -> np.ndarray:
        return von_mises(self.stress)

    @property
    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.u, axis=1).max())

    def surface(self, scale: float = 1.0):
        """Triangulated boundary of the deformed mesh with the nodal fields (for the studio Scene / renders)."""
        Q = self.mesh.boundary_faces()
        used = np.unique(Q)
        remap = -np.ones(len(self.mesh.nodes), int)
        remap[used] = np.arange(len(used))
        Qr = remap[Q]
        T = np.vstack([Qr[:, [0, 1, 2]], Qr[:, [0, 2, 3]]])
        V = self.mesh.nodes[used] + scale * self.u[used]
        return V, T, {"von_mises": self.von_mises[used], "displacement": np.linalg.norm(self.u[used], axis=1),
                      "sxx": self.stress[used, 0]}


@dataclass
class SolidFEM:
    mesh: HexMesh
    E: float = 210e9
    nu: float = 0.3
    incompatible: bool = True
    _fixed: dict = field(default_factory=dict)
    _f: np.ndarray | None = None

    def __post_init__(self):
        self._f = np.zeros(3 * len(self.mesh.nodes))

    # -- boundary conditions
    def fix(self, nodes, dofs=(0, 1, 2), value: float = 0.0) -> SolidFEM:
        for n in np.atleast_1d(nodes):
            for d in dofs:
                self._fixed[3 * int(n) + d] = value
        return self

    def load_nodes(self, nodes, force) -> SolidFEM:
        """A total force shared equally on ``nodes``."""
        nodes = np.atleast_1d(nodes)
        for d in range(3):
            self._f[3 * nodes + d] += force[d] / len(nodes)
        return self

    def load_face(self, nodes, force) -> SolidFEM:
        """A total force as a uniform traction on the boundary faces whose four nodes are all in ``nodes``
        (consistent nodal loads)."""
        Q = self.mesh.boundary_faces()
        s = set(int(n) for n in np.atleast_1d(nodes))
        Q = Q[[all(int(v) in s for v in q) for q in Q]]
        P = self.mesh.nodes[Q]
        area = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1)
        w = area / area.sum()
        for q, wi in zip(Q, w, strict=True):
            for n in q:
                for d in range(3):
                    self._f[3 * n + d] += force[d] * wi / 4
        return self

    def pressure(self, nodes, p: float) -> SolidFEM:
        """Pressure ``p`` (Pa, positive pushing into the body) on the boundary faces made of ``nodes``."""
        Q = self.mesh.boundary_faces()
        s = set(int(n) for n in np.atleast_1d(nodes))
        Q = Q[[all(int(v) in s for v in q) for q in Q]]
        P = self.mesh.nodes[Q]
        nA = 0.5 * np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])            # outward area vector
        for q, a in zip(Q, nA, strict=True):
            for n in q:
                self._f[3 * n: 3 * n + 3] += -p * a / 4
        return self

    # -- element matrices
    def _element_matrices(self):
        X = self.mesh.nodes[self.mesh.elements]                              # (m, 8, 3)
        D = elasticity_matrix(self.E, self.nu)
        gps = np.array([[a, b, c] for c in (-_G, _G) for b in (-_G, _G) for a in (-_G, _G)])
        J0 = np.einsum("ak,mai->mki", _dN(np.zeros(3)), X)                     # (m, 3, 3) at the centre
        det0 = np.linalg.det(J0)
        J0inv = np.linalg.inv(J0)
        m = len(X)
        Kuu = np.zeros((m, 24, 24))
        Kua = np.zeros((m, 24, 9))
        Kaa = np.zeros((m, 9, 9))
        Bs, Bas = [], []
        for xi in gps:
            dN = _dN(xi)
            J = np.einsum("ak,mai->mki", dN, X)
            det = np.linalg.det(J)
            dNdx = np.einsum("mki,ak->mai", np.linalg.inv(J), dN)            # (m, 8, 3)
            B = _B(dNdx)
            Bs.append(B)
            if self.incompatible:
                dP = np.diag(-2 * xi)                                         # d(1 - xi_k^2)/d xi_j
                dPdx = np.einsum("mki,ak->mai", J0inv, dP) * (det0 / det)[:, None, None]
                Ba = _B(dPdx)
                Bas.append(Ba)
                Kua += np.einsum("mji,jk,mkl,m->mil", B, D, Ba, det)
                Kaa += np.einsum("mji,jk,mkl,m->mil", Ba, D, Ba, det)
            Kuu += np.einsum("mji,jk,mkl,m->mil", B, D, B, det)
        if self.incompatible:
            KaaInv = np.linalg.inv(Kaa)
            K = Kuu - np.einsum("mia,mab,mjb->mij", Kua, KaaInv, Kua)
        else:
            KaaInv = None
            K = Kuu
        return K, D, gps, Bs, Bas, Kua, KaaInv

    def solve(self) -> FEMResult:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        n = len(self.mesh.nodes)
        K_e, D, gps, Bs, Bas, Kua, KaaInv = self._element_matrices()
        dof = (3 * self.mesh.elements[:, :, None] + np.arange(3)).reshape(len(self.mesh.elements), 24)
        rows = np.repeat(dof, 24, axis=1).ravel()
        cols = np.tile(dof, (1, 24)).ravel()
        K = sp.coo_matrix((K_e.ravel(), (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
        fixed = np.array(sorted(self._fixed), int)
        uf = np.array([self._fixed[k] for k in fixed])
        free = np.setdiff1d(np.arange(3 * n), fixed)
        u = np.zeros(3 * n)
        u[fixed] = uf
        rhs = self._f[free] - K[free][:, fixed] @ uf
        u[free] = spl.spsolve(K[free][:, free].tocsc(), rhs)
        R = (K @ u - self._f).reshape(n, 3)
        # stresses at the Gauss points, extrapolated to the nodes, averaged
        ue = u[dof]                                                          # (m, 24)
        alpha = -np.einsum("mab,mib,mi->ma", KaaInv, Kua, ue) if self.incompatible else None
        sg = []
        for g in range(8):
            eps = np.einsum("mij,mj->mi", Bs[g], ue)
            if self.incompatible:
                eps += np.einsum("mij,mj->mi", Bas[g], alpha)
            sg.append(eps @ D.T)
        sg = np.stack(sg, 1)                                                 # (m, 8 gauss, 6)
        # extrapolation: node a sits at xi = sqrt(3) * xi_a in the Gauss-point coordinates
        Ex = np.array([_N(math.sqrt(3) * _XI[a]) for a in range(8)])        # (8 nodes, 8 gauss)
        # gauss order above is (a fastest): matches _XI ordering only after mapping
        gp_sign = np.sign(gps)
        order = [int(np.nonzero((np.sign(_XI) == gp_sign[g]).all(1))[0][0]) for g in range(8)]
        Ex = Ex[:, order]
        sn = np.einsum("ag,mgk->mak", Ex, sg)
        acc = np.zeros((n, 6))
        cnt = np.zeros(n)
        np.add.at(acc, self.mesh.elements.ravel(), sn.reshape(-1, 6))
        np.add.at(cnt, self.mesh.elements.ravel(), 1)
        stress = acc / cnt[:, None]
        return FEMResult(self.mesh, u.reshape(n, 3), stress, R,
                         {"dofs": 3 * n, "elements": len(self.mesh.elements), "E": self.E, "nu": self.nu,
                          "element": "C3D8I" if self.incompatible else "C3D8"})

    # -- CalculiX cross-check
    def calculix(self, work: str) -> np.ndarray | None:
        """Solve the same model with CalculiX (C3D8I) and return its nodal displacements, or None when ``ccx`` is
        not installed."""
        exe = shutil.which("ccx")
        if not exe:
            return None
        os.makedirs(work, exist_ok=True)
        L = ["*NODE"] + [f"{i + 1}, {x:.12g}, {y:.12g}, {z:.12g}" for i, (x, y, z) in enumerate(self.mesh.nodes)]
        L += [f"*ELEMENT, TYPE={'C3D8I' if self.incompatible else 'C3D8'}, ELSET=ALL"]
        L += [f"{e + 1}, " + ", ".join(str(int(v) + 1) for v in el) for e, el in enumerate(self.mesh.elements)]
        L += ["*MATERIAL, NAME=M", "*ELASTIC", f"{self.E:.12g}, {self.nu}", "*SOLID SECTION, ELSET=ALL, MATERIAL=M",
              "*STEP", "*STATIC", "*BOUNDARY"]
        L += [f"{k // 3 + 1}, {k % 3 + 1}, {k % 3 + 1}, {v:.12g}" for k, v in sorted(self._fixed.items())]
        L += ["*CLOAD"] + [f"{k // 3 + 1}, {k % 3 + 1}, {v:.12g}" for k, v in enumerate(self._f) if v != 0]
        L += ["*NODE FILE", "U", "*END STEP"]
        with open(os.path.join(work, "model.inp"), "w") as f:
            f.write("\n".join(L) + "\n")
        subprocess.run([exe, "model"], cwd=work, capture_output=True, timeout=3600)
        U = np.zeros((len(self.mesh.nodes), 3))
        ok = False
        block = False
        with open(os.path.join(work, "model.frd")) as f:                    # fixed-width result file
            for line in f:
                if line.startswith(" -4") and "DISP" in line:
                    block = True
                elif block and line.startswith(" -3"):
                    break
                elif block and line.startswith(" -1"):
                    i = int(line[3:13])
                    U[i - 1] = [float(line[13 + 12 * k: 25 + 12 * k]) for k in range(3)]
                    ok = True
        return U if ok else None


# ---------------------------------------------------------------------- post-processor figures
def fea_figure(res: FEMResult, field: str = "von_mises", scale: float | None = None, bands: int = 12,
               title: str = "", unit: str = "MPa", factor: float = 1e-6, view=(22, -58), ax=None,
               vrange: tuple[float, float] | None = None, ghost: bool = True, edges: bool = True,
               zoom: float = 0.95):
    """The deformed boundary mesh coloured by ``field`` in discrete bands with black element edges and a
    post-processor legend (S, Mises), the undeformed outline in grey. ``scale``: deformation scale factor (default:
    the largest displacement drawn as 8 % of the part's length)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = res.mesh
    Lx = np.ptp(mesh.nodes, axis=0).max()
    if scale is None:
        scale = 0.08 * Lx / max(res.max_displacement, 1e-30)
    vals = {"von_mises": res.von_mises, "displacement": np.linalg.norm(res.u, axis=1),
            "sxx": res.stress[:, 0]}[field] * (factor if field != "displacement" else 1e3)
    Q = mesh.boundary_faces()
    P = mesh.nodes[Q] + scale * res.u[Q]
    e, a = np.radians(view)
    eye = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    nrm = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    front = nrm @ eye > 0                                                    # orthographic back-face culling
    Q, P = Q[front], P[front]
    fv = vals[Q].mean(1)
    lo, hi = vrange if vrange is not None else (float(vals.min()), float(vals.max()))
    lev = np.linspace(lo, hi, bands + 1)
    cmap = plt.get_cmap("jet", bands)
    norm = colors.BoundaryNorm(lev, bands)
    fc = cmap(norm(np.clip(fv, lo, hi - 1e-12 * (hi - lo + 1e-30))))
    own = ax is None
    if own:
        fig = plt.figure(figsize=(10, 6.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    if ghost:
        P0 = mesh.nodes[Q]
        ax.add_collection3d(Poly3DCollection(P0, facecolors=(0, 0, 0, 0), edgecolors=(0.55, 0.55, 0.55, 0.25),
                                             linewidths=0.2))
    shade = _lambert(P)
    fc[:, :3] *= (0.62 + 0.38 * shade)[:, None]
    ax.add_collection3d(Poly3DCollection(P, facecolors=fc, edgecolors=(0, 0, 0, 0.55) if edges else fc,
                                         linewidths=0.25 if edges else 0))
    allp = np.vstack([P.reshape(-1, 3), mesh.nodes])
    lo3, hi3 = allp.min(0), allp.max(0)
    span = np.maximum(hi3 - lo3, 0.18 * (hi3 - lo3).max())
    c = 0.5 * (lo3 + hi3)
    ax.set_xlim(c[0] - span[0] / 2, c[0] + span[0] / 2)
    ax.set_ylim(c[1] - span[1] / 2, c[1] + span[1] / 2)
    ax.set_zlim(c[2] - span[2] / 2, c[2] + span[2] / 2)
    ax.set_box_aspect(tuple(span), zoom=zoom)
    ax.set_proj_type("ortho")
    if own:
        fig.subplots_adjust(0.12, 0.0, 1.0, 1.0)
    ax.view_init(*view)
    ax.set_axis_off()
    # post-processor legend
    name = {"von_mises": f"S, Mises ({unit})", "displacement": "U, Magnitude (mm)", "sxx": f"S, S11 ({unit})"}[field]
    cax = fig.add_axes([0.04, 0.50, 0.025, 0.38])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=lev)
    cb.ax.set_yticklabels([f"{v:+.3e}" for v in lev], fontsize=7, family="monospace")
    cb.ax.set_title(name + "\n(Avg: 100%)", fontsize=8, loc="left", family="monospace")
    fig.text(0.04, 0.04, f"{title}\nDeformed shape: U, deformation scale factor {scale:.4g}\n"
             f"{res.info.get('element', 'C3D8I')} elements: {res.info.get('elements', 0)}, DOF: "
             f"{res.info.get('dofs', 0)}", fontsize=7.5, family="monospace", color="#333")
    _triad(fig)
    return fig, scale


def _lambert(P: np.ndarray) -> np.ndarray:
    n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    light = np.array([0.35, -0.55, 0.76])
    return np.abs(n @ (light / np.linalg.norm(light)))


def _triad(fig):
    ax = fig.add_axes([0.86, 0.04, 0.1, 0.12])
    ax.set_axis_off()
    for (dx, dy), lab, col in (((1, 0.25), "X", "#d62728"), ((0.0, 1), "Z", "#1f77b4"), ((-0.55, -0.35), "Y", "#2ca02c")):
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
        ax.text(dx * 1.15, dy * 1.15, lab, color=col, fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.8, 1.3)
    ax.set_ylim(-0.6, 1.3)
