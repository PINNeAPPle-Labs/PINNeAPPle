"""Finite-volume operators on an OpenFOAM mesh (Gauss theorem over the faces of each cell), and the conservative
residuals and boundary balances built on them.

Face values: linear interpolation between the two cells of an internal face; on boundary faces the value written
in the field's ``boundaryField`` (the owner-cell value where nothing is written, e.g. ``zeroGradient``). Face
fluxes come from ``phi`` when the solver wrote it, which makes the continuity check exact up to the solver
tolerance. Faces of ``empty`` patches (the front and back of a 2-D case) carry no flux.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .fields import FieldData

__all__ = ["FV"]


class FV:
    def __init__(self, fd: FieldData):
        g = fd.geom
        self.fd = fd
        self.own, self.nei, self.ni, self.nc = g["owner"], g["neighbour"], g["n_internal"], g["n_cells"]
        self.S, self.Cf, self.C, self.V = g["areas"], g["face_centres"], g["cell_centres"], g["volumes"]
        self.empty = g["empty"]
        self.dirs = g["dirs"]
        self.nb = len(self.own) - self.ni
        self.Smag = np.linalg.norm(self.S, axis=1)
        own_i, nei = self.own[:self.ni], self.nei
        dP = np.linalg.norm(self.Cf[:self.ni] - self.C[own_i], axis=1)
        dN = np.linalg.norm(self.Cf[:self.ni] - self.C[nei], axis=1)
        self.wP = dN / np.maximum(dP + dN, 1e-300)
        d = self.C[nei] - self.C[own_i]
        self.coef_i = self.Smag[:self.ni] ** 2 / np.maximum(np.abs(np.einsum("ij,ij->i", d, self.S[:self.ni])), 1e-300)
        db = self.Cf[self.ni:] - self.C[self.own[self.ni:]]
        self.coef_b = self.Smag[self.ni:] ** 2 / np.maximum(np.abs(np.einsum("ij,ij->i", db, self.S[self.ni:])), 1e-300)
        self.coef_b[self.empty] = 0.0
        self.interior = np.ones(self.nc, bool)
        bcells = self.own[self.ni:][~self.empty]
        self.interior[bcells] = False

    # ------------------------------------------------------------------ basics
    def bvals(self, name: str, cell: np.ndarray, source: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Boundary-face values of a field (owner value where the file gives none)."""
        src = self.fd.boundary if source is None else source
        own_b = cell[self.own[self.ni:]]
        b = src.get(name)
        if b is None or b.shape[0] != self.nb:
            return own_b.copy()
        b = np.array(b, float).reshape(own_b.shape)
        bad = ~np.isfinite(b)
        b[bad] = own_b[bad]
        return b

    def face(self, cell: np.ndarray, b: np.ndarray) -> np.ndarray:
        w = self.wP if cell.ndim == 1 else self.wP[:, None]
        fi = w * cell[self.own[:self.ni]] + (1 - w) * cell[self.nei]
        return np.concatenate([fi, b])

    def sum_faces(self, q: np.ndarray, absolute: bool = False) -> np.ndarray:
        """Per cell: sum over its faces of a face quantity counted outward (+owner, -neighbour)."""
        q = np.abs(q) if absolute else q
        shape = (self.nc,) + q.shape[1:]
        out = np.zeros(shape)
        np.add.at(out, self.own, q)
        np.add.at(out, self.nei, (1 if absolute else -1) * q[:self.ni])
        return out

    def flux(self, ucell: np.ndarray, ub: np.ndarray, phi: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if phi is not None:
            F = np.concatenate([phi[0], phi[1]])
        else:
            F = np.einsum("ij,ij->i", self.face(ucell, ub), self.S)
        F[self.ni:][self.empty] = 0.0
        return F

    def grad(self, cell: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Gauss gradient of a scalar, (nc, 3)."""
        pf = self.face(cell, b)
        G = self.sum_faces(pf[:, None] * self.S) / self.V[:, None]
        G[:, ~self.dirs] = 0.0
        return G

    def lap(self, cell: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(sum of gamma dphi/dn |S| over the faces) / V and the same with absolute face contributions."""
        fi = self.coef_i * (cell[self.nei] - cell[self.own[:self.ni]])
        fb = self.coef_b * (b - cell[self.own[self.ni:]])
        q = gamma * np.concatenate([fi, fb])
        return self.sum_faces(q) / self.V, self.sum_faces(q, absolute=True) / self.V

    # ------------------------------------------------------------------ balances over the boundary
    def patch_sums(self, face_q: np.ndarray) -> dict[str, float]:
        """Sum of a face quantity over each non-empty patch. ``face_q`` holds either every face (internal first) or
        only the boundary faces, as the balances pass it."""
        base = 0 if len(face_q) == self.nb else self.ni
        out = {}
        for pt in self.fd.patches:
            if pt["type"] == "empty" or pt["nFaces"] == 0:
                continue
            a = base + pt["startFace"] - self.ni
            out[pt["name"]] = float(np.sum(face_q[a:a + pt["nFaces"]], axis=0).sum())
        return out

    def patch_slice(self, name: str) -> slice | None:
        for pt in self.fd.patches:
            if pt["name"] == name:
                a = pt["startFace"] - self.ni
                return slice(a, a + pt["nFaces"])
        return None


# ---------------------------------------------------------------------------------------------- residuals
def fv_residuals(fv: FV, physics: str, p: dict[str, Any]) -> tuple[list[tuple[str, np.ndarray, list, str]], list[str]]:
    """[(name, residual, terms, description)], reasons. Cell residuals are per unit volume."""
    fd = fv.fd
    out, why = [], []
    if physics == "incompressible-flow":
        if "U" not in fd.fields:
            return [], ["no velocity field U"]
        U = fd.fields["U"]
        Ub = fv.bvals("U", U)
        F = fv.flux(U, Ub, fd.phi)
        div = fv.sum_faces(F) / fv.V
        out.append(("continuity", div, [fv.sum_faces(F, absolute=True) / fv.V], "div u = 0"))
        if "p" not in fd.fields:
            why.append("momentum: no pressure field p")
        elif p.get("nu") is None:
            why.append("momentum: no valid kinematic viscosity nu (missing, or rejected by PC504)")
        else:
            rho = float(p.get("rho") or 1.0)
            Uf = fv.face(U, Ub)
            conv = fv.sum_faces(F[:, None] * Uf) / fv.V[:, None]
            conv_abs = fv.sum_faces(np.abs(F[:, None] * Uf), absolute=True) / fv.V[:, None]
            gp = fv.grad(fd.fields["p"], fv.bvals("p", fd.fields["p"])) / rho
            visc, visc_abs = np.zeros_like(U), np.zeros_like(U)
            for i in range(3):
                visc[:, i], visc_abs[:, i] = fv.lap(U[:, i], Ub[:, i], p["nu"])
            ddt = np.zeros_like(U)
            if fd.previous and "U" in fd.previous["fields"] and fd.time is not None:
                dt = fd.time - fd.previous["time"]
                if dt > 0:
                    ddt = (U - fd.previous["fields"]["U"]) / dt
            R = ddt + conv + gp - visc
            sel = fv.dirs
            out.append(("momentum", R[:, sel], [ddt[:, sel], conv_abs[:, sel], gp[:, sel], visc_abs[:, sel]],
                        "du/dt + div(phi u) + grad p/rho - nu lap u = 0"))
    elif physics == "heat":
        if "T" not in fd.fields:
            return [], ["no temperature field T"]
        T = fd.fields["T"]
        lap, lap_abs = fv.lap(T, fv.bvals("T", T), p["k"])
        q = np.full(fv.nc, float(p.get("q") or 0.0))
        ddt = np.zeros(fv.nc)
        if fd.previous and "T" in fd.previous["fields"] and fd.time is not None:
            dt = fd.time - fd.previous["time"]
            if dt > 0:
                ddt = p["rho_cp"] * (T - fd.previous["fields"]["T"]) / dt
        out.append(("heat", lap + q - ddt, [lap_abs, q, ddt], "rho_cp dT/dt = k lap T + q"))
    elif physics == "elasticity":
        why.append("elasticity residual on OpenFOAM meshes: use the boundary force balance (PC104)")
    return out, why


def fv_mass_balance(fv: FV) -> dict[str, Any] | None:
    fd = fv.fd
    if "U" not in fd.fields and fd.phi is None:
        return None
    U = fd.fields.get("U")
    F = fv.flux(U, fv.bvals("U", U), fd.phi) if U is not None or fd.phi is not None else None
    Fb = F[fv.ni:]
    net, tot = float(Fb.sum()), float(np.abs(Fb).sum())
    area = float(fv.Smag[fv.ni:][~fv.empty].sum())
    uref = float(np.max(np.linalg.norm(U, axis=1))) if U is not None else 0.0
    return {"net": net, "total": tot, "area": area, "uref": uref, "patches": fv.patch_sums(Fb),
            "source": "phi" if fd.phi is not None else "U"}


def fv_heat_balance(fv: FV, p: dict[str, Any]) -> dict[str, Any] | None:
    fd = fv.fd
    if "T" not in fd.fields:
        return None
    T = fd.fields["T"]
    Tb = fv.bvals("T", T)
    qb = p["k"] * fv.coef_b * (Tb - T[fv.own[fv.ni:]])         # heat flow INTO the domain through each face
    src = float(np.sum((p.get("q") or 0.0) * fv.V))
    return {"net": float(qb.sum()) + src, "total": float(np.abs(qb).sum()) + abs(src), "source": src,
            "patches": fv.patch_sums(qb)}
