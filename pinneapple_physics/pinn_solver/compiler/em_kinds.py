"""Electromagnetic PDE kinds of the problem compiler: Maxwell's equations in differential and integral form.

Every kind returns a list of residual tensors ``(N, 1)`` at the collocation points (the compiler squares and
averages them). Units are the caller's: ``eps``, ``mu``, ``sigma`` default to 1 (a non-dimensional problem); for
SI pass the SI values or, better, scale the problem (``x / L``, ``t c / L``, fields by a reference amplitude) so the
network sees O(1) numbers. Space- or field-dependent material maps come from ``ctx`` callables
``eps_fn(X, ctx)``, ``mu_fn``, ``nu_fn``, ``sigma_fn`` evaluated at the points (piecewise-constant maps are the
usual case; a preset may put them in the PDE ``params``, under the caller's ``ctx``; in the differential kinds
their jumps are not differentiated, so interface conditions are only
imposed by the integral kinds, which evaluate the maps on the edges of each control square).

Differential kinds (``fields`` in brackets):

``electrostatics`` [phi]               div(eps grad phi) + rho = 0, rho from ``ctx["rho_fn"]`` (or ``source_fn``)
``magnetostatics_az`` [Az]             div(nu grad Az) + Jz = 0, ``nu = 1/mu``, Jz from ``ctx["J_fn"]``
``eddy_current_az`` [Ar, Ai]           nu lap(A) - j omega sigma A + Js = 0 (phasor ``exp(+j omega t)``, A = Ar + j Ai)
``maxwell_tm`` [Ez, Hx, Hy] + t        eps dEz/dt - (dHy/dx - dHx/dy) + sigma Ez + Jz = 0, mu dHx/dt + dEz/dy = 0,
                                       mu dHy/dt - dEz/dx = 0 (2-D TMz; ``maxwell_te`` is the dual, in compile.py)
``maxwell_3d`` [Ex, Ey, Ez, Hx, Hy, Hz] + t
                                       eps dE/dt - curl H + sigma E + J = 0, mu dH/dt + curl E = 0; with
                                       ``params["gauss"] = True`` also div(eps E) - rho and div(mu H)
``maxwell_tm_frequency`` [Er, Ei]      lap(E) + k0^2 eps_r E = f for the phasor Ez = Er + j Ei; complex ``eps_r``
                                       from ``ctx["eps_fn"]`` (scatterers) or ``params["eps_r"]``, ``params["eps_i"]``

Integral kinds (2-D, each collocation point is the centre of a square control volume of half-width
``params["h"]``; Gauss-Legendre quadrature with ``params["quad"]`` points per edge, default 4; residuals divided by
the area so they have the units of the differential form and converge to it as ``h -> 0``):

``electrostatics_integral`` [phi]      oint eps grad(phi).n dl + int rho dA = 0 (Gauss: oint D.n = Q_enclosed)
``maxwell_tm_integral`` [Ez, Hx, Hy] + t
                                       oint H.dl = int (eps dEz/dt + sigma Ez + Jz) dA (Ampere-Maxwell) and, along x
                                       and y, the two Faraday line laws mu d/dt int Hy dx = Ez(x+h) - Ez(x-h) and
                                       mu d/dt int Hx dy = -(Ez(y+h) - Ez(y-h)) (per unit length in z)

References: Jackson, *Classical Electrodynamics* 3rd ed., sec. 6.1 (Maxwell's equations) and I.3 of the
introduction (integral forms); Jin, *The Finite Element Method in Electromagnetics* 3rd ed., ch. 4 (the 2-D
scalar formulations used by the static, eddy-current and TM-frequency kinds).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from .autograd_ops import ensure_tensor, grad, laplacian, time_derivative

__all__ = ["EM_KINDS", "em_residuals"]

EM_KINDS = frozenset({
    "electrostatics", "magnetostatics_az", "eddy_current_az", "maxwell_tm", "maxwell_3d", "maxwell_tm_frequency",
    "electrostatics_integral", "maxwell_tm_integral",
})


def _need(fields: dict, names: Sequence[str], kind: str) -> list[torch.Tensor]:
    for n in names:
        if n not in fields:
            raise ValueError(f"{kind} expects field '{n}' (fields: {sorted(fields)})")
    return [fields[n] for n in names]


def _at(ctx: dict, key: str, X: torch.Tensor, default, like: torch.Tensor, *alts: str) -> torch.Tensor:
    """A material map or source from ``ctx[key](X_numpy, ctx)`` (or an alternative key), else the default."""
    fn = ctx.get(key)
    for a in alts:
        fn = fn or ctx.get(a)
    if fn is None:
        return torch.full_like(like, float(default))
    v = fn(X.detach().cpu().numpy(), ctx)
    t = torch.as_tensor(np.asarray(v), device=like.device)
    if torch.is_complex(t):
        return t.reshape(-1, 1)
    return t.to(like.dtype).reshape(-1, 1)


def _idx(coords: Sequence[str], name: str) -> int:
    if name not in coords:
        raise ValueError(f"coordinate '{name}' missing (coords: {tuple(coords)})")
    return list(coords).index(name)


def _d(u: torch.Tensor, x: torch.Tensor, i: int) -> torch.Tensor:
    return grad(u, x)[:, i:i + 1]


def _spatial(coords: Sequence[str]) -> list[int]:
    return [i for i, c in enumerate(coords) if c != "t"]


def _gauss_legendre(n: int, device, dtype):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.as_tensor(x, device=device, dtype=dtype), torch.as_tensor(w, device=device, dtype=dtype)


def em_residuals(kind: str, *, fields: dict[str, torch.Tensor], xcol: torch.Tensor, coords: Sequence[str],
                 params: dict[str, Any], ctx: dict[str, Any], model: Callable, field_names: Sequence[str],
                 split_fields: Callable) -> list[torch.Tensor]:
    p = params
    # callables in the PDE params (a preset's material maps and sources) are defaults under the caller's ctx
    ctx = {**{k: v for k, v in p.items() if callable(v)}, **(ctx or {})}
    sp = _spatial(coords)
    if kind == "electrostatics":
        (phi,) = _need(fields, field_names[:1], kind)
        eps = _at(ctx, "eps_fn", xcol, p.get("eps", 1.0), phi)
        rho = _at(ctx, "rho_fn", xcol, 0.0, phi, "source_fn")
        return [eps * laplacian(phi, xcol, sp) + rho]

    if kind == "magnetostatics_az":
        (A,) = _need(fields, field_names[:1], kind)
        nu_default = p.get("nu", 1.0 / float(p.get("mu", 1.0)))
        nu = _at(ctx, "nu_fn", xcol, nu_default, A)
        J = _at(ctx, "J_fn", xcol, 0.0, A, "source_fn")
        return [nu * laplacian(A, xcol, sp) + J]

    if kind == "eddy_current_az":
        Ar, Ai = _need(fields, ("Ar", "Ai"), kind)
        omega = float(p.get("omega", 1.0))
        nu = _at(ctx, "nu_fn", xcol, p.get("nu", 1.0 / float(p.get("mu", 1.0))), Ar)
        sig = _at(ctx, "sigma_fn", xcol, p.get("sigma", 1.0), Ar)
        Js = _at(ctx, "J_fn", xcol, 0.0, Ar, "source_fn")
        return [nu * laplacian(Ar, xcol, sp) + omega * sig * Ai + Js,
                nu * laplacian(Ai, xcol, sp) - omega * sig * Ar]

    if kind == "maxwell_tm":
        Ez, Hx, Hy = _need(fields, ("Ez", "Hx", "Hy"), kind)
        ti, xi, yi = _idx(coords, "t"), _idx(coords, "x"), _idx(coords, "y")
        eps = _at(ctx, "eps_fn", xcol, p.get("eps", p.get("epsilon", 1.0)), Ez)
        mu = _at(ctx, "mu_fn", xcol, p.get("mu", 1.0), Ez)
        sig = _at(ctx, "sigma_fn", xcol, p.get("sigma", 0.0), Ez)
        Jz = _at(ctx, "J_fn", xcol, 0.0, Ez, "source_fn")
        gE = grad(Ez, xcol)
        return [eps * time_derivative(Ez, xcol, ti) - (_d(Hy, xcol, xi) - _d(Hx, xcol, yi)) + sig * Ez + Jz,
                mu * time_derivative(Hx, xcol, ti) + gE[:, yi:yi + 1],
                mu * time_derivative(Hy, xcol, ti) - gE[:, xi:xi + 1]]

    if kind == "maxwell_3d":
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        E = _need(fields, names[:3], kind)
        H = _need(fields, names[3:], kind)
        ti = _idx(coords, "t")
        ix = [_idx(coords, c) for c in "xyz"]
        eps = _at(ctx, "eps_fn", xcol, p.get("eps", 1.0), E[0])
        mu = _at(ctx, "mu_fn", xcol, p.get("mu", 1.0), E[0])
        sig = _at(ctx, "sigma_fn", xcol, p.get("sigma", 0.0), E[0])
        J = [_at(ctx, f"J{c}_fn", xcol, 0.0, E[0]) for c in "xyz"]
        gE = [grad(e, xcol) for e in E]
        gH = [grad(h, xcol) for h in H]

        def curl(g):
            return [g[2][:, ix[1]:ix[1] + 1] - g[1][:, ix[2]:ix[2] + 1],
                    g[0][:, ix[2]:ix[2] + 1] - g[2][:, ix[0]:ix[0] + 1],
                    g[1][:, ix[0]:ix[0] + 1] - g[0][:, ix[1]:ix[1] + 1]]

        cH, cE = curl(gH), curl(gE)
        res = [eps * time_derivative(E[k], xcol, ti) - cH[k] + sig * E[k] + J[k] for k in range(3)]
        res += [mu * time_derivative(H[k], xcol, ti) + cE[k] for k in range(3)]
        if p.get("gauss", False):
            rho = _at(ctx, "rho_fn", xcol, 0.0, E[0])
            res.append(eps * sum(gE[k][:, ix[k]:ix[k] + 1] for k in range(3)) - rho)
            res.append(mu * sum(gH[k][:, ix[k]:ix[k] + 1] for k in range(3)))
        return res

    if kind == "maxwell_tm_frequency":
        Er, Ei = _need(fields, ("Er", "Ei"), kind)
        k0 = float(p.get("k0", 1.0))
        if ctx.get("eps_fn") is not None:
            e = ctx["eps_fn"](xcol.detach().cpu().numpy(), ctx)
            e = np.asarray(e, complex).reshape(-1, 1)
            er = torch.as_tensor(e.real, device=Er.device, dtype=Er.dtype)
            ei = torch.as_tensor(e.imag, device=Er.device, dtype=Er.dtype)
        else:
            er = torch.full_like(Er, float(p.get("eps_r", 1.0)))
            ei = torch.full_like(Er, float(p.get("eps_i", 0.0)))
        fr = torch.zeros_like(Er)
        fi = torch.zeros_like(Er)
        if ctx.get("source_fn") is not None:
            f = np.asarray(ctx["source_fn"](xcol.detach().cpu().numpy(), ctx), complex).reshape(-1, 1)
            fr = torch.as_tensor(f.real, device=Er.device, dtype=Er.dtype)
            fi = torch.as_tensor(f.imag, device=Er.device, dtype=Er.dtype)
        k2 = k0 * k0
        return [laplacian(Er, xcol, sp) + k2 * (er * Er - ei * Ei) - fr,
                laplacian(Ei, xcol, sp) + k2 * (er * Ei + ei * Er) - fi]

    if kind in ("electrostatics_integral", "maxwell_tm_integral"):
        return _integral(kind, xcol=xcol, coords=coords, p=p, ctx=ctx, model=model, field_names=field_names,
                         split_fields=split_fields)
    raise ValueError(f"unknown electromagnetic kind {kind!r}")


def _integral(kind, *, xcol, coords, p, ctx, model, field_names, split_fields):
    """Square control volumes ``[x - h, x + h] x [y - h, y + h]`` centred at the collocation points."""
    h = float(p.get("h", 0.02))
    nq = int(p.get("quad", 4))
    xi, yi = _idx(coords, "x"), _idx(coords, "y")
    s, w = _gauss_legendre(nq, xcol.device, xcol.dtype)
    N = xcol.shape[0]
    ex = torch.zeros(xcol.shape[1], device=xcol.device, dtype=xcol.dtype)
    ex[xi] = 1.0
    ey = torch.zeros_like(ex)
    ey[yi] = 1.0

    def at(offsets: torch.Tensor):
        """Model fields at ``xcol + offset`` for every offset (Q, d): returns (pts (N*Q, d), fields)."""
        pts = (xcol[:, None, :] + offsets[None, :, :]).reshape(-1, xcol.shape[1])
        return pts, split_fields(ensure_tensor(model(pts)), list(field_names))

    def edge(fixed: torch.Tensor, along: torch.Tensor):
        """Quadrature points on the edge ``fixed * h + s * h * along`` (Q, d)."""
        return h * (fixed[None, :] + s[:, None] * along[None, :])

    def edge_sum(v: torch.Tensor) -> torch.Tensor:
        """Sum over the quadrature points of each edge with the weights (length element h)."""
        return h * (v.reshape(N, nq) * w[None, :]).sum(1, keepdim=True)

    if kind == "electrostatics_integral":
        # outward flux of D = -eps grad(phi) through the four edges + enclosed charge = 0, i.e.
        # oint eps dphi/dn dl + int rho dA = 0
        total = 0.0
        for normal, along, k in ((ex, ey, xi), (-ex, ey, xi), (ey, ex, yi), (-ey, ex, yi)):
            pts, f = at(edge(normal, along))
            phi = f[field_names[0]]
            eps = _at(ctx, "eps_fn", pts, p.get("eps", 1.0), phi)
            dn = float(normal[k]) * grad(phi, pts)[:, k:k + 1]
            total = total + edge_sum(eps * dn)
        # area integral of rho by the tensor rule
        off = h * (s[:, None, None] * ex + s[None, :, None] * ey).reshape(-1, xcol.shape[1])
        pts = (xcol[:, None, :] + off[None]).reshape(-1, xcol.shape[1])
        rho = _at(ctx, "rho_fn", pts, 0.0, pts[:, :1], "source_fn")
        ww = (w[:, None] * w[None, :]).reshape(-1)
        q = (h * h) * (rho.reshape(N, -1) * ww[None]).sum(1, keepdim=True)
        return [(total + q) / (4.0 * h * h)]

    ti = _idx(coords, "t")
    # Ampere-Maxwell: oint H.dl (counter-clockwise) = int (eps dEz/dt + sigma Ez + Jz) dA
    circ = 0.0
    for fixed, along, comp, sign in ((-ey, ex, "Hx", 1.0), (ex, ey, "Hy", 1.0), (ey, ex, "Hx", -1.0),
                                     (-ex, ey, "Hy", -1.0)):
        _, f = at(edge(fixed, along))
        circ = circ + sign * edge_sum(f[comp])
    off = h * (s[:, None, None] * ex + s[None, :, None] * ey).reshape(-1, xcol.shape[1])
    pts, f = at(off)
    Ez = f["Ez"]
    eps = _at(ctx, "eps_fn", pts, p.get("eps", 1.0), Ez)
    sig = _at(ctx, "sigma_fn", pts, p.get("sigma", 0.0), Ez)
    Jz = _at(ctx, "J_fn", pts, 0.0, Ez, "source_fn")
    src = eps * time_derivative(Ez, pts, ti) + sig * Ez + Jz
    ww = (w[:, None] * w[None, :]).reshape(-1)
    area = (h * h) * (src.reshape(N, -1) * ww[None]).sum(1, keepdim=True)
    res = [(circ - area) / (4.0 * h * h)]
    # Faraday along x: mu d/dt int_{x-h}^{x+h} Hy dx = Ez(x + h) - Ez(x - h); along y with Hx and a minus sign
    for along, comp, sign in ((ex, "Hy", 1.0), (ey, "Hx", -1.0)):
        pts, f = at(h * s[:, None] * along[None, :])
        mu = _at(ctx, "mu_fn", pts, p.get("mu", 1.0), f[comp])
        flux_rate = edge_sum(mu * time_derivative(f[comp], pts, ti))
        _, fp = at((h * along)[None, :])
        _, fm = at((-h * along)[None, :])
        res.append((flux_rate - sign * (fp["Ez"] - fm["Ez"])) / (2.0 * h))
    return res
