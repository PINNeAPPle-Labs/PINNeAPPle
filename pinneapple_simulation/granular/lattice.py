"""From a crystal lattice to a material and a particle shape -- the ``Atoms to Material`` chain.

A DEM run needs a Young's modulus, a density and a particle shape.  All three follow from the crystal structure and
the interatomic potential when the material is a simple one, and this module computes them rather than asking for
them:

1. **Lattice constant** -- minimise the cohesive energy per atom over the cubic lattice constant ``a``;
2. **Elastic constants** ``C11``, ``C12``, ``C44`` -- second derivatives of the energy density under homogeneous
   strain.  For a monatomic Bravais lattice the affine (homogeneous) deformation is the exact response, so no
   internal relaxation is missing;
3. **Polycrystal moduli** -- Voigt / Reuss bounds and their Hill average, then ``E`` and ``nu``;
4. **Crystal shape** -- surface energies per facet from the broken-bond count and a Wulff construction, which
   returns a :class:`~pinneapple_simulation.granular.shapes.ConvexPolyhedron` that the contact model uses directly.

Two physics checks come for free and are asserted in the tests: a central-force potential at zero stress obeys the
**Cauchy relation** ``C12 = C44`` (Born & Huang, *Dynamical Theory of Crystal Lattices*), and the elastic constants
must satisfy the **Born stability** inequalities (Born 1940, Math. Proc. Cambridge Philos. Soc. 36:160-172).  A
potential or a structure that fails either is reported, not smoothed over.

    >>> from pinneapple_simulation.granular.lattice import Crystal, LennardJones
    >>> crystal = Crystal("fcc", LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0))
    >>> a = crystal.relaxed_lattice_constant()
    >>> 1.0 < a < 1.3
    True
    >>> constants = crystal.elastic_constants(a)
    >>> bool(constants["cauchy_rel_error"] < 1e-3)
    True

Units are reduced: ``sigma`` for lengths, ``epsilon`` for energies, so elastic constants come out in
``epsilon / sigma^3``.  :func:`material_from_lattice` rescales them to SI with a ``sigma`` in metres and an
``epsilon`` in joules, which is all a real material needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contacts import Material
from .shapes import ConvexPolyhedron

__all__ = [
    "LennardJones",
    "Crystal",
    "material_from_lattice",
    "cubic_polycrystal_moduli",
    "wulff_polyhedron",
]

_BASIS = {
    "sc": ((0.0, 0.0, 0.0),),
    "bcc": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    "fcc": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)),
    "diamond": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
                (0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)),
}
_ATOMS_PER_CELL = {name: len(basis) for name, basis in _BASIS.items()}


@dataclass
class LennardJones:
    """``4 epsilon [(sigma/r)^12 - (sigma/r)^6]`` truncated at ``cutoff * sigma``.

    The truncation is the shifted-force form by default: the energy *and* the force are made continuous at the
    cutoff (``phi(r) - phi(rc) - (r - rc) phi'(rc)``).  A plain truncation leaves a step of ~5e-3 epsilon in the
    energy, and a second derivative of a discontinuous function is noise -- which shows up as a broken Cauchy
    relation and meaningless elastic constants, so it is not an option for this module.
    """

    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 3.0
    shifted_force: bool = True

    def _bare(self, r: np.ndarray) -> np.ndarray:
        sr = self.sigma / np.maximum(r, 1.0e-12)
        s6 = sr ** 6
        return 4.0 * self.epsilon * (s6 * s6 - s6)

    def _bare_derivative(self, r: np.ndarray) -> np.ndarray:
        sr = self.sigma / np.maximum(r, 1.0e-12)
        s6 = sr ** 6
        return 24.0 * self.epsilon * (s6 * s6 - 2.0 * s6) / np.maximum(r, 1.0e-12)

    def energy(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        rc = self.cutoff * self.sigma
        e = self._bare(r)
        if self.shifted_force:
            e = e - self._bare(np.array(rc)) - (r - rc) * self._bare_derivative(np.array(rc))
        return np.where(r > rc, 0.0, e)

    def __call__(self, r) -> np.ndarray:  # pragma: no cover - convenience
        return self.energy(r)


@dataclass
class Crystal:
    """A cubic crystal structure with a pair potential."""

    structure: str = "fcc"
    potential: LennardJones = field(default_factory=LennardJones)
    atomic_mass: float = 1.0                 # [amu] or any consistent unit; only used for the density

    def __post_init__(self) -> None:
        if self.structure not in _BASIS:
            raise ValueError(f"unknown structure {self.structure!r}; available: {sorted(_BASIS)}")

    # ---- geometry ---------------------------------------------------------------------------------------
    @property
    def atoms_per_cell(self) -> int:
        return _ATOMS_PER_CELL[self.structure]

    def basis(self) -> np.ndarray:
        return np.array(_BASIS[self.structure], dtype=float)

    def _sites(self, a: float, radius: float) -> np.ndarray:
        """All basis atoms of the lattice inside ``radius`` of the origin (cluster, no periodicity needed)."""
        n = int(math.ceil(radius / a)) + 2
        grid = np.stack(np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), np.arange(-n, n + 1),
                                    indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
        basis = (grid[:, None, :] + self.basis()[None, :, :]).reshape(-1, 3) * a
        keep = np.linalg.norm(basis, axis=1) <= radius
        site = basis[keep]
        return site if len(site) else np.zeros((1, 3))

    def neighbor_vectors(self, a: float, margin: float = 1.0) -> np.ndarray:
        """Bond vectors from one atom to every neighbour inside the potential cutoff (each direction once)."""
        reach = self.potential.cutoff * self.potential.sigma * margin
        sites = self._sites(a, reach)
        keep = np.linalg.norm(sites, axis=1) <= reach + 1.0e-9
        vecs = sites[keep]
        rounded = np.unique(np.round(vecs, 10), axis=0)
        return rounded[np.linalg.norm(rounded, axis=1) > 1.0e-9]

    def cell_volume(self, a: float) -> float:
        return a ** 3

    def atomic_volume(self, a: float) -> float:
        return self.cell_volume(a) / self.atoms_per_cell

    # ---- energetics -------------------------------------------------------------------------------------
    def energy_per_atom(self, a: float, vectors: np.ndarray | None = None,
                        deformation: np.ndarray | None = None) -> float:
        """Cohesive energy per atom of the (possibly homogeneously strained) lattice."""
        vecs = self.neighbor_vectors(a) if vectors is None else vectors
        if deformation is not None:
            vecs = vecs @ np.asarray(deformation, dtype=float).T
        r = np.linalg.norm(vecs, axis=1)
        return 0.5 * float(np.sum(self.potential.energy(r)))

    def relaxed_lattice_constant(self, a_lo: float = 0.8, a_hi: float = 1.6, tol: float = 1.0e-10) -> float:
        """Lattice constant that minimises the cohesive energy (golden-section search on ``a``)."""
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = a_lo * self.potential.sigma, a_hi * self.potential.sigma
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = self.energy_per_atom(c), self.energy_per_atom(d)
        for _ in range(200):
            if abs(hi - lo) < tol:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = self.energy_per_atom(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = self.energy_per_atom(d)
        return 0.5 * (lo + hi)

    def _energy_density(self, a: float, vectors: np.ndarray, strain: np.ndarray) -> float:
        deformation = np.eye(3) + np.asarray(strain, dtype=float)
        return self.energy_per_atom(a, vectors, deformation) / self.atomic_volume(a)

    def elastic_constants(self, a: float | None = None, step: float = 2.0e-3) -> dict:
        """``C11``, ``C12``, ``C44`` from central differences of the energy density, plus their physical checks."""
        a = self.relaxed_lattice_constant() if a is None else float(a)
        vectors = self.neighbor_vectors(a, margin=1.05)            # margin so a strained bond stays inside
        zero = np.zeros((3, 3))

        def second_derivative(direction: np.ndarray) -> float:
            plus = self._energy_density(a, vectors, step * direction)
            minus = self._energy_density(a, vectors, -step * direction)
            return (plus - 2.0 * self._energy_density(a, vectors, zero) + minus) / step ** 2

        # the probe is the symmetric strain tensor e (so ``shear`` means e_xy = e_yx = s/2); the energy density is
        # 1/2 C11 e_xx^2 for uniaxial, (C11 + C12) h^2 for biaxial e_xx = e_yy = h, and 1/2 C44 s^2 for shear
        uniaxial = np.diag([1.0, 0.0, 0.0])
        biaxial = np.diag([1.0, 1.0, 0.0])
        shear = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        c11 = second_derivative(uniaxial)
        c12 = second_derivative(biaxial) / 2.0 - c11
        c44 = second_derivative(shear)
        born = {
            "C11_minus_C12": c11 - c12,
            "C11_plus_2C12": c11 + 2.0 * c12,
            "C44": c44,
            "stable": bool(c11 - c12 > 0.0 and c11 + 2.0 * c12 > 0.0 and c44 > 0.0),
        }
        scale = max(abs(c11), abs(c12), abs(c44), 1.0e-30)
        return {
            "lattice_constant": a,
            "C11": c11,
            "C12": c12,
            "C44": c44,
            "bulk_modulus": (c11 + 2.0 * c12) / 3.0,
            "cauchy_rel_error": abs(c12 - c44) / scale,
            "born_stability": born,
            "energy_per_atom": self.energy_per_atom(a),
            "atomic_volume": self.atomic_volume(a),
        }

    # ---- surfaces and shape -----------------------------------------------------------------------------
    def surface_energy(self, normal, a: float | None = None) -> float:
        """Broken-bond surface energy for the facet with the given normal [energy / area]."""
        a = self.relaxed_lattice_constant() if a is None else float(a)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        vectors = self.neighbor_vectors(a)
        projection = vectors @ n
        crossing = projection > 1.0e-12
        if not np.any(crossing):
            raise ValueError(f"no bond crosses the facet {normal}: the broken-bond count is zero")
        bond_energy = self.potential.energy(np.linalg.norm(vectors[crossing], axis=1))
        return float(-np.sum(projection[crossing] * bond_energy)) / (2.0 * self.atomic_volume(a))

    def facet_energies(self, normals=None, a: float | None = None) -> dict:
        a = self.relaxed_lattice_constant() if a is None else float(a)
        normals = normals or [(1, 1, 1), (1, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 1)]
        return {tuple(int(x) for x in hkl): self.surface_energy(hkl, a) for hkl in normals}

    def wulff_shape(self, a: float | None = None, facets=None) -> ConvexPolyhedron:
        """The equilibrium crystal shape: the Wulff construction over the facet surface energies."""
        energies = self.facet_energies(a=a, normals=facets)
        return wulff_polyhedron(energies)


def wulff_polyhedron(facet_energies: dict) -> ConvexPolyhedron:
    """Wulff construction: the polytope ``{x : n . x <= gamma(n)}`` for the given facets.

    Wulff (1901, Z. Kristallogr. 34:449-530): the equilibrium shape of a crystal is the intersection of half-spaces
    placed at a distance proportional to each facet's surface energy.  Facets with a low energy therefore grow and
    the high-energy ones are cut away -- which is why an fcc metal is octahedral at ``(111)`` and not a cube.
    """
    from .overlap import polytope_from_halfspaces

    if not facet_energies:
        raise ValueError("wulff_polyhedron needs at least one facet")
    normals, offsets = [], []
    for hkl, gamma in facet_energies.items():
        n = np.asarray(hkl, dtype=float)
        n = n / np.linalg.norm(n)
        normals.append(n)
        offsets.append(float(gamma))
        # the symmetry-related facets: every sign combination and every axis permutation of the same family
        for perm in ((1, 2, 0), (2, 0, 1)):
            p = n[list(perm)]
            for signs in _signs(3):
                variant = p * np.array(signs)
                if not any(np.allclose(variant, v, atol=1e-12) for v in normals):
                    normals.append(variant)
                    offsets.append(float(gamma))
    verts, faces = polytope_from_halfspaces(np.array(normals), np.array(offsets))
    return ConvexPolyhedron(verts, faces)


def _signs(n: int):
    from itertools import product

    return [s for s in product((1.0, -1.0), repeat=n)]


def cubic_polycrystal_moduli(C11: float, C12: float, C44: float) -> dict:
    """Voigt / Reuss / Hill moduli of a random cubic polycrystal (Voigt 1928; Reuss 1929; Hill 1952)."""
    k_v = (C11 + 2.0 * C12) / 3.0
    g_v = (C11 - C12 + 3.0 * C44) / 5.0
    s11 = (C11 + C12) / ((C11 - C12) * (C11 + 2.0 * C12))
    s12 = -C12 / ((C11 - C12) * (C11 + 2.0 * C12))
    s44 = 1.0 / C44
    k_r = 1.0 / (3.0 * (s11 + 2.0 * s12))
    g_r = 1.0 / ((4.0 / 5.0) * (s11 - s12) + (3.0 / 5.0) * s44)
    k_h, g_h = 0.5 * (k_v + k_r), 0.5 * (g_v + g_r)
    e = 9.0 * k_h * g_h / (3.0 * k_h + g_h)
    nu = (3.0 * k_h - 2.0 * g_h) / (2.0 * (3.0 * k_h + g_h))
    return {
        "K_voigt": k_v, "G_voigt": g_v, "K_reuss": k_r, "G_reuss": g_r,
        "K_hill": k_h, "G_hill": g_h, "E": e, "nu": nu,
        "voigt_reuss_gap": (g_v - g_r) / g_h if g_h else 0.0,
    }


def material_from_lattice(crystal: Crystal, sigma_m: float = 3.40e-10, epsilon_J: float = 1.0e-21,
                          atomic_mass_kg: float = 6.63e-26, a: float | None = None,
                          facet_normals=None) -> dict:
    """Rescale a reduced-unit crystal into SI and return a DEM :class:`Material` plus a particle shape.

    ``sigma_m`` and ``epsilon_J`` set the length and energy scale of the potential (for argon, 3.40e-10 m and
    1.66e-21 J respectively); ``atomic_mass_kg`` sets the density.  Nothing else is fitted.
    """
    constants = crystal.elastic_constants(a)
    moduli = cubic_polycrystal_moduli(constants["C11"], constants["C12"], constants["C44"])
    scale = epsilon_J / sigma_m ** 3
    density = atomic_mass_kg / (constants["atomic_volume"] * sigma_m ** 3)
    material = Material(E=moduli["E"] * scale, nu=moduli["nu"], density=density)
    return {
        "material": material,
        "shape": crystal.wulff_shape(a, facets=facet_normals),
        "elastic_constants": {k: constants[k] * scale for k in ("C11", "C12", "C44", "bulk_modulus")},
        "moduli": {k: (moduli[k] * scale if k in ("K_voigt", "G_voigt", "K_reuss", "G_reuss", "K_hill", "G_hill",
                                                  "E") else moduli[k])
                   for k in moduli},
        "density": density,
        "lattice_constant_m": constants["lattice_constant"] * sigma_m,
        "born_stable": constants["born_stability"]["stable"],
        "cauchy_rel_error": constants["cauchy_rel_error"],
    }
