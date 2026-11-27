"""The five design spaces as three-dimensional problems.

The first five families (:mod:`pinneapple_neural.lpm_families`) are 2-D: a section, a meridian, a plate. The problems
an industrial surrogate is asked for are not — a wing carries a spanwise load, a heat sink has a fin array, a bracket
is a solid. This module keeps the same reference for each physics and answers it in three dimensions.

What each family's reference is, and what it is not, is stated per family and repeated in every card. The one that
matters most: no family here solves the 3-D compressible Navier-Stokes. ``pp_wing3d`` is a 3-D lifting surface with
the Prandtl-Glauert compressibility rule; it produces a pressure field with a spanwise load and a swept, tapered,
twisted planform, but it does not resolve a shock, so its transonic answer is the linearised one and the card says so.
A 3-D transonic field needs a body-fitted compressible solver, which is a data-generation project of its own.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_design.aero import vlm
from pinneapple_design.aero.airframe import Airframe
from pinneapple_design.aero.geometry import BOUNDS as SECTION_BOUNDS
from pinneapple_design.aero.geometry import latin_hypercube, naca4, valid
from pinneapple_design.aero.panel import airfoil_nodes, panel_solve

from .lpm_pipeline import DesignSpec, FieldFamily, Reference

__all__ = ["Bracket3D", "FinHeatSink3D", "Wing3D"]

#: chordwise panels per side in the section solve: 40 is enough for a section Cp to 1 % (see tests)
N_PER_SIDE = 40
#: spanwise strips of the lifting surface; the load varies along the span, the section solve does not
N_STRIPS = 12
#: fins across the heat sink, fixed: the point cloud of a design has to line up with every other design
N_FINS = 7


def _station_fracs() -> np.ndarray:
    """Where along the semi-span the surface strips sit: the lattice's own cosine spacing, not a uniform one.

    The vortex lattice clusters its strips toward the tips (``ye = -b/2 cos(theta)``), and each strip's lift belongs at
    that strip's centre. Sampling the surface uniformly and putting the k-th strip's lift on the k-th uniform station
    would smear the load by up to half a strip width, which at the tip is most of the load.
    """
    th = np.linspace(0.0, math.pi, 2 * N_STRIPS + 1)
    ye = -np.cos(th)
    centre = 0.5 * (ye[:-1] + ye[1:])[N_STRIPS:]                 # the right half, root to tip
    return np.abs(centre)


def _beta(mach: float) -> float:
    """Prandtl-Glauert factor, ``sqrt(1 - M^2)``."""
    return math.sqrt(max(1.0 - mach * mach, 1e-6))


class Wing3D(FieldFamily):
    """Surface Cp over a swept, tapered, twisted wing at a compressible free stream.

    The reference is a **3-D vortex lattice** (Horseshoe vortices on the camber surface, flow tangency at the panel
    control points, Kutta condition by construction) coupled strip by strip to the **2-D Hess-Smith panel solution** of
    the wing's own section. The lattice gives the spanwise load of the planform; the section solution gives the shape
    of the pressure distribution that carries it; Prandtl-Glauert carries both to the free-stream Mach number.

    The coupling is what makes the answer a field rather than a number: the section is solved at the incidence that
    reproduces the lattice's local lift, so a tip that unloads because of sweep unloads in the pressure too.

    Assumptions the card repeats: attached potential flow, linearised compressibility (no shock), no boundary layer,
    no viscous drag, no aeroelastic deflection. Valid to the drag-divergence Mach number, above which the wave drag the
    lattice does not contain is the dominant term.
    """

    name = "pp_wing3d"
    title = "Swept wing at a compressible free stream: surface Cp in three dimensions"
    domain = "aerodynamics / 3-D wing, transonic-limited potential flow"
    dim = 3
    field_names = ("cp",)
    field_units = ("-",)
    field_scale = (1.0,)
    characteristic_length_m = 1.0
    n_points = N_STRIPS * 2 * N_PER_SIDE
    tolerance = 0.10
    geometry_params = {"aspect_ratio": (4.0, 10.0), "taper": (0.30, 1.0), "sweep_le_deg": (0.0, 35.0),
                       "dihedral_deg": (-3.0, 5.0), "twist_deg": (-5.0, 0.0), "incidence_deg": (0.0, 3.0)}
    operating_params = {"alpha_deg": (-2.0, 8.0), "mach": (0.20, 0.82)}
    reference = Reference(
        name="3-D vortex lattice on the camber surface, strip-coupled to the 2-D Hess-Smith panel solution, "
             "Prandtl-Glauert compressibility",
        citation=("Hess & Smith (1967); Moran, An Introduction to Theoretical and Computational Aerodynamics (1984) "
                  "ch. 4 and 7; Katz & Plotkin, Low-Speed Aerodynamics (2001) ch. 12 (vortex lattice); "
                  "Prandtl (1927) / Glauert (1928) compressibility rule"),
        assumptions=("attached potential flow: no separation, no shock, no boundary layer, no viscous drag",
                     "Prandtl-Glauert is the linearised compressible rule, so the transonic answer is the "
                     "subsonic one stretched by 1/beta; the drag divergence the Korn rule would flag is not in the "
                     "field",
                     "the section solve is 2-D and sees only the local lift the lattice gives it, so spanwise flow "
                     "inside the boundary layer is absent",
                     "rigid wing: no aeroelastic twist", "chord 1 m, free stream 1 m/s, incompressible section"),
        implements="solver",
    )

    # ------------------------------------------------------------------------------------------------ the design
    def sample_geometry(self, rng: np.random.Generator) -> dict[str, float]:
        """Planform parameters plus a CST section inside the validation bounds of the aero library."""
        out = {k: float(rng.uniform(*b)) for k, b in self.geometry_params.items()}
        for _ in range(50):
            shape = latin_hypercube(1, SECTION_BOUNDS, rng)[0]
            if valid(shape)[0]:
                out.update({f"cst_{i}": float(v) for i, v in enumerate(shape)})
                return out
        raise RuntimeError("no valid CST section drawn in 50 attempts")

    def _airframe(self, spec: DesignSpec) -> Airframe:
        g = spec.geometry
        shape = np.array([g[f"cst_{i}"] for i in range(len(SECTION_BOUNDS))])
        return Airframe(wing_area=16.0, aspect_ratio=g["aspect_ratio"], taper=g["taper"],
                        sweep_le=g["sweep_le_deg"], dihedral=g["dihedral_deg"], twist=g["twist_deg"],
                        incidence=g["incidence_deg"], section=shape, gear=False)

    def _section(self, spec: DesignSpec) -> np.ndarray:
        g = spec.geometry
        return np.array([g[f"cst_{i}"] for i in range(len(SECTION_BOUNDS))])

    @staticmethod
    def _station(af: Airframe, frac: float) -> tuple[float, float, float]:
        """Leading edge x, chord and z of the wing at ``frac`` of the semi-span (0 root, 1 tip)."""
        y = frac * af.span / 2.0
        chord = af.root_chord * (1.0 - (1.0 - af.taper) * frac)
        x_le = af.wing_x + y * math.tan(math.radians(af.sweep_le))
        z = af.wing_z_root() + y * math.tan(math.radians(af.dihedral))
        return x_le, chord, z

    # -------------------------------------------------------------------------------------------- the geometry
    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """The wing surface as ``N_STRIPS`` sections of the real thickness, outward normals in the section plane."""
        af = self._airframe(spec)
        shape = self._section(spec)
        nodes = airfoil_nodes(shape, N_PER_SIDE, close_te=True)
        # ``close_te`` closes the ring by repeating the first node, so the loop is the nodes without that copy: 2 *
        # N_PER_SIDE segments, which is exactly the number of panels the section solve returns a Cp for
        ring = nodes[:-1] if np.allclose(nodes[0], nodes[-1]) else nodes
        following = np.roll(ring, -1, axis=0)
        mid = 0.5 * (ring + following)
        tangent = following - ring
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        area = 0.5 * np.sum(ring[:, 0] * following[:, 1] - following[:, 0] * ring[:, 1])
        if area < 0.0:
            normal = -normal
        points, normals = [], []
        for frac in _station_fracs():
            x_le, chord, z = self._station(af, frac)
            y = frac * af.span / 2.0
            angle = math.radians(af.incidence + af.twist * frac)      # washout: negative at the tip
            ca, sa = math.cos(angle), math.sin(angle)
            # the section lives in the (x, z) plane; twist/incidence rotate it about the span axis (y)
            off = np.stack([mid[:, 0] * chord, np.zeros_like(mid[:, 0]), mid[:, 1] * chord], axis=1)
            nrm = np.stack([normal[:, 0], np.zeros_like(normal[:, 0]), normal[:, 1]], axis=1)
            rot = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
            points.append(off @ rot.T + np.array([x_le, y, z]))
            normals.append(nrm @ rot.T)
        return np.concatenate(points), np.concatenate(normals)

    def surface_measure(self, spec: DesignSpec) -> np.ndarray:
        """The length of the section panel each point represents, in metres.

        A predicted field is only a load once it is integrated, and integrating it needs the measure of the surface at
        each point. For this family a point is the midpoint of a section panel of the real thickness, so the weight is
        that panel's length -- taking the distance between consecutive midpoints instead overestimates it where the
        section curves, which is exactly where the suction peak is.
        """
        af = self._airframe(spec)
        nodes = airfoil_nodes(self._section(spec), N_PER_SIDE, close_te=True)
        ring = nodes[:-1] if np.allclose(nodes[0], nodes[-1]) else nodes
        panel = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        out = []
        for frac in _station_fracs():
            out.append(panel * self._station(af, frac)[1])
        return np.concatenate(out)

    # ----------------------------------------------------------------------------------------------- the field
    def field(self, spec: DesignSpec) -> np.ndarray:
        """Cp at every surface point: the lattice's spanwise load carried by the section's own distribution."""
        af = self._airframe(spec)
        shape = self._section(spec)
        alpha = math.radians(spec.operating["alpha_deg"])
        beta = _beta(spec.operating["mach"])
        sol = vlm.solve(af, x_ref=0.25 * af.root_chord, nc=4, ns_wing=N_STRIPS, tail=False)
        # the lattice gives the incompressible load; the section is solved at the incidence that reproduces it, and
        # Prandtl-Glauert enters once, on the pressure: dividing Cp by beta raises the lift by 1/beta as well, which
        # is the rule -- dividing the load too would count compressibility twice
        cl_strip = sol.strip_cl(alpha, 0.0)
        # the section's own lift slope, from two incompressible panel solves of the section itself
        lo, hi = panel_solve(shape, 0.0, n_per_side=N_PER_SIDE, close_te=True), \
            panel_solve(shape, 2.0, n_per_side=N_PER_SIDE, close_te=True)
        slope = (hi.cl_pressure - lo.cl_pressure) / 2.0               # per degree
        if abs(slope) < 1e-6:
            raise ValueError(f"{self.name}: the section has no lift slope")
        strips = []
        for k in range(N_STRIPS):
            # the lattice runs from -b/2 to +b/2, so the second half is the right wing, the side the geometry uses
            target = float(cl_strip[N_STRIPS + k])
            # cl(alpha) is not exactly linear -- camber and thickness bend it -- so the secant that starts the search
            # is corrected once with the cl the section actually returned: the section integral then carries exactly
            # the lift the lattice asked for, which is the identity the tests check
            alpha_eff = (target - lo.cl_pressure) / slope
            result = lo if abs(alpha_eff) < 1e-9 else panel_solve(
                shape, alpha_eff, n_per_side=N_PER_SIDE, close_te=True)
            for _ in range(2):
                gap = target - result.cl_pressure
                if abs(gap) < 1e-6:
                    break
                alpha_eff += gap / slope
                result = panel_solve(shape, alpha_eff, n_per_side=N_PER_SIDE, close_te=True)
            strips.append(result.cp / beta)                           # Prandtl-Glauert on the pressure
        cp = np.concatenate(strips)
        if cp.shape[0] != self.n_points:
            raise ValueError(f"{self.name}: the section solve returned {cp.shape[0]} points per strip against the "
                             f"{self.n_points // N_STRIPS} the geometry carries")
        return cp[:, None]

    # ------------------------------------------------------------------------------------------------- gates
    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A 60 deg swept, 18 % thick wing at M 0.93: outside the planform and beyond the Mach the family trains on."""
        shape = np.round(naca4("2415"), 6)
        geometry = {"aspect_ratio": 3.0, "taper": 0.2, "sweep_le_deg": 60.0, "dihedral_deg": 8.0,
                    "twist_deg": -9.0, "incidence_deg": 6.0}
        return [("a 60 deg swept, AR-3 wing at M 0.93",
                 DesignSpec({**geometry, **{f"cst_{i}": float(v) for i, v in enumerate(shape)}},
                            {"alpha_deg": 12.0, "mach": 0.93}, label="probe:sweep"))]

    def validate(self, spec: DesignSpec) -> str:
        ok, why = valid(self._section(spec))
        if not ok:
            return f"the section is outside the library's bounds: {why}"
        if spec.operating["mach"] >= 1.0:
            return "Prandtl-Glauert has no meaning at or above M = 1"
        return ""


# ------------------------------------------------------------------------------------------------ solid bodies

def _box_grid(lo, hi, n, faces=("x-", "x+", "y-", "y+", "z-", "z+")):
    """Points and outward normals on the faces of a box, ``n`` samples along each of the two in-plane edges.

    The bodies in this module are unions of boxes and cylinders, so their surfaces are written down rather than
    extracted: a face is a rectangle, its normal is the axis, and nothing has to be intersected or projected.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts, nrm = [], []
    t = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(t, t, indexing="ij")
    for axis in range(3):
        for sign in (-1, 1):
            if f"{'xyz'[axis]}{'+' if sign > 0 else '-'}" not in faces:
                continue
            a, b = [i for i in range(3) if i != axis]
            p = np.zeros((n * n, 3))
            p[:, axis] = hi[axis] if sign > 0 else lo[axis]
            p[:, a] = lo[a] + u.ravel() * (hi[a] - lo[a])
            p[:, b] = lo[b] + v.ravel() * (hi[b] - lo[b])
            nn = np.zeros((n * n, 3))
            nn[:, axis] = sign
            pts.append(p)
            nrm.append(nn)
    return np.concatenate(pts), np.concatenate(nrm)


class FinHeatSink3D(FieldFamily):
    """Temperature over an extruded-fin heat sink in three dimensions: conduction in the solid, convection out.

    The base plate carries a chip of known power on its underside; the fins and the exposed top of the base lose heat
    to air at ``h`` and ``T_inf``. The field is the temperature of every solid surface point, which is what a thermal
    engineer reads off a heat sink: how hot the chip footprint is, how much of the fin is doing nothing.

    The reference is the 3-D steady conduction equation in the solid with a Robin (convective) boundary condition,
    solved by the immersed finite-cell method (Burman-Hansbo ghost-penalty stabilisation, Nitsche boundary conditions).
    It is a solver, not a closed form: what a card can claim about it is energy balance, mesh convergence and the
    1-D fin limit, and the tests check all three.
    """

    name = "pp_fin3d"
    title = "Extruded fin heat sink: 3-D fin and base temperature under a chip of known power"
    domain = "electronics cooling / 3-D conjugate fin array"
    dim = 3
    field_names = ("temperature_c",)
    field_units = ("degC",)
    field_scale = (50.0,)
    characteristic_length_m = 0.10
    n_points = 5 * 5 * (3 + 5 * 7)
    tolerance = 0.05
    geometry_params = {"fin_height_mm": (12.0, 40.0), "fin_thickness_mm": (1.0, 3.0),
                       "length_mm": (50.0, 90.0), "base_thickness_mm": (3.0, 8.0)}
    operating_params = {"power_w": (10.0, 80.0), "h_w_m2k": (25.0, 120.0), "t_inf_c": (15.0, 40.0)}
    reference = Reference(
        name="3-D steady conduction with a convective (Robin) boundary, immersed finite-cell method",
        citation=("Burman & Hansbo (2010, 2012) ghost penalty; Nitsche (1971); code: "
                  "pinneapple_simulation.immersed.solve_heat"),
        assumptions=("the air side is a boundary condition, not a flow: h is given, and no natural or forced "
                     "convection over the array is solved",
                     "constant conductivity, no radiation, no contact resistance between chip and base",
                     "the chip is a uniform heat input over a footprint, not a die with its own spreading resistance"),
        implements="solver",
    )

    # -- geometry ------------------------------------------------------------------------------------------------
    def _boxes(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """Base plate corners and the fins that stand on it, in metres."""
        g = spec.geometry
        n = N_FINS
        L = g["length_mm"] * 1e-3
        t_f = g["fin_thickness_mm"] * 1e-3
        h_f = g["fin_height_mm"] * 1e-3
        t_b = g["base_thickness_mm"] * 1e-3
        width = (2 * n + 1) * t_f                       # gaps and fins of equal thickness: the usual extrusion
        base = np.array([[0.0, 0.0, 0.0], [L, width, t_b]])
        fins = np.array([[[0.0, (2 * k + 1) * t_f, t_b], [L, (2 * k + 2) * t_f, t_b + h_f]] for k in range(n)])
        return base, fins

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """Exposed base (top and sides) and every fin face, at a fixed count independent of the fin count."""
        base, fins = self._boxes(spec)
        n = 5
        b_pts, b_nrm = _box_grid(base[0], base[1], n, faces=("z+", "x+", "y-"))
        pts, nrm = [b_pts], [b_nrm]
        for lo, hi in fins:
            p, q = _box_grid(lo, hi, n, faces=("x+", "x-", "y+", "y-", "z+"))
            pts.append(p)
            nrm.append(q)
        return np.concatenate(pts), np.concatenate(nrm)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A sink with 12x the convection coefficient and 4x the power of the trained family."""
        return [("a 480 W sink at 900 W/m2K",
                 DesignSpec({"fin_height_mm": 60.0, "fin_thickness_mm": 2.0, "length_mm": 120.0,
                             "base_thickness_mm": 10.0},
                            {"power_w": 480.0, "h_w_m2k": 900.0, "t_inf_c": 25.0}, label="probe:hot"))]

    def _h(self, spec: DesignSpec) -> float:
        """Background grid spacing: about ten cells along the base, which is a converged-enough 3-D solve and still
        a second per design."""
        return float(spec.geometry["length_mm"]) * 1e-3 / 10.0

    # -- field ---------------------------------------------------------------------------------------------------
    def field(self, spec: DesignSpec) -> np.ndarray:
        from pinneapple_simulation import immersed as im

        base, fins = self._boxes(spec)
        shapes = [im.box(base[0], base[1])] + [im.box(a, b) for a, b in fins]
        geo = shapes[0]
        for s in shapes[1:]:
            geo = im.union(geo, s)
        disc = im.Discretization(geo, self._h(spec), order=1)
        k = 200.0                                        # aluminium 6061, W/m.K
        bcs = [
            im.HeatInput(spec.operating["power_w"],
                         where=lambda X: X[:, 2] < base[0][2] + 1e-6),
            im.Convection(spec.operating["h_w_m2k"], spec.operating["t_inf_c"],
                          where=lambda X: X[:, 2] > base[0][2] + 1e-6),
        ]
        sol = im.solve_heat(disc, k, 0.0, bcs)
        pts, _ = self.points_and_normals(spec)
        return sol(pts)[:, None]


def _occ_boxes(boxes) -> Callable:                                  # noqa: F821 - gmsh builds it
    """A gmsh/OCC volume that is the union of the given boxes: the CAD the conforming tet mesher needs."""
    def build(g):
        created = [g.model.occ.addBox(*np.asarray(lo, float),
                                      *(np.asarray(hi, float) - np.asarray(lo, float))) for lo, hi in boxes]
        fused, _ = g.model.occ.fuse([(3, created[0])], [(3, v) for v in created[1:]])
        return fused[0][1]
    return build


class Bracket3D(FieldFamily):
    """An L-bracket under a tip load, in three dimensions: displacement and von Mises over the solid surface.

    Two plates at a right angle, the foot bolted (clamped) and a load on the free end of the upright. The field is
    what a structural engineer checks before a part is made: where it moves and where it is working hardest. The
    corner fillet is the parameter that trades mass for stress, and the model sees it because it is in the geometry.

    The reference is 3-D linear elasticity on **conforming quadratic tetrahedra** (gmsh, P2), the path this repository
    verifies against CalculiX. The immersed cut-cell grid was tried first and rejected: on a thin bending plate at
    order one, halving the grid moved the tip displacement by half again, which is not a reference. A card on this
    family can claim linearity in the load and mesh convergence, and the tests check both.
    """

    name = "pp_bracket3d"
    title = "L-bracket under a tip load: 3-D displacement and von Mises stress over the surface"
    domain = "structural / 3-D linear elasticity"
    dim = 3
    field_names = ("displacement_m", "von_mises_pa")
    field_units = ("m", "Pa")
    field_scale = (1e-4, 1e7)
    characteristic_length_m = 0.10
    n_points = 2 * 6 * 4 * 4
    tolerance = 0.08
    geometry_params = {"arm_mm": (60.0, 100.0), "width_mm": (25.0, 60.0), "thickness_mm": (6.0, 14.0),
                       "upright_mm": (40.0, 80.0)}
    operating_params = {"load_n": (50.0, 500.0)}
    reference = Reference(
        name="3-D linear elasticity on conforming quadratic tetrahedra (gmsh P2)",
        citation=("Zienkiewicz & Taylor, The Finite Element Method, vol. 1 (2000) ch. 6-7; "
                  "code: pinneapple_simulation.numerical_solvers.tet_fem, verified against CalculiX C3D10 in "
                  "tests/test_immersed.py"),
        assumptions=("small strain, isotropic homogeneous linear elasticity: no plasticity, no contact, no buckling",
                     "the bolt is a clamped face, not a preloaded joint",
                     "a point-like load spread over the free-end face, not a bolted fitting",
                     "von Mises is singular at the edge of the clamped face -- the exact solution is unbounded there -- "
                     "so the stress is reported as the mesh gives it, and it converges only away from the clamp; the "
                     "displacement does converge, and that is what the family's test asks of the mesh"),
        implements="solver",
    )

    # -- geometry ------------------------------------------------------------------------------------------------
    def _plates(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        g = spec.geometry
        arm, width = g["arm_mm"] * 1e-3, g["width_mm"] * 1e-3
        t, up = g["thickness_mm"] * 1e-3, g["upright_mm"] * 1e-3
        foot = np.array([[0.0, 0.0, 0.0], [arm, width, t]])
        # the upright stands at the free end: standing it at the clamped end would put its whole back face in the
        # same plane as the clamp, and the part would be rigid
        upright = np.array([[arm - t, 0.0, t], [arm, width, t + up]])
        return foot, upright

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        foot, up = self._plates(spec)
        p1, n1 = _box_grid(foot[0], foot[1], 4)
        p2, n2 = _box_grid(up[0], up[1], 4)
        return np.concatenate([p1, p2]), np.concatenate([n1, n2])

    def _mesh_size(self, spec: DesignSpec) -> float:
        """Half the plate thickness, so bending has elements through it, but never finer than a twentieth of the arm:
        the quadratic tet solve is direct and its cost grows fast with the element count."""
        g = spec.geometry
        return float(max(g["thickness_mm"] / 2.0, g["arm_mm"] / 20.0)) * 1e-3

    # -- field ---------------------------------------------------------------------------------------------------
    def field(self, spec: DesignSpec) -> np.ndarray:
        """Conforming quadratic tetrahedra (gmsh + P2), sampled back at the surface points of the geometry.

        A cut-cell grid at order one does not converge on this part: the plate is thin and bending-dominated, and
        halving the grid moved the tip displacement by half again. Conforming P2 elements are what the repository
        already verifies against CalculiX (``tests/test_immersed.py``), so the reference here is the one with a
        measured answer.
        """
        from pinneapple_simulation.immersed import bcs as BC
        from pinneapple_simulation.numerical_solvers import tet_fem as T

        foot, up = self._plates(spec)
        mesh = T.gmsh_mesh(_occ_boxes([foot, up]), self._mesh_size(spec), order=2)
        E, nu = 69e9, 0.33                                  # 6061 aluminium
        x_fixed = foot[0][0]
        z_top = up[1][2]
        bcs = [
            BC.Displacement(np.zeros(3), where=lambda X, n: np.isclose(X[:, 0], x_fixed, atol=1e-9)),
            BC.Force(np.array([spec.operating["load_n"], 0.0, 0.0]),
                    where=lambda X, n: np.isclose(X[:, 2], z_top, atol=1e-9)),
        ]
        sol = T.solve_elasticity(mesh, E, nu, bcs=bcs)
        pts, _ = self.points_and_normals(spec)
        return np.stack([np.linalg.norm(sol(pts), axis=1), sol.von_mises(pts)], axis=1)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A load ten times the trained one on a part twice as long."""
        return [("a 5 kN load on a 200 mm arm",
                 DesignSpec({"arm_mm": 200.0, "width_mm": 20.0, "thickness_mm": 3.0, "upright_mm": 150.0},
                            {"load_n": 5000.0}, label="probe:load"))]

    def validate(self, spec: DesignSpec) -> str:
        if spec.operating["load_n"] <= 0.0:
            return "the load must be positive for the reference to be a tip load"
        return ""


# ------------------------------------------------------------------------------ the 3-D nozzle: a scarfed exit

class ScarfedNozzle3D(FieldFamily):
    """A diverging nozzle whose exit plane is cut at an angle: wall Mach and p/p0 over the three-dimensional wall.

    A scarfed nozzle is what a thrust-vectoring installation uses: the exit plane is not perpendicular to the axis, so
    the expansion ratio, the exit Mach and the wall pressure all vary with the azimuth. The wall is a surface of
    revolution, and the field lives on it.

    The reference is the **sector-wise quasi-1-D isentropic** model, which is the standard preliminary-design rule for
    a scarfed contour: each meridional sector expands from the throat to its own exit station with its own area ratio,
    and the sectors do not exchange mass or momentum. That last sentence is the assumption to read twice — the real
    three-dimensional flow *does* communicate, and the non-uniformity the communication leaves behind is exactly the
    loss a scarf produces. A card on this family says so.

    What it gets right, and the tests check: with the scarf at zero the sectors are identical and reproduce the
    quasi-1-D solution of :class:`pinneapple_neural.lpm_families.ConicalNozzle` exactly; the area-Mach inversion is the
    same closed form, so A/A*(M) round-trips; and the exit Mach falls as the scarf shortens a sector's expansion.
    """

    name = "pp_nozzle3d"
    title = "Scarfed (vectoring) nozzle: wall Mach and pressure over the three-dimensional wall"
    domain = "propulsion / thrust-vectoring nozzles"
    dim = 3
    field_names = ("mach", "pressure_over_chamber")
    field_units = ("-", "-")
    field_scale = (5.0, 1.0)
    characteristic_length_m = 0.25
    n_points = 16 * 8
    tolerance = 0.03
    geometry_params = {"area_ratio_exit": (2.0, 20.0), "throat_radius_m": (0.020, 0.080),
                       "half_angle_deg": (10.0, 25.0), "scarf_deg": (0.0, 18.0)}
    operating_params = {"gamma": (1.15, 1.30)}
    reference = Reference(
        name="sector-wise quasi-1-D isentropic expansion on a scarfed contour",
        citation=("Sutton & Biblarz, Rocket Propulsion Elements, 9th ed. (2017) ch. 3 and 11 (scarfed and vectoring "
                  "nozzles); NACA Report 1135 (1953) eqs. 43-45; code: pinneapple_physics.closed_form.compressible "
                  "and .rocket_nozzle, the same inverter the 2-D family uses"),
        assumptions=("each azimuthal sector is a quasi-1-D isentropic streamtube: the sectors do not exchange mass, "
                     "momentum or heat, which is what a scarf costs in reality and is not modelled here",
                     "the wall is a cone from the throat to the exit, with no throat curvature and no boundary layer",
                     "constant gamma, calorically perfect gas, flow attached and expanded to each sector's area ratio",
                     "the field is p/p_chamber and Mach; the physical pressure is that times the chamber pressure"),
        implements="analytic",
    )

    #: axial stations from the throat to each sector's exit, and the azimuths around the wall
    N_AX = 16
    N_TH = 8

    def _contour(self, spec: DesignSpec) -> tuple[float, float, float]:
        """Throat radius, cone half-angle tangent and the nominal (unscarfed) exit station."""
        g = spec.geometry
        rt = g["throat_radius_m"]
        r_exit = rt * math.sqrt(g["area_ratio_exit"])
        tan_half = math.tan(math.radians(g["half_angle_deg"]))
        return rt, tan_half, (r_exit - rt) / tan_half

    def _stations(self, spec: DesignSpec) -> np.ndarray:
        """``(N_AX, N_TH)`` axial stations: every azimuth runs from the throat to its own scarfed exit.

        The exit plane is ``x + tan(scarf) y = L``, and on the cone ``x = (L - tan(scarf) sin(theta) r_t) /
        (1 + tan(scarf) sin(theta) tan(half))`` -- written out rather than iterated, so the same geometry comes back
        every time.
        """
        rt, tan_half, length = self._contour(spec)
        theta = np.linspace(0.0, 2.0 * np.pi, self.N_TH, endpoint=False)
        scarf = math.tan(math.radians(spec.geometry["scarf_deg"]))
        x_exit = (length - scarf * np.sin(theta) * rt) / (1.0 + scarf * np.sin(theta) * tan_half)
        s = np.linspace(0.0, 1.0, self.N_AX)[:, None]
        return s * x_exit[None, :]

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """The diverging wall of revolution, with the normal pointing into the flow."""
        rt, tan_half, _ = self._contour(spec)
        x = self._stations(spec)
        th = np.linspace(0.0, 2.0 * np.pi, self.N_TH, endpoint=False)
        theta = th[None, :]
        r = rt + x * tan_half
        points = np.stack([x, r * np.cos(theta), r * np.sin(theta)], axis=-1).reshape(-1, 3)
        # meridional outward normal of a cone is (-tan(half), 1)/sqrt(1+tan^2); into the flow it flips
        scale = math.hypot(tan_half, 1.0)
        base = np.stack([np.full(self.N_TH, -tan_half / scale), np.cos(th) / scale, np.sin(th) / scale], axis=1)
        normals = -np.broadcast_to(base[None, :, :], (self.N_AX, self.N_TH, 3)).reshape(-1, 3)
        return points, normals

    def field(self, spec: DesignSpec) -> np.ndarray:
        from pinneapple_physics.closed_form.compressible import isentropic_ratios
        from pinneapple_physics.closed_form.rocket_nozzle import mach_from_area_ratio

        gamma = spec.operating["gamma"]
        rt, tan_half, _ = self._contour(spec)
        x = self._stations(spec)
        ratio = ((rt + x * tan_half) / rt) ** 2                       # A/A* of each station of each sector
        mach = np.array([[1.0 if abs(a - 1.0) < 1e-9 else mach_from_area_ratio(float(a), gamma, supersonic=True)
                          for a in row] for row in ratio])
        pressure = 1.0 / isentropic_ratios(mach, gamma)["p0_p"]
        return np.stack([mach.reshape(-1), pressure.reshape(-1)], axis=1)

    def validate(self, spec: DesignSpec) -> str:
        if spec.geometry["scarf_deg"] >= 45.0:
            return "a scarf beyond 45 degrees is outside the sector-wise model's range"
        if spec.geometry["area_ratio_exit"] < 1.0:
            return "the exit must be larger than the throat"
        return ""

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A 35 deg scarf with an area ratio of 40: twice the trained angle and twice the trained expansion."""
        return [("a 35 deg scarf at an area ratio of 40",
                 DesignSpec({"area_ratio_exit": 40.0, "throat_radius_m": 0.05, "half_angle_deg": 15.0,
                             "scarf_deg": 35.0}, {"gamma": 1.2}, label="probe:scarf"))]


# ------------------------------------------------------------------------ the 3-D exchanger: the tube sheet

class TubeSheet3D(FieldFamily):
    """The tube sheet of a shell-and-tube exchanger: temperature through a plate drilled for nine tubes.

    The tube sheet is the part that decides whether an exchanger survives: it sees the hot tube-side fluid on the
    bores, the cool shell-side fluid on both faces and the rim, and the temperature difference between them is what
    produces the thermal stress the mechanical design has to carry. The field is that temperature over the plate and
    the bore walls.

    Geometry: bore radius, pitch, thickness and margin — the four numbers that set how much metal sits between the
    tubes. Operating point: both inlet temperatures and both film coefficients.

    The reference is 3-D steady conduction in the plate with a Robin condition on every surface: ``h_tube`` and the
    hot inlet on the bores, ``h_shell`` and the cold inlet on the faces and the rim, solved by the immersed
    finite-cell method. It is a solver, so what a card can claim is energy balance, mesh convergence and the sign of
    every gradient, and the tests check those.
    """

    name = "pp_exchanger3d"
    title = "Tube sheet of a shell-and-tube exchanger: 3-D temperature from the bores to the shell side"
    domain = "energy / heat exchanger mechanical and thermal design"
    dim = 3
    field_names = ("temperature_c",)
    field_units = ("degC",)
    field_scale = (50.0,)
    characteristic_length_m = 0.10
    n_points = 9 * 6 * 3 + 2 * 4 * 4
    tolerance = 0.05
    geometry_params = {"bore_radius_mm": (2.5, 6.0), "pitch_mm": (14.0, 30.0), "thickness_mm": (12.0, 30.0),
                       "margin_mm": (8.0, 20.0)}
    operating_params = {"t_hot_c": (80.0, 180.0), "t_cold_c": (15.0, 45.0),
                        "h_tube_w_m2k": (500.0, 3000.0), "h_shell_w_m2k": (200.0, 1500.0)}
    reference = Reference(
        name="3-D steady conduction in the tube sheet with a convective boundary on every surface",
        citation=("Burman & Hansbo (2010, 2012) ghost penalty; Nitsche (1971); Kern (1950) for the film "
                  "coefficients the operating point carries; code: pinneapple_simulation.immersed.solve_heat"),
        assumptions=("the two fluids are boundary conditions, not flows: h_tube and h_shell are given and no "
                     "shell-side or tube-side flow is solved",
                     "constant conductivity and no fouling resistance, no axial conduction along the tubes outside "
                     "the plate, no thermal contact resistance, no radiation",
                     "the plate is stress-free: this is the temperature that a thermal-stress analysis would take as "
                     "input, not the stress itself"),
        implements="solver",
    )

    #: bores: a 3 x 3 square pattern, fixed, because the field of a design has to line up with every other design
    N_BORES = 3
    N_TH = 6
    N_AX = 3

    def _plate(self, spec: DesignSpec) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        """Plate corners, bore radius, thickness and the bore centres (all in metres)."""
        g = spec.geometry
        r_b, pitch = g["bore_radius_mm"] * 1e-3, g["pitch_mm"] * 1e-3
        thickness, margin = g["thickness_mm"] * 1e-3, g["margin_mm"] * 1e-3
        side = 3 * pitch + 2 * margin
        centres = np.array([[margin + pitch * (i + 0.5), margin + pitch * (j + 0.5)]
                            for i in range(self.N_BORES) for j in range(self.N_BORES)])
        return np.array([[0.0, 0.0, 0.0], [side, side, thickness]]), r_b, thickness, side, centres

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """The bore walls and the two faces: the surfaces the model reads the field from."""
        (lo, hi), r_b, thickness, side, centres = self._plate(spec)
        theta = np.linspace(0.0, 2.0 * np.pi, self.N_TH, endpoint=False)
        z = np.linspace(0.25 * thickness, 0.75 * thickness, self.N_AX)
        wall = np.array([[[(cx + r_b * math.cos(t), cy + r_b * math.sin(t), zi) for t in theta]
                          for zi in z] for cx, cy in centres]).reshape(-1, 3)
        wall_n = np.array([[[(math.cos(t), math.sin(t), 0.0) for t in theta] for _ in z]
                           for _ in centres]).reshape(-1, 3)
        # the faces are sampled on the ligament centres and in the margin bands: every one of them is solid for any
        # bore and pitch the family draws, so a design's point cloud lines up with every other design's
        pitch = centres[1, 0] - centres[0, 0]
        margin = centres[0, 0] - 0.5 * pitch
        u = np.array([0.5 * margin, margin + pitch, margin + 2.0 * pitch, side - 0.5 * margin])
        gx, gy = np.meshgrid(u, u, indexing="ij")
        faces, face_n = [], []
        for zface, sign in ((0.0, -1.0), (thickness, 1.0)):
            pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zface)], axis=1)
            faces.append(pts)
            face_n.append(np.tile([0.0, 0.0, sign], (len(pts), 1)))
        return np.concatenate([wall] + faces), np.concatenate([wall_n] + face_n)

    def _h(self, spec: DesignSpec) -> float:
        """A quarter of the pitch: the metal between two bores is what the field turns on."""
        return float(spec.geometry["pitch_mm"]) * 0.25e-3

    def field(self, spec: DesignSpec) -> np.ndarray:
        from pinneapple_simulation import immersed as im

        (lo, hi), r_b, _, _, centres = self._plate(spec)
        solid = im.box(lo, hi)
        for cx, cy in centres:
            solid = im.difference(solid, im.cylinder((cx, cy, 0.5 * hi[2]), r_b, 2.0 * hi[2], axis=2))
        disc = im.Discretization(solid, self._h(spec), order=1)

        def on_bore(X, n):
            # a point is on a bore wall when its distance to the nearest bore axis is the bore radius and it is not on
            # one of the two faces: the tolerance is the cut-cell boundary's own, about 1e-6 m
            radial = np.hypot(X[:, None, 0] - centres[None, :, 0], X[:, None, 1] - centres[None, :, 1])
            on_wall = np.abs(radial.min(axis=1) - r_b) < 2e-5
            inside = (X[:, 2] > 1e-5) & (X[:, 2] < hi[2] - 1e-5)
            return on_wall & inside

        bcs = [im.Convection(spec.operating["h_tube_w_m2k"], spec.operating["t_hot_c"], where=on_bore),
               im.Convection(spec.operating["h_shell_w_m2k"], spec.operating["t_cold_c"],
                             where=lambda X, n: ~on_bore(X, n))]
        sol = im.solve_heat(disc, 16.0, 0.0, bcs)                    # stainless steel, W/m.K
        pts, _ = self.points_and_normals(spec)
        return sol(pts)[:, None]

    def validate(self, spec: DesignSpec) -> str:
        g, op = spec.geometry, spec.operating
        if op["t_hot_c"] <= op["t_cold_c"]:
            return "the hot inlet must be hotter than the cold inlet for the reference to have a direction"
        if 2.0 * g["bore_radius_mm"] >= 0.8 * g["pitch_mm"]:
            return "the bores touch: the ligament between tubes must be positive"
        return ""

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """A plate with 4x the film coefficients and a 160 K difference across it."""
        return [("a 160 K difference at 6000 W/m2K",
                 DesignSpec({"bore_radius_mm": 3.0, "pitch_mm": 12.0, "thickness_mm": 10.0, "margin_mm": 6.0},
                            {"t_hot_c": 190.0, "t_cold_c": 30.0, "h_tube_w_m2k": 6000.0, "h_shell_w_m2k": 4000.0},
                            label="probe:hot"))]


def families_3d() -> dict[str, FieldFamily]:
    """The 3-D families by name. Kept apart from :func:`pinneapple_neural.lpm_pipeline.families` so the five 2-D
    families and their measured numbers stay what they are."""
    fams = (Wing3D(), FinHeatSink3D(), Bracket3D(), ScarfedNozzle3D(), TubeSheet3D())
    return {f.name: f for f in fams}
