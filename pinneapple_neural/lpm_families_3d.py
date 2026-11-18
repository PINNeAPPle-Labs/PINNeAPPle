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
        for k in range(N_STRIPS):
            frac = (k + 0.5) / N_STRIPS
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
        for k in range(N_STRIPS):
            chord = self._station(af, (k + 0.5) / N_STRIPS)[1]
            out.append(panel * chord)
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


def families_3d() -> dict[str, FieldFamily]:
    """The 3-D families by name. Kept apart from :func:`pinneapple_neural.lpm_pipeline.families` so the five 2-D
    families and their measured numbers stay what they are."""
    fams = (Wing3D(), FinHeatSink3D(), Bracket3D())
    return {f.name: f for f in fams}
