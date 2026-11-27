"""Interplanetary trajectory design: patched-conic Lambert arcs, low-thrust Sims-Flanagan legs, evolutionary search
over planet sequences, and the N-body check that says whether the conic answer is a trajectory at all.

The package answers the three questions a preliminary mission design asks, in the order it asks them:

1. **What does it cost to get from one planet to another in a given time?** :func:`lambert` solves Lambert's problem
   in universal variables; :class:`LambertArc` carries the two velocities and the arc's transfer angle.
2. **What does the whole sequence cost?** :class:`MGAProblem` decodes ``[t0, T1, ..., Tn]`` into planet states
   (:func:`default_ephemeris`) and Lambert arcs, and adds up the departure impulse, the powered flybys and the
   arrival impulse. :func:`cassini1` is the ESA GTOP benchmark problem, so a number from this package can be read
   against a published one.
3. **Is the answer a mission or only a solution of the conic problem?** :func:`verify_design` re-flies the design in
   the restricted N-body field and reports the miss at every encounter in km, planet radii and Hill radii.

    from pinneapple_physics.astro.trajectory_design import cassini1, optimise, verify_design

    search = optimise(cassini1(), method="scipy-de", budget=20_000, seed=0)
    print(search.summary())                       # measured Δv against the published best, and the gap
    report = verify_design(search.best_design())  # N-body verdict on the same design
    print(report.verdict, report.tcm_dv_km_s)

Modules
-------
:mod:`.bodies`
    Planet states for design: the local JPL DE kernel when it is installed (rotated to the J2000 ecliptic) and the
    published JPL approximate elements as the offline path, so a design script runs with nothing downloaded.
:mod:`.lambert`
    Lambert's problem in universal variables with a Newton solve on the exact slope, and the universal-variable
    two-body propagator the Sims-Flanagan legs are made of.
:mod:`.segments`
    Low-thrust legs the way Sims & Flanagan (1997) posed them: bounded impulses at segment midpoints, mass falling
    with the flow of the engine, and a deterministic closure-and-cheapest solver on top.
:mod:`.gtop`
    The GTOP-style multiple-gravity-assist problem: the decision vector, the impulse budget, the published values it
    is compared with, and the ``pinneapple_design`` evaluator that lets the library's optimisers run on it.
:mod:`.optimisers`
    NSGA-II, differential evolution, CMA-ES when it is installed, and MAP-Elites -- thin layers over
    ``pinneapple_design.design_optimizer``, not a second evolutionary implementation.
:mod:`.trust`
    The restricted N-body propagation of a design, the miss at each encounter, the correction impulse that would
    re-fly it and the verdict: ``reproduces-the-conic``, ``needs-correction`` or ``conic-only``.

Guide: ``docs/core_concepts/trajectory_design.md``. The numbers in it are measured, with the commands that measure
them, and the gap between the search and the published optimum is reported rather than smoothed over.

``TrustReport`` here is the conic-check verdict of :mod:`.trust`; it is not
:class:`pinneapple_analysis.trust.TrustReport`, which is the library-wide decision object for a physics result.
"""
from .bodies import (
    ACT_ELEMENTS,
    DESIGN_BODIES,
    J2000_ELEMENTS,
    AnalyticEphemeris,
    GTOPEphemeris,
    KernelEphemeris,
    default_ephemeris,
)
from .gtop import (
    CASSINI1,
    CASSINI1_BEST_X,
    DAY,
    MJD2000_JD,
    PUBLISHED_BEST,
    MGADesign,
    MGAEvaluator,
    MGAProblem,
    MGASequence,
    capture_dv,
    cassini1,
    free_flyby_constraint,
    powered_flyby,
)
from .lambert import (
    LambertArc,
    LambertError,
    lambert,
    propagate_universal,
    stumpff,
    transfer_angle_deg,
)
from .optimisers import (
    CMAES_AVAILABLE,
    METHODS,
    EvolutionaryDesignOptimizer,
    MapElitesResult,
    TrajectorySearch,
    build_problem,
    cmaes_available,
    compute_pareto_front,
    map_elites,
    optimise,
)
from .segments import (
    G0,
    SimsFlanaganDesign,
    SimsFlanaganLeg,
    Thruster,
    design_leg,
    evaluate_leg,
    max_impulse,
)
from .trust import (
    EncounterCheck,
    TrustReport,
    planet_hill_radius_km,
    propagate_design,
    trajectory_correction,
    verify_design,
)

__all__ = ["AnalyticEphemeris", "GTOPEphemeris", "KernelEphemeris", "default_ephemeris", "J2000_ELEMENTS",
           "ACT_ELEMENTS", "DESIGN_BODIES",
           "LambertArc", "LambertError", "lambert", "propagate_universal", "stumpff", "transfer_angle_deg",
           "SimsFlanaganLeg", "SimsFlanaganDesign", "Thruster", "design_leg", "evaluate_leg", "max_impulse", "G0",
           "MGASequence", "MGADesign", "MGAProblem", "MGAEvaluator", "CASSINI1", "CASSINI1_BEST_X", "DAY",
           "MJD2000_JD", "PUBLISHED_BEST", "powered_flyby", "capture_dv",
           "cassini1", "free_flyby_constraint", "optimise", "map_elites", "build_problem", "TrajectorySearch",
           "MapElitesResult", "METHODS", "cmaes_available", "CMAES_AVAILABLE", "EvolutionaryDesignOptimizer",
           "compute_pareto_front", "verify_design", "propagate_design", "trajectory_correction", "TrustReport",
           "EncounterCheck", "planet_hill_radius_km"]
