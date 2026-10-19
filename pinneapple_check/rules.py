"""The rule table of ``pinneapple check``: every rule has an id (``PC101``), a short name, a default severity, the
trust-card stage it gives evidence for, an explanation and a fix hint.

Ids are grouped by hundreds, so ``--select PC1`` picks a whole family:

* ``PC0xx`` reading the inputs
* ``PC1xx`` conservation (mass, energy, momentum, divergence)
* ``PC2xx`` boundary and initial conditions
* ``PC3xx`` PDE residual
* ``PC4xx`` bounds and finite values
* ``PC5xx`` units
* ``PC6xx`` out-of-distribution use of a model
* ``PC7xx`` regression against a stored reference
* ``PC8xx`` integrity of artefacts (hashes, signatures, pickles)
* ``PC9xx`` method status in the catalog
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Rule", "RULES", "get_rule", "select_rules", "DOCS_URL"]

DOCS_URL = "https://pinneapple-labs.github.io/PINNeAPPle/guides/physics_checks_in_ci/"

ERROR, WARNING = "error", "warning"


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    title: str                  # one line, used as the annotation title
    severity: str               # error | warning (default; the config can change it per rule)
    stage: str                  # trust-card stage: data_geometry | model | physics | benchmark | uncertainty | decision
    evidence: str               # name of the veriphysics guardrail check this rule feeds ("" if none)
    explanation: str
    fix: str

    @property
    def help_uri(self) -> str:
        return f"{DOCS_URL}#{self.id.lower()}"


def _r(id, name, title, severity, stage, evidence, explanation, fix):
    return Rule(id, name, title, severity, stage, evidence, " ".join(explanation.split()), " ".join(fix.split()))


RULES: dict[str, Rule] = {r.id: r for r in [
    # ------------------------------------------------------------------ PC0 inputs
    _r("PC001", "unreadable-target", "A declared result or model could not be read", ERROR, "data_geometry", "",
       """The file or folder named in the configuration does not exist, has an unsupported format, or failed to
       parse. Nothing else can be checked for this target, so this is always an error.""",
       """Check the path (relative to the configuration file) and the format: an OpenFOAM case folder, .vtu/.vtk,
       .npz, .h5/.hdf5, a .csv of quantities of interest, a Python callable `file.py:function` or a checkpoint."""),
    _r("PC002", "nothing-checked", "A target matched no applicable check", WARNING, "decision", "",
       """No rule could run on this target: no physics was declared or inferred, no bounds, no reference. A
       linter that checks nothing passes everything, so this is reported.""",
       """Declare `physics = "incompressible-flow" | "heat" | "elasticity" | "conservation"`, bounds, boundary
       conditions or a reference for the target."""),
    # ------------------------------------------------------------------ PC1 conservation
    _r("PC101", "mass-imbalance", "Net mass flux through the boundary is not zero", ERROR, "physics",
       "conservation_mass",
       """For an incompressible (or steady) flow the volume flux entering the domain must leave it. The check sums
       the flux through every boundary face (from `phi` when the solver wrote it, else from the face velocity) or,
       for a model, integrates u.n over the faces of the domain box, and divides the net flux by the total absolute
       flux. A non-zero value means a leaking boundary condition, an unconverged run or a model that creates or
       destroys mass. For `darcy` it compares the net outward Darcy flux with the integral of the source.""",
       """Look at the inlet/outlet patches named in the message; check that the run converged (residuals) and that
       the outlet is not a fixed-value velocity. For a PINN, add a mass-balance or divergence penalty, or use a
       divergence-free architecture (stream function)."""),
    _r("PC102", "divergence", "Velocity field is not divergence-free", ERROR, "physics", "conservation_divergence",
       """Incompressible flow requires div(u) = 0 in every cell. The check computes the discrete divergence (finite
       volume on OpenFOAM meshes, finite differences on grids, least squares on point clouds, central differences
       on a model) and reports its RMS relative to the RMS of the velocity gradients, a number between 0 and 1.
       For `maxwell` the same measure is applied to Gauss's laws, div D = rho and div B = 0.""",
       """Converge the pressure-velocity coupling further (more correctors, tighter p tolerance). For a model,
       raise the weight of the continuity residual or sample more collocation points where the message says the
       divergence is largest."""),
    _r("PC103", "energy-imbalance", "Heat entering the domain does not balance heat leaving it", ERROR, "physics",
       "conservation_energy",
       """For steady conduction the conductive flux k dT/dn through the boundary plus the heat generated inside
       must be zero. The check reports the net heat flow relative to the total absolute boundary heat flow. For
       `maxwell` it is Poynting's theorem over the grid box: the change of electromagnetic energy plus the
       Poynting flux through the faces plus the work J.E must vanish.""",
       """Check the thermal boundary conditions (a fixed temperature on all walls with a source has no steady
       solution only if the flux cannot leave), the source term and the conductivity; converge further. For
       `maxwell`: the fields are not a solution (wrong frequency, material or source), or energy leaves through
       faces that were not stored (crop less)."""),
    _r("PC104", "momentum-imbalance", "Forces on the body do not balance", ERROR, "physics", "conservation_momentum",
       """For static elasticity the tractions on the boundary plus the body force must sum to zero (Newton's
       second law with no acceleration). The check reports the norm of the resultant relative to the total
       absolute traction.""",
       """A missing support, a load applied twice or wrong material constants. Check the declared Young's modulus
       and Poisson ratio and the displacement boundary conditions."""),
    _r("PC105", "conserved-total-drift", "A conserved total changes in time", ERROR, "physics", "conservation_total",
       """For a generic conservation law on a closed domain (or with the boundary flux declared) the integral of
       the conserved quantity must stay constant. The check compares the total at every stored time with the
       first one.""",
       """Use a conservative discretisation, reduce the time step, or declare the boundary flux column so the
       balance accounts for it."""),
    # ------------------------------------------------------------------ PC2 BC/IC
    _r("PC201", "boundary-condition", "Boundary values differ from the declared boundary condition", ERROR, "physics",
       "boundary_condition",
       """The value of a field on a boundary patch (or a model's prediction on a boundary of its domain) differs
       from the value declared in the configuration by more than the tolerance (relative to the field scale).""",
       """Fix the boundary condition in the case (the patch is named in the message and the annotation points at
       its entry), or raise the boundary-loss weight of the model."""),
    _r("PC202", "initial-condition", "The solution at the initial time differs from the declared initial condition",
       ERROR, "physics", "initial_condition",
       """At the first time (stored snapshot, or t = t0 for a model with a time input) the field must match the
       declared initial condition.""",
       """Check the initial field written by the pre-processing, or the initial-condition loss of the model."""),
    _r("PC203", "wall-velocity", "Velocity on a wall patch is not zero", WARNING, "physics", "boundary_condition",
       """On a patch of type `wall` the fluid sticks to the wall (no-slip). A non-zero face velocity on a wall
       usually means a moving wall that was not declared, or a slip condition by mistake.""",
       """If the wall moves (a lid, a rotating part), declare its velocity under `[result.bc.<patch>]`; otherwise
       set `noSlip` on the patch."""),
    # ------------------------------------------------------------------ PC3 residual
    _r("PC301", "pde-residual", "PDE residual of the stored fields is too large", ERROR, "physics", "pde_residual",
       """The governing equation of the declared physics is evaluated on the stored fields and its RMS residual is
       divided by the RMS of the sum of the absolute values of its terms: 0 means the equation holds exactly, 1
       means the terms do not balance at all. Discretisation error makes it non-zero on coarse meshes, so the
       default tolerance is loose. For `darcy` the residual of the finite-volume discretisation is turned into a
       guaranteed bound on the relative energy-norm error (tolerance `darcy-energy`). For `maxwell` the laws are
       Faraday's, Ampere-Maxwell's and charge continuity; the message also gives the ratio of the residual on a
       grid twice as coarse, about 4 when the residual is only discretisation error.""",
       """Converge the solver, refine the mesh where the residual is largest, or check the declared parameters
       (viscosity, conductivity, source) match the ones used in the run."""),
    _r("PC302", "model-residual", "PDE residual of a model's predictions is too large", ERROR, "physics",
       "pde_residual",
       """The model is evaluated at sampled interior points of its domain and the PDE residual is computed with
       central differences. Reported as the relative residual described for PC301.""",
       """Train longer, add collocation points where the residual is largest, or check that the declared
       parameters match the ones the model was trained with."""),
    # ------------------------------------------------------------------ PC4 bounds
    _r("PC303", "metamorphic-relation", "A model breaks a relation its physics implies", ERROR, "physics", "",
       """A declared metamorphic relation does not hold for the model's predictions: more of an input moves an output
       the wrong way (`monotonic`), a change that must not matter moves the output (`invariant`, e.g. a similarity
       group kept fixed), a mirrored input does not give the mirrored output (`symmetric`), or a linear model is not
       linear (`superposition`). These hold whatever the true answer is, so they test a surrogate where no reference
       solution exists. The worst sample found is reported as the counterexample
       (`pinneapple_analysis.validation.metamorphic`).""",
       """Look at the counterexample: a sign error, an input swapped or in the wrong unit, or a fit that is not
       constrained in that region. If the relation does not hold for the real physics there, narrow its box or
       remove it rather than loosening the tolerance."""),
    _r("PC401", "out-of-bounds", "A field leaves its declared bounds", ERROR, "data_geometry", "bounds",
       """A value is outside the range declared under `bounds` in the configuration.""",
       """Check the solver limiters (e.g. MULES for a phase fraction), the time step, or the model's output
       activation (a sigmoid keeps a fraction in [0, 1])."""),
    _r("PC402", "unphysical-value", "A field takes a physically impossible value", ERROR, "data_geometry", "bounds",
       """Some quantities have hard physical limits whatever the problem: absolute temperature and density are
       positive, phase and mass fractions lie in [0, 1], turbulent kinetic energy, dissipation and eddy viscosity
       are non-negative. The quantity is inferred from the field name (T, rho, alpha.*, Y*, k, epsilon, omega,
       nut) and its unit.""",
       """Bound the solution (positivity-preserving schemes, bounded limiters, a softplus on the model output) or,
       if the name is misleading, declare `bounds` for the field explicitly."""),
    _r("PC403", "non-finite", "A field contains NaN or infinite values", ERROR, "data_geometry", "bounds",
       """NaN or Inf in a stored field or a model prediction. Every other check on that field is unreliable.""",
       """The run diverged or the model was evaluated outside its numeric range; inspect the time step and the
       residual history."""),
    # ------------------------------------------------------------------ PC5 units
    # ---- datasets (pinneapple_data.audit)
    _r("PC451", "dataset-physics", "Training data do not solve the stated PDE or break a declared constraint", ERROR,
       "data_geometry", "",
       """Samples of a dataset differ from the solution of their own discrete problem (or their boundary values or
       flux balance are off), or rows break a relation declared between columns. A model trained on such data
       inherits the error, and model-level checks may never see it (NVIDIA/physicsnemo#2036: Darcy data generated
       with a wrongly scaled cross term).""",
       """Regenerate the data with the documented equation and discretisation; the failing samples are listed in
       the evidence."""),
    _r("PC452", "dataset-diversity", "Training data are less diverse than stated or contain duplicates", ERROR,
       "data_geometry", "",
       """The effective number of distinct samples (participation ratio of the principal components) collapses, or
       samples are exact or near copies of each other (NVIDIA/physicsnemo#2035: every Fourier coefficient of a
       permeability sample shared one value).""",
       """Check the sampler: independent random coefficients per mode and per sample, a fresh seed per sample."""),
    _r("PC453", "dataset-leakage", "Test samples are copies of training samples", ERROR, "data_geometry", "",
       """A test sample's nearest training sample is closer than a thousandth of the typical spacing between
       training samples: the held-out error measures memorisation, not generalisation.""",
       """Split before deduplicating, split by case or geometry rather than by row, and drop the flagged pairs."""),
    _r("PC501", "unit-mismatch", "A field's unit differs from the declared unit", ERROR, "data_geometry",
       "dimensional_analysis",
       """The unit written in the file (OpenFOAM `dimensions`, a CSV header `T [degC]`, an HDF5 `units` attribute)
       does not match the unit declared in the configuration: either a different dimension (m/s versus Pa) or the
       same dimension on a different scale (degC versus K).""",
       """Convert the data, or correct the declared unit."""),
    _r("PC502", "unit-missing", "A field used by a physics check has no unit", WARNING, "data_geometry",
       "dimensional_analysis",
       """Neither the file nor the configuration gives the unit of a field that a physics check uses, so its
       magnitude cannot be checked against the parameters.""",
       """Declare it: `units = { T = "K", U = "m/s" }` under the target."""),
    _r("PC503", "implausible-magnitude", "Values are implausible for the declared unit", WARNING, "data_geometry",
       "dimensional_analysis",
       """The values are far outside the typical range of the quantity in the declared unit, which is the usual sign
       of a unit slip: temperatures around 20 declared in kelvin, pressures in bar declared in Pa.""",
       """Check which unit the data is really in and declare that one."""),
    _r("PC504", "parameter-unit", "A physical parameter has the wrong dimension", ERROR, "data_geometry",
       "dimensional_analysis",
       """A parameter given with a unit (`nu = "1e-5 m2/s"`) has a dimension different from the one the declared
       physics needs (kinematic viscosity, thermal conductivity, Young's modulus...).""",
       """Give the parameter in a unit of the right dimension; `nu` is kinematic (m2/s), `mu` dynamic (Pa s)."""),
    # ------------------------------------------------------------------ PC6 OOD
    _r("PC601", "out-of-range-query", "A model is queried outside its training range", ERROR, "model", "",
       """At least one query (a row of the `queries` file) has an input outside the range the model was trained on.
       The training range comes from the configuration (`train-range`), the checkpoint metadata
       (`domain_bounds`) or the model card. A surrogate extrapolates silently: its answer there is not
       evidence.""",
       """Restrict the queries to the training range, retrain on a range that covers them, or route those queries
       to the solver (a fallback)."""),
    _r("PC602", "sparse-training-region", "A query falls in a gap of the training data", WARNING, "model", "",
       """The query is inside the training box but far from every training sample (more than three typical
       spacings, on standardised inputs), so the model interpolates over a hole.""",
       """Add training samples near the reported query, or treat its answer as an extrapolation."""),
    # ------------------------------------------------------------------ PC7 regression
    _r("PC701", "field-regression", "A field differs from the stored reference", ERROR, "benchmark",
       "reference_data_match",
       """Relative L2 error between the field and the reference (same points, or the reference interpolated on
       them) is above the tolerance.""",
       """If the change is intended, regenerate the reference and commit it with the change; otherwise find the
       commit that moved the result."""),
    _r("PC702", "qoi-regression", "A quantity of interest differs from the stored reference", ERROR, "benchmark",
       "reference_data_match",
       """A quantity of interest (drag, lift, a peak temperature...) differs from its reference value by more than
       its relative or absolute tolerance.""",
       """Same as PC701: update the reference deliberately, or fix the regression."""),
    _r("PC703", "reference-unusable", "The reference cannot be compared", WARNING, "benchmark",
       "reference_data_match",
       """The reference file is missing, unreadable, lacks the field, or has a shape that cannot be matched to the
       result.""",
       """Regenerate the reference with the same mesh/points and field names, or give `fields` to compare."""),
    _r("PC704", "spectral-bias", "A field loses or gains small-scale energy against the reference", WARNING,
       "benchmark", "reference_data_match",
       """Energy in the high-wavenumber band of the field differs from the reference's by more than the tolerance
       (`E_pred / E_ref - 1` over the top third of the resolved shells). A smoothed field has a small relative L2
       error and still misses the small scales that carry heat transfer, mixing or loads; added noise shows up as a
       positive bias. Computed on grid results with the same shape as the reference
       (`pinneapple_analysis.validation.spectral`).""",
       """Check the model's spectral bias (more Fourier modes, a finer output grid, a loss on gradients or spectra),
       or raise `tolerances.hf-bias` deliberately if the small scales do not matter for the use."""),
    # ------------------------------------------------------------------ PC8 integrity
    _r("PC801", "hash-mismatch", "An artefact does not match its recorded hash", ERROR, "data_geometry", "",
       """The SHA-256 of a file differs from the one in the integrity manifest (or a listed file is missing): the
       artefact changed after it was recorded.""",
       """If the change is intended, rebuild the manifest (`python -m pinneapple_security manifest <dir>`) in the
       same commit; otherwise restore the file."""),
    _r("PC802", "bad-signature", "An artefact's signature does not verify", ERROR, "data_geometry", "",
       """The detached signature (`<file>.sig`) does not verify with the configured public key, or is missing
       when signatures are required.""",
       """Re-sign the artefact with the release key (`python -m pinneapple_security sign`), or check that the
       configured public key is the right one."""),
    _r("PC803", "unsafe-checkpoint", "A model checkpoint can execute code when loaded", ERROR, "model", "",
       """The pickle stream of the checkpoint imports callables outside the allow-list, so `torch.load` would run
       them. The checker does not load such a file.""",
       """Re-save the weights only (`torch.save(model.state_dict(), ...)`) or as safetensors."""),
    _r("PC804", "untracked-artefact", "An artefact is not covered by the integrity manifest", WARNING,
       "data_geometry", "",
       """A file in a folder covered by the manifest is not listed in it, so a change to it would go unnoticed.""",
       """Rebuild the manifest so it includes the file."""),
    # ------------------------------------------------------------------ PC9 catalog
    _r("PC901", "unvalidated-method", "The model's method is not validated in the method catalog", WARNING, "model",
       "",
       """The method id declared for the model (`method = "S12"`) has status `tested` or `untested` in the
       PINNeAPPle method catalog: no reference test checks it against a known solution.""",
       """Prefer a validated method for decisions, or add a reference test for this one."""),
]}


def get_rule(rule_id: str) -> Rule:
    try:
        return RULES[rule_id.upper()]
    except KeyError:
        raise KeyError(f"unknown rule {rule_id!r}; `pinneapple check --list-rules` lists them") from None


def _matches(rule_id: str, prefixes) -> bool:
    return any(rule_id.startswith(p.upper()) for p in prefixes)


def select_rules(select=None, ignore=None, extend_select=None) -> set[str]:
    """Rule ids enabled by ruff-style prefixes: ``select`` (default: all), minus ``ignore``, plus ``extend_select``.
    ``ALL`` and ``PC`` select every rule. PC001 (unreadable target) stays on."""
    sel = [s for s in (select or ["ALL"])]
    sel = ["PC" if s.upper() == "ALL" else s for s in sel]
    for p in [*sel, *(ignore or []), *(extend_select or [])]:
        if not any(r.startswith(p.upper()) for r in RULES):
            raise KeyError(f"no rule matches {p!r}; `pinneapple check --list-rules` lists them")
    on = {r for r in RULES if _matches(r, sel)}
    on |= {r for r in RULES if _matches(r, extend_select or [])}
    return {r for r in on if not _matches(r, ignore or [])} | {"PC001"}     # an unreadable target is never ignored
