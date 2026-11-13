"""The universal physics data schema: one description for a physics dataset, whoever produced it.

Why this exists
---------------
A PDE dataset arrives as an HDF5 file from PDEBench, a Zarr store from The Well, a folder of OpenFOAM fields
or a VTU from CalculiX. Every one of them carries the same five things -- a geometry, a mesh, the physics that
was solved, the fields that came out and where they came from -- and every one of them spells them
differently. Anything downstream (a PINN, an FNO, a benchmark, the verification layer, the API) then has to
re-learn each spelling, and the places where the spellings disagree are exactly where silent errors live:
a temperature in K read as degC, cell-centred values treated as nodal, a y axis flipped, a field that lost its
provenance.

This package fixes one spelling. :class:`~pinneapple_data.schema.models.BundleManifest` is a pydantic model
(``pinneapple.physics-data/1``) that says what a dataset *is*; :class:`~pinneapple_data.schema.bundle.PhysicsDataBundle`
is that manifest plus the arrays it references, saved as a directory with a hashed ``bundle.json`` and an
``arrays.npz``. The rules it enforces are the ones that bite in practice:

* **units** -- every array with a physical meaning declares a unit, and it has to parse
  (:func:`pinneapple_data.physical_units.parse_unit`); dimensionless is written ``"1"``;
* **location** -- every field says whether its values live on nodes or at cell centres, and its shape has to
  agree with the mesh it is attached to;
* **frame and origin** -- the coordinate arrays are named, ordered and unit-bearing, so "the y axis is flipped"
  and "x in mm" are recorded instead of assumed;
* **provenance** -- solver, version, configuration, the SHA-256 of every input file and the licence travel with
  the data (the licence is a field, not a README sentence);
* **verification** -- checks are recorded with their tolerance and result, and a check that did not run is
  ``"not run"``, never "passed".

What this is not
----------------
Not a file format converter (the adapters in :mod:`pinneapple_data.datasets` do that), not a mesh library and
not a replacement for ``pinneapple_data.contracts`` (``ParameterSpace`` / ``DatasetSpec`` / ``ModelRef`` /
``StudyResult``, which describe *studies*). A ``DatasetSpec`` points at a table of cases; a bundle is the
arrays of one case (or of a batch of cases) with the physics they satisfy.

Example
-------
    >>> import numpy as np
    >>> from pinneapple_data.schema import PhysicsDataBundle, Field, AxisSpec, Physics
    >>> u = np.zeros((32, 64))                   # 32 time steps, 64 points
    >>> b = PhysicsDataBundle.from_arrays(
    ...     {"u": u, "t": np.linspace(0, 1, 32), "x": np.linspace(0, 1, 64)},
    ...     fields=[Field.from_array("u", u, unit="m/s", location="nodal", axes=("t", "x"))],
    ...     axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
    ...     physics=Physics(governing="Burgers: u_t + u u_x = nu u_xx"),
    ...     source="Burgers, nu=0.01")
    >>> b.save("burgers.ppb")                    # bundle.json + arrays.npz
    >>> PhysicsDataBundle.load("burgers.ppb") == b
    True
"""
from .bundle import PhysicsDataBundle, load, save
from .gates import GateReport, run_gates
from .models import (
    SCHEMA_VERSION,
    ArrayRef,
    AxisSpec,
    BoundaryCondition,
    BundleManifest,
    CheckResult,
    Field,
    Fields,
    FileRef,
    Geometry,
    InitialCondition,
    Material,
    Mesh,
    Parameter,
    Physics,
    Provenance,
    Region,
    Verification,
)

__all__ = [
    "SCHEMA_VERSION", "ArrayRef", "AxisSpec", "BoundaryCondition", "BundleManifest", "CheckResult", "Field",
    "Fields", "FileRef", "Geometry", "InitialCondition", "Material", "Mesh", "Parameter", "Physics",
    "Provenance", "Region", "Verification", "PhysicsDataBundle", "save", "load", "GateReport", "run_gates",
]
