"""Built-in benchmark problems and methods (importing this module registers them)."""
from . import calculix, fields, standard  # noqa: F401
from .fields import ArrayProblem, FieldOperatorProblem  # noqa: F401

try:                                         # need optional pieces (OpenFOAM for the airfoil data, the heat models)
    from . import airfoil, inverse_heat  # noqa: F401
except ImportError:                          # pragma: no cover
    pass
