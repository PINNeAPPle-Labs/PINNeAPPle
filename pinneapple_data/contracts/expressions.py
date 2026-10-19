"""Safe evaluation of constraint expressions: re-exported from :mod:`pinneapple_data.expressions`, which needs only
NumPy (``pinneapple check`` imports it without the contract models and their pydantic dependency)."""
from pinneapple_data.expressions import (  # noqa: F401
    FUNCTIONS,
    MAX_LENGTH,
    ExpressionError,
    check,
    evaluate,
    names_in,
    names_in_tree,
)
