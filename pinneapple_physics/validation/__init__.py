"""Bounded numerical physics checks; separate from independent validation."""

from .fv_conservation import check_fv_conservation_1d

__all__ = ["check_fv_conservation_1d"]
