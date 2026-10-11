"""Tribology: sliding wear of a bar against a counterface (Archard's law with contact-pressure redistribution)."""
from .archard import WEAR_MATERIALS, BarWear, WearMaterial, winkler_parabolic_contact

__all__ = ["WEAR_MATERIALS", "WearMaterial", "BarWear", "winkler_parabolic_contact"]
