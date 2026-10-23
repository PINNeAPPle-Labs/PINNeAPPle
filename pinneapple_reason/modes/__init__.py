"""Built-in failure modes and countermeasures, one module per physical domain (imported on first use of the
library). Add a domain by writing a module like these and calling ``register_mode`` / ``register_countermeasure``."""
from . import corrosion, fluids, structural, thermal, water  # noqa: F401

__all__ = ["corrosion", "fluids", "structural", "thermal", "water"]
