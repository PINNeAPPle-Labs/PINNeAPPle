"""Catalog of what PINNeAPPle knows about, with sources.

- ``resources``: public datasets, CAD/geometry sets, pretrained models and
  hosted benchmarks, each with a checked license and a license gate.
- ``methods``: every solver, training method, equation and problem in the
  library, with its code location, its references and its validation status.
- ``stages``: the same question asked the other way round — at each step of
  the physics-AI workflow, which *open tools* exist and which public
  *datasets* feed it, with the licence read and the wiring checked.
"""
from . import stages
from .resources import Resource, fetch, get_resource, list_resources, require_allowed

__all__ = ["Resource", "fetch", "get_resource", "list_resources", "require_allowed", "stages"]
