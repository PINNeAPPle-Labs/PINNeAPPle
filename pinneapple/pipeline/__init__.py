"""``pinneapple.pipeline``: the surrogate-model pipeline (see :mod:`pinneapple_pipeline`).

>>> import pinneapple as pp
>>> pp.SurrogatePipeline.from_preset("burgers_in_viscosity")       # doctest: +SKIP

``python -m pinneapple.pipeline run experiment.yaml``.
"""
from pinneapple_pipeline import *  # noqa: F401,F403
from pinneapple_pipeline import __all__  # noqa: F401
