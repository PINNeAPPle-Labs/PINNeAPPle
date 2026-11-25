"""The Digital Twin Library: reference models of engineering systems, one per domain.

What this package is
--------------------
Each module here is a *reference model* of a real system written to be read, run and checked
in seconds: FPSO and subsea production, a blast furnace, a chemical plant, a reusable
booster, a lunar base, a gas pipeline network, a compression station, a green hydrogen plant, a
desalination train, an industrial refrigeration plant, a data centre and a carbon capture island. They share :class:`~.base.ReferenceModel`, so every one of them takes
SI parameters with declared ranges, returns a machine-readable :class:`~.base.TwinRun` with
its conservation balances and its invariants, and states the limits of its own physics.

Where a twin goes after this
----------------------------
The fidelity ladder of the Digital Twin Library issues is 0D -> 1D -> 2D/3D -> trained
surrogate -> commissioning. This package is the bottom rung, and it is the rung the others
need: a model cheap enough to run in CI is also the model that generates the solver data a
surrogate is trained on, and the one a
:class:`~pinneapple_systems.digital_twin.product.spec.TwinSpec` can wrap to be commissioned
against measured history.

Use
---
>>> from pinneapple_systems.digital_twin.library import catalogue, run, describe
>>> [entry.slug for entry in catalogue()]
['blast_furnace', 'chemical_plant', 'compression_station', 'data_center_cooling',
 'desalination_plant', 'fpso_subsea', 'green_hydrogen_plant', 'industrial_refrigeration',
 'lunar_base', 'pipeline_network', 'reusable_booster']
>>> result = run("fpso_subsea")               # the declared default scenario
>>> result.ok
True
>>> print(result.report())
>>> print(describe("fpso_subsea"))            # parameters, assumptions, limits, sources
"""

from __future__ import annotations

from .base import (
    Balance,
    BalanceFailed,
    Check,
    LibraryError,
    Parameter,
    ParameterError,
    Quantity,
    ReferenceModel,
    TwinRun,
    close,
    relative_error,
)
from .blast_furnace import BlastFurnace
from .carbon_capture import CarbonCapturePlant
from .chemical_plant import ChemicalPlant
from .compression_station import CompressionStation
from .data_center import DataCentre
from .desalination import DesalinationPlant
from .fpso import FPSOSubsea
from .green_hydrogen import GreenHydrogenPlant
from .lunar_base import LunarBase
from .pipeline_network import PipelineNetwork
from .refrigeration import RefrigerationPlant
from .registry import TWINS, TwinEntry, build, catalogue, compare, describe, run
from .reusable_booster import ReusableBooster
from .scenarios import (
    SCENARIOS,
    Scenario,
    load_scenarios,
    manifest,
    run_scenario,
    save_scenarios,
    scenario_names,
)

__all__ = [
    # contract
    "ReferenceModel", "TwinRun", "Parameter", "Quantity", "Balance", "Check",
    "LibraryError", "ParameterError", "BalanceFailed", "close", "relative_error",
    # registry, scenarios and status
    "TWINS", "TwinEntry", "catalogue", "build", "run", "describe", "compare",
    "Scenario", "SCENARIOS", "scenario_names", "run_scenario", "manifest", "save_scenarios",
    "load_scenarios",
    # systems
    "FPSOSubsea", "BlastFurnace", "ChemicalPlant", "ReusableBooster", "LunarBase",
    "PipelineNetwork", "CompressionStation", "GreenHydrogenPlant", "DesalinationPlant",
    "RefrigerationPlant", "DataCentre", "CarbonCapturePlant",
]
