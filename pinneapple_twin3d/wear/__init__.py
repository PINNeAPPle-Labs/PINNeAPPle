"""Generator of 3D wear (refractory / liner / pipe) digital twins from a spec.

    from pinneapple_twin3d.wear import presets, synthetic_campaign, export_wear_twin
    spec, hotspots = presets.get("bof")
    ds = synthetic_campaign(spec, hotspots)          # clearly labelled synthetic
    export_wear_twin(ds, "out/bof"); serve("out/bof")

Pipeline: ``VesselSpec`` -> ``WearDataset`` -> ``forecast`` -> ``optimize`` -> ``Scene`` (viewer) + report.
"""
from .model import MEASURED, SYNTHETIC, VesselSpec, WearDataset, ZoneSpec
from .forecast import backtest, forecast, forecast_history
from .optimize import OptimizerConfig, explain, optimize
from .calibrate import Calibration, StopRecord, calibrate
from .synthetic import HotSpot, synthetic_campaign
from . import presets
from .build import export_wear_twin, scene_from_wear

__all__ = ["ZoneSpec", "VesselSpec", "WearDataset", "SYNTHETIC", "MEASURED", "forecast", "forecast_history",
           "backtest", "OptimizerConfig", "optimize", "calibrate", "Calibration", "StopRecord", "explain", "HotSpot", "synthetic_campaign", "presets",
           "scene_from_wear", "export_wear_twin"]
