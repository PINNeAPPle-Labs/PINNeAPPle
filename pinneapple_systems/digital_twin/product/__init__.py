"""The closed digital-twin product: spec, commissioning, live run, evidence, drift policy.

``pinneapple_systems.digital_twin`` is the runtime — streams, filters, monitors. This
package turns it into a product with a lifecycle, entry criteria and an artifact to hand over:

=============================  ============================================================
stage                          what it produces
=============================  ============================================================
:mod:`.spec`                   :class:`TwinSpec` — the contract, with acceptance gates and a
                               digest recorded in every later artifact
:mod:`.model`                  the model contract (``model(X, theta)``) and
                               :class:`BiasModel` so an existing surrogate fits
:mod:`.assimilation`           :class:`LatentStateTwin` — EnKF over the state the model
                               cannot know
:mod:`.commission`             preflight checklist + held-out backtest
:mod:`.evidence`               :class:`TwinEvidence` — one JSON file: what passed, the
                               tracking numbers with their sources, and what was not checked
:mod:`.policy`                 :class:`DriftMonitor` — NORMAL → WATCH → ALARM → RECALIBRATE
:mod:`.runner`                 :func:`commission`, :func:`run_live`, :func:`export_scene`,
                               :func:`describe` (AAS / DTDL / NGSI-LD / Ditto / OPC UA)
=============================  ============================================================

Relationship to the runtime
---------------------------
``pinneapple_systems.operations`` (the library behind ``apps/twin_runtime``) is the **runtime**:
models, EnKF, forecasts, alerts, storage, MQTT/REST/OPC UA. This package is the
**commissioning layer** it does not have: a signable contract, a preflight that can refuse, a
held-out backtest against acceptance gates, and filed evidence. Use them together — commission
here, run there — or use this layer alone with any ``model(X, theta)``.

Quick start
-----------
>>> from pinneapple_systems.digital_twin.product import TwinSpec, commission, run_live
>>> spec = TwinSpec.from_yaml("exchanger.yaml")
>>> evidence = commission(spec, model, history)
>>> evidence.commissioned
True
>>> result = run_live(spec, model, frames, evidence=evidence)
>>> evidence.save("twin_evidence.json")
"""
from .assimilation import LatentStateTwin, TwinEstimate
from .commission import backtest, preflight
from .evidence import (
    FAIL,
    INFO,
    PASS,
    WARN,
    CheckItem,
    CheckReport,
    FieldMetrics,
    TrackingReport,
    TwinEvidence,
)
from .model import BiasModel, TwinModelError, ensure_twin_model
from .policy import DriftMonitor, DriftPolicy, TwinMode
from .runner import RunResult, commission, describe, export_scene, run_live
from .spec import (
    AcceptanceGates,
    FieldSpec,
    ModelSpec,
    SensorSpec,
    SpecError,
    TwinSpec,
)

__all__ = [
    # spec
    "TwinSpec", "FieldSpec", "SensorSpec", "ModelSpec", "AcceptanceGates", "SpecError",
    # model contract
    "BiasModel", "ensure_twin_model", "TwinModelError",
    # estimation
    "LatentStateTwin", "TwinEstimate",
    # commissioning
    "preflight", "backtest", "commission",
    # evidence
    "TwinEvidence", "CheckReport", "CheckItem", "TrackingReport", "FieldMetrics",
    "PASS", "WARN", "FAIL", "INFO",
    # policy
    "DriftPolicy", "DriftMonitor", "TwinMode",
    # runtime
    "run_live", "RunResult", "export_scene", "describe",
]
