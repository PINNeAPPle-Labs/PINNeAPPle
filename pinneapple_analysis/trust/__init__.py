"""pinneapple_analysis.trust — pre-consumption trust scoring for a trained
model's own predictions. See ``trust_gate.py`` for how this differs from
``pinneapple_llm.guardrail.PhysicsGuardrail`` (verifies LLM-proposed
specs, not deployed-model predictions) and
``pinneapple_systems.digital_twin.monitoring.anomaly.MahalanobisDetector``
(scores live sensor innovation vectors, not a model's prediction).

``drift.py`` is the same question asked after deployment: a window of served traffic against the validation
baseline, with labels (error drift) and without (output and predictive-uncertainty drift), an envelope alarm from
the gate above, and the thresholds calibrated on a separate held-out split."""
from __future__ import annotations

from .trust_gate import TrustGate, TrustScore

__all__ = ["TrustGate", "TrustScore"]

from .trust_report import Check, TrustReport  # noqa: E402  (relatório de confiança para deploy)

__all__ += ["Check", "TrustReport"]

from .field_card import FieldTrustCard, field_checks, trust_card  # noqa: E402  (physics checks of a field on a grid)

__all__ += ["FieldTrustCard", "field_checks", "trust_card"]

from .drift import DriftBaseline, DriftMonitor, DriftReport, DriftThresholds, RetrainDecision  # noqa: E402

__all__ += ["DriftBaseline", "DriftMonitor", "DriftReport", "DriftThresholds", "RetrainDecision"]

from .corrections import (  # noqa: E402  (correção de especialista como checagem do relatório)
    Correction,
    CorrectionStore,
    correction_checks,
    with_corrections,
)

__all__ += ["Correction", "CorrectionStore", "correction_checks", "with_corrections"]

from .flow_card import (  # noqa: E402  (conservação de massa e energia medidas na previsão, #159)
    FlowSpec,
    energy_balance,
    flow_checks,
    mass_balance,
)

__all__ += ["FlowSpec", "energy_balance", "flow_checks", "mass_balance"]

from .signed_report import (  # noqa: E402  (relatório assinado com o modelo e as entradas, resíduo revisto ao abrir)
    ArtifactMismatch,
    ResidualMismatch,
    ResidualNotComputable,
    ResidualSpec,
    SignatureFailure,
    SignedReportError,
    SignedTrustReport,
    TrustReportVerification,
)

__all__ += ["ArtifactMismatch", "ResidualMismatch", "ResidualNotComputable", "ResidualSpec", "SignatureFailure",
            "SignedReportError", "SignedTrustReport", "TrustReportVerification"]
