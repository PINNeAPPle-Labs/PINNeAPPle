"""The twin contract: what a digital twin is asked to deliver, signed before it runs.

A ``TwinSpec`` is the artifact a client and an engineer agree on *before* commissioning:
which quantities the twin estimates, which sensors measure them and in which units, which
model is behind it and the accuracy it claims, and — the important part — the
:class:`AcceptanceGates` that decide whether the twin is allowed to go live.

Without gates, "the twin works" is an opinion. With them it is a number that either
clears a threshold or does not, and the spec digest records which revision was accepted.

Limits
------
The spec does not validate the physics; it validates that a claim exists and can be
falsified. A spec whose gates are too loose will commission a bad twin honestly — set the
gates from the engineering requirement (the temperature the process may not exceed, the
thickness that forces a stop), not from what the model already achieves.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "SpecError",
    "FieldSpec",
    "SensorSpec",
    "ModelSpec",
    "AcceptanceGates",
    "TwinSpec",
    "FIELD_KINDS",
]

FIELD_KINDS = ("state", "parameter", "telemetry")
_STATUSES = ("none", "ekf", "enkf")


class SpecError(ValueError):
    """The spec is not usable: a missing claim, a contradiction, an unobservable field."""


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """One quantity the twin estimates.

    ``kind``
        ``state`` — an unmeasured quantity the twin must infer (fouling resistance,
        wall thickness, reaction extent). These are what a twin is *for*.
        ``parameter`` — a design/configuration value the twin carries but does not infer.
        ``telemetry`` — a measured quantity the twin predicts so it can be compared.

    ``alarm_above`` / ``alarm_below``
        The engineering limit for this field. When present, commissioning measures how far
        *ahead* of the measured crossing the twin raised its own alarm — the only number
        that makes a twin worth running.
    """

    name: str
    unit: str
    kind: str = "state"
    lo: float | None = None
    hi: float | None = None
    alarm_above: float | None = None
    alarm_below: float | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise SpecError(f"field {self.name!r}: use a Python identifier (letters, digits, _)")
        if not self.unit:
            raise SpecError(f"field {self.name!r}: unit is required, even when dimensionless ('-')")
        if self.kind not in FIELD_KINDS:
            raise SpecError(f"field {self.name!r}: kind must be one of {FIELD_KINDS}")
        if self.lo is not None and self.hi is not None and not self.lo < self.hi:
            raise SpecError(f"field {self.name!r}: lo ({self.lo}) must be below hi ({self.hi})")
        if self.alarm_above is not None and self.alarm_below is not None:
            raise SpecError(f"field {self.name!r}: declare alarm_above or alarm_below, not both")

    @property
    def span(self) -> float | None:
        """Physical span (hi − lo), the denominator that makes errors comparable."""
        if self.lo is None or self.hi is None:
            return None
        return float(self.hi - self.lo)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class SensorSpec:
    """A measurement available to the twin, with the envelope that makes it checkable.

    ``driver``
        Marks the channel as an *input* to the model rather than something the twin
        predicts: the inlet temperature, the mass flow, the ambient. Drivers are appended to
        the model input matrix after the coordinates, in declaration order, and their values
        change every frame — which is what lets one twin cover a whole operating envelope
        instead of a single steady point. A driver needs no ``field``.

    ``at``
        Coordinates where the model is evaluated to compare against this sensor. A twin
        that cannot say *where* it is predicting cannot be compared to a thermocouple.

    ``holdout``
        Marks the sensor as ground truth for commissioning: it is never assimilated, only
        compared. Without at least one holdout, a backtest measures nothing but how well
        the filter fits the data it was fed.

    ``max_gap_s``
        The longest silence accepted from this channel. Longer gaps fail preflight, because
        a twin that stops hearing a sensor keeps predicting confidently.
    """

    sensor_id: str
    unit: str
    field: str = ""
    lo: float = 0.0
    hi: float = 1.0
    at: tuple[float, ...] = ()
    max_gap_s: float = 3600.0
    noise_sigma: float | None = None
    holdout: bool = False
    driver: bool = False

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise SpecError("sensor_id is required")
        if not self.unit:
            raise SpecError(f"sensor {self.sensor_id!r}: unit is required")
        if not self.lo < self.hi:
            raise SpecError(f"sensor {self.sensor_id!r}: envelope lo ({self.lo}) must be below hi ({self.hi})")
        if self.max_gap_s <= 0:
            raise SpecError(f"sensor {self.sensor_id!r}: max_gap_s must be positive")
        if self.noise_sigma is not None and self.noise_sigma <= 0:
            raise SpecError(f"sensor {self.sensor_id!r}: noise_sigma must be positive")
        if self.driver:
            if self.holdout:
                raise SpecError(
                    f"sensor {self.sensor_id!r}: a driver is an input, it cannot also be the "
                    "held-out ground truth"
                )
        elif not self.field:
            raise SpecError(
                f"sensor {self.sensor_id!r}: name the field it measures, "
                "or mark it driver=True if it is a model input"
            )

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["at"] = list(self.at)
        return d


@dataclass(frozen=True)
class ModelSpec:
    """The model behind the twin, and the accuracy it claims.

    ``accuracy_floor``
        The absolute error, in field units, that the model's author claims on unseen
        operating points. Commissioning reports it next to the measured tracking error so
        an improvement (or a regression) is visible instead of assumed.
    """

    model_id: str
    version: str = ""
    accuracy_floor: float | None = None
    validity: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise SpecError("model_id is required")
        if self.accuracy_floor is not None and self.accuracy_floor <= 0:
            raise SpecError("accuracy_floor must be positive (it is an error magnitude)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "accuracy_floor": self.accuracy_floor,
            "validity": {k: list(v) for k, v in self.validity.items()},
        }


@dataclass(frozen=True)
class AcceptanceGates:
    """The thresholds that decide whether a twin goes live.

    Errors are expressed as a fraction of the field span so a gate means the same thing on
    a 5 °C and a 500 °C process. ``min_coverage`` is the fraction of *held-out* observations
    that must fall inside the twin's own 95 % band — the check that catches a confident twin
    that is confidently wrong.
    """

    max_rmse_frac: float = 0.05
    max_bias_frac: float = 0.02
    min_coverage: float = 0.90
    min_alarm_lead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_rmse_frac <= 0:
            raise SpecError("max_rmse_frac must be positive")
        if self.max_bias_frac < 0:
            raise SpecError("max_bias_frac cannot be negative")
        if not 0.0 < self.min_coverage <= 1.0:
            raise SpecError("min_coverage must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwinSpec:
    """A complete, checkable definition of one digital twin.

    Build it directly, from a dict, or from YAML::

        spec = TwinSpec.from_yaml("twin.yaml")
        report = commission(spec, history, model)
        if not report.passed:
            ...   # the reasons are in report.items, each with a suggested fix
    """

    asset_id: str
    asset_name: str
    asset_type: str
    fields: Sequence[FieldSpec]
    sensors: Sequence[SensorSpec]
    model: ModelSpec
    asset_unit: str = ""
    coord_names: Sequence[str] = ()
    sampling_s: float = 60.0
    horizon_s: float | None = None
    warn_horizon_s: float = 6 * 3600.0
    assimilation: str = "enkf"
    gates: AcceptanceGates = field(default_factory=AcceptanceGates)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise SpecError("asset_id is required: it is the key every other artifact is filed under")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "sensors", tuple(self.sensors))
        object.__setattr__(self, "coord_names", tuple(self.coord_names))
        if not self.fields:
            raise SpecError("a twin with no fields estimates nothing")
        if not self.sensors:
            raise SpecError("a twin with no sensors cannot be checked against reality")
        if self.assimilation not in _STATUSES:
            raise SpecError(f"assimilation must be one of {_STATUSES}")
        if self.sampling_s <= 0:
            raise SpecError("sampling_s must be positive")
        if self.warn_horizon_s <= 0:
            raise SpecError("warn_horizon_s must be positive")

        field_names = [f.name for f in self.fields]
        if len(set(field_names)) != len(field_names):
            raise SpecError(f"duplicate field names in {field_names}")

        sensor_ids = [s.sensor_id for s in self.sensors]
        dupes = {s for s in sensor_ids if sensor_ids.count(s) > 1}
        if dupes:
            raise SpecError(f"duplicate sensor_id(s): {sorted(dupes)}")

        predicted = [s for s in self.sensors if not s.driver]
        unknown = sorted({s.field for s in predicted} - set(field_names))
        if unknown:
            raise SpecError(f"sensor(s) reference field(s) that do not exist: {unknown}")

        arities = {len(s.at) for s in predicted}
        if len(arities) > 1:
            raise SpecError(
                f"predicted sensors disagree on the number of coordinates {sorted(arities)}; "
                "the model is evaluated on one grid for all of them"
            )
        n_coords = arities.pop() if arities else 0
        if self.coord_names and len(self.coord_names) != n_coords:
            raise SpecError(
                f"coord_names has {len(self.coord_names)} entries but sensors carry {n_coords} "
                "coordinate(s)"
            )

        # Observability: a *predicted* field no sensor sees, or seen only by a holdout, is an
        # estimate nobody can update. Parameters are exempt by definition — they are inferred
        # from the predicted fields, and P02b in preflight checks they are identifiable.
        measurable = [f.name for f in self.fields if f.kind != "parameter"]
        observed = {s.field for s in predicted}
        unobserved = [f for f in measurable if f not in observed]
        if unobserved:
            raise SpecError(
                f"field(s) {unobserved} have no sensor: a twin must be observable to be verifiable"
            )
        blind = [f for f in measurable if not self.sensors_for(f, holdout=False)]
        if blind:
            raise SpecError(
                f"field(s) {blind} are seen only by holdout sensors, so the twin could never "
                "correct them"
            )

        if not any(s.holdout for s in self.sensors):
            raise SpecError(
                "mark at least one sensor as holdout: without held-out ground truth, "
                "commissioning measures only how well the filter fits its own input"
            )

    # -- convenience -------------------------------------------------------

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def driver_sensors(self) -> tuple[SensorSpec, ...]:
        return tuple(s for s in self.sensors if s.driver)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Model input columns, in order: coordinates, then drivers as declared."""
        return tuple(self.coord_names) + tuple(s.sensor_id for s in self.driver_sensors)

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"field {name!r} is not in this spec (have {list(self.field_names)})")

    def sensors_for(
        self,
        field_name: str,
        *,
        holdout: bool | None = None,
    ) -> tuple[SensorSpec, ...]:
        out = tuple(
            s for s in self.sensors
            if not s.driver and s.field == field_name
        )
        if holdout is not None:
            out = tuple(s for s in out if s.holdout is holdout)
        return out

    def assimilated_sensors(self) -> tuple[SensorSpec, ...]:
        return tuple(s for s in self.sensors if not s.holdout and not s.driver)

    def holdout_sensors(self) -> tuple[SensorSpec, ...]:
        return tuple(s for s in self.sensors if s.holdout)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": {
                "id": self.asset_id,
                "name": self.asset_name,
                "type": self.asset_type,
                "unit": self.asset_unit,
            },
            "fields": [f.to_dict() for f in self.fields],
            "sensors": [s.to_dict() for s in self.sensors],
            "model": self.model.to_dict(),
            "coord_names": list(self.coord_names),
            "sampling_s": self.sampling_s,
            "horizon_s": self.horizon_s,
            "warn_horizon_s": self.warn_horizon_s,
            "assimilation": self.assimilation,
            "gates": self.gates.to_dict(),
            "notes": self.notes,
        }

    def digest(self) -> str:
        """Stable SHA-256 of the spec, short form. Name this in every report you file."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TwinSpec":
        asset = data.get("asset") or {}
        gates = data.get("gates") or {}
        model = data.get("model") or {}
        try:
            return cls(
                asset_id=asset["id"],
                asset_name=asset.get("name", asset["id"]),
                asset_type=asset.get("type", "asset"),
                asset_unit=asset.get("unit", ""),
                fields=tuple(FieldSpec(**f) for f in data["fields"]),
                sensors=tuple(
                    SensorSpec(**{**s, "at": tuple(s.get("at") or ())}) for s in data["sensors"]
                ),
                model=ModelSpec(
                    model_id=model["model_id"],
                    version=model.get("version", ""),
                    accuracy_floor=model.get("accuracy_floor"),
                    validity={k: tuple(v) for k, v in (model.get("validity") or {}).items()},
                ),
                coord_names=tuple(data.get("coord_names") or ()),
                sampling_s=float(data.get("sampling_s", 60.0)),
                horizon_s=data.get("horizon_s"),
                warn_horizon_s=float(data.get("warn_horizon_s", 6 * 3600.0)),
                assimilation=data.get("assimilation", "enkf"),
                gates=AcceptanceGates(**gates),
                notes=data.get("notes", ""),
            )
        except KeyError as exc:
            raise SpecError(f"spec is missing required key {exc.args[0]!r}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TwinSpec":
        """Load from YAML. PyYAML is imported here only, so the rest of the module is stdlib."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("PyYAML is required to load a spec from YAML: pip install pyyaml") from exc
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required to write a spec to YAML: pip install pyyaml") from exc
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    # -- edits keep the gates honest ---------------------------------------

    def with_gates(self, **changes: Any) -> "TwinSpec":
        """Return a copy with new gates. Editing a gate is a decision; it gets a new digest."""
        return replace(self, gates=replace(self.gates, **changes))
