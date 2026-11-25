"""A trust report signed together with the model and the inputs it describes, whose physics residual is recomputed on
load (#644, item B7 of ``docs/dev/LATENT_AND_CYBER_EVALUATION.md``).

A signature says *who* produced a report, not that the number in it is right: a signer can be honest and still record
a residual that the model does not have (an older tool, a run against other inputs, a typo in a hand-written table).
So a signed report here is three things, and verification re-establishes all three:

1. **the report** (``TrustReport.to_dict``: checks, weights, decision inputs), inside the DSSE-signed payload;
2. **the model and the inputs**, as their own bytes, with their SHA-256 recorded in the signed payload;
3. **the physics residual**, recomputed from the loaded model and the loaded inputs and compared with the value the
   signed payload records.

What each layer catches, on its own:

===================================  ==========================================================================
tampering with ...                   caught because
===================================  ==========================================================================
the report, its checks or the spec  the DSSE signature over the payload does not verify
the model weights or the inputs      the SHA-256 of those bytes is not the one in the signed payload
an internally consistent lie          the residual recomputed from the model and the inputs is not the recorded one
===================================  ==========================================================================

The third line is the reason this module exists: the first two are what ``pinneapple_security.signing`` and
``pinneapple_security.integrity`` already did. A bundle whose residual was fabricated *and* re-signed with the right
key still fails, because the recomputation does not read the recorded number.

The residual is the same one ``pinneapple_analysis.trust.field_card`` computes: three-point finite differences on the
model's own prediction over a tensor-product grid, rms over the interior divided by the rms of the sum of the terms
that must balance (each second derivative and the source). It is recomputed with that function, not reimplemented, so
the recorded value and the verified value cannot drift apart by a change of formula.

>>> import torch, pinneapple_security.signing as signing
>>> from pinneapple_analysis.trust import TrustReport, checagem_erro_relativo
>>> from pinneapple_analysis.trust.signed_report import ResidualSpec, SignedTrustReport
>>> model = HarmonicNet()                                            # any torch module: points -> values
>>> grid = {"x": [0.0, 0.25, 0.5, 0.75, 1.0], "y": [0.0, 0.25, 0.5, 0.75, 1.0]}
>>> report = TrustReport([checagem_erro_relativo("accuracy", 0.01, 0.02, critica=True)], model="harmonic")
>>> key = signing.generate_key()                                     # Ed25519; hmac_key() needs no dependency
>>> bundle = SignedTrustReport.create(report, model=model, inputs=grid, residual=ResidualSpec(pde="laplace"), key=key)
>>> bundle.save("harmonic.pptrust")
>>> checked = SignedTrustReport.load("harmonic.pptrust").verify(key.public())
>>> checked.residual_recomputed == checked.residual_recorded
True

What it does not do: it does not sandbox the bundle. Rebuilding the model imports the class the payload names (or
calls the ``model_factory`` the caller passes), so a bundle is exactly as safe as the key that signed it — verify with
the public key of a signer you trust, and pass ``model_factory`` when the class is not importable without arguments.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .field_card import field_checks
from .trust_report import TrustReport

__all__ = ["BUNDLE_FORMAT", "ResidualSpec", "SignedTrustReport", "TrustReportVerification",
           "SignedReportError", "SignatureFailure", "ArtifactMismatch", "ResidualNotComputable", "ResidualMismatch",
           "RESIDUAL_PDES"]

#: What the file says it is; a bundle of another format is refused rather than guessed at.
BUNDLE_FORMAT = "pinneapple.signed-trust-report/1"

#: The PDEs the grid residual knows (``field_card.field_checks``): homogeneous ``laplace``/``poisson`` and
#: ``heat`` with a diffusivity, each optional source given as data.
RESIDUAL_PDES = ("laplace", "poisson", "heat")

#: The name of the check that records a physics residual in a ``TrustReport`` (``from_trust_score`` uses it too).
RESIDUAL_CHECK = "residuo_pde"

_FILE_KEYS = frozenset({"format", "envelope", "model", "inputs"})
_PAYLOAD_KEYS = frozenset({"format", "created_at", "report", "model", "inputs", "residual"})


class SignedReportError(Exception):
    """Anything a signed report refuses, with the layer that refused it in the message."""


class SignatureFailure(SignedReportError):
    """The DSSE envelope does not verify: the signed payload changed, or it was signed by another key."""


class ArtifactMismatch(SignedReportError):
    """The model or the inputs are not the bytes that were signed."""


class ResidualMismatch(SignedReportError):
    """The recorded physics residual is not the one the signed model and inputs produce."""


class ResidualNotComputable(SignedReportError):
    """The residual could not be computed from these inputs (too few points, unknown PDE, no time axis)."""


# -------------------------------------------------------------------------------------------------- the residual
@dataclass(frozen=True)
class ResidualSpec:
    """What to recompute, as data: the residual is boilerplate of the equation, never code from the bundle.

    ``pde`` is one of :data:`RESIDUAL_PDES`; ``source`` is the optional ``f`` of ``laplace(u) = f`` as a grid-shaped
    array of numbers; ``alpha`` is the diffusivity of the heat equation, ``time_coord`` its time axis. ``rel_tol`` is
    how close the recomputed value must be to the recorded one — round-off of the same arithmetic in a different
    process, not a modelling tolerance.
    """

    pde: str = "laplace"
    source: Sequence[Any] | None = None
    alpha: float = 1.0
    time_coord: str = "t"
    rel_tol: float = 1e-6
    check: str = RESIDUAL_CHECK

    def __post_init__(self) -> None:
        if str(self.pde).lower() not in RESIDUAL_PDES:
            raise ValueError(f"residual pde must be one of {RESIDUAL_PDES}, got {self.pde!r}")
        if not 0.0 < float(self.rel_tol) < 1.0:
            raise ValueError(f"rel_tol must be in (0, 1), got {self.rel_tol!r}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"pde": str(self.pde).lower(), "alpha": float(self.alpha),
                               "time_coord": self.time_coord, "rel_tol": float(self.rel_tol), "check": self.check}
        if self.source is not None:
            out["source"] = _numbers(self.source)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualSpec:
        return cls(pde=data.get("pde", "laplace"), source=data.get("source"),
                   alpha=float(data.get("alpha", 1.0)), time_coord=str(data.get("time_coord", "t")),
                   rel_tol=float(data.get("rel_tol", 1e-6)), check=str(data.get("check", RESIDUAL_CHECK)))


def _numbers(value: Any) -> Any:
    """``value`` as nested lists of JSON numbers (a numpy array, a torch tensor or a sequence alike)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_numbers(v) for v in value]
    return float(value)


def _grid(inputs: Mapping[str, Any]) -> dict[str, list[float]]:
    """The tensor-product grid, validated: one strictly increasing 1-D axis per coordinate."""
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("inputs must be a mapping of coordinate name -> 1-D axis (the grid the residual is taken on)")
    out: dict[str, list[float]] = {}
    for name, axis in inputs.items():
        values = [float(v) for v in np.asarray(_numbers(axis), dtype=np.float64).ravel()]
        if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError(f"grid axis {name!r} must be strictly increasing with at least 2 points")
        out[str(name)] = values
    return out


def _coords(grid: Mapping[str, Sequence[float]]) -> np.ndarray:
    axes = [np.asarray(grid[n], dtype=np.float64) for n in grid]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)


def _predict(model: Any, grid: Mapping[str, Sequence[float]]) -> np.ndarray:
    """The model's own prediction on the grid, as a field of the grid's shape.

    The model takes an ``(N, ndim)`` tensor of points and returns ``(N,)`` or ``(N, 1)`` (a ``Solution``-shaped
    object with ``.y`` is unwrapped). Anything else is refused by shape, with the shape in the message.
    """
    import torch

    pts = _coords(grid)
    dtype = _prepared_dtype(model)
    with torch.no_grad():
        out = model(torch.as_tensor(pts, dtype=dtype))
    if hasattr(out, "y"):
        out = out.y
    values = np.asarray((out.detach().cpu().numpy() if hasattr(out, "detach") else out), dtype=np.float64).ravel()
    shape = tuple(len(grid[n]) for n in grid)
    if values.size != int(np.prod(shape)):
        raise ResidualNotComputable(f"the model returned {values.size} values for a grid of {shape} "
                                    f"({int(np.prod(shape))} points): it must predict one value per point")
    return values.reshape(shape)


def _prepared_dtype(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        return parameter.dtype
    return torch.float64


def _prepared(model: Any) -> Any:
    """A copy in ``eval`` mode and float64: the same arithmetic at signing time and at verification time."""
    import torch

    try:
        copy_model = copy.deepcopy(model)
    except Exception:                                   # noqa: BLE001 - a module that refuses deepcopy is used as is
        model.eval()
        return model
    return copy_model.to(torch.float64).eval()


def residual_value(model: Any, grid: Mapping[str, Sequence[float]], spec: ResidualSpec) -> float:
    """The physics residual of the model's own prediction on ``grid``, by the field card's grid check.

    Same three-point finite differences and same normalisation as ``field_checks`` (rms of the interior residual over
    the rms of the sum of the terms that must balance), so this number means what a trust card's "PDE residual" means.
    Raises :class:`ResidualNotComputable` when the check cannot run, quoting the check's own reason.
    """
    prepared = _prepared(model)
    values = _predict(prepared, grid)
    checks = field_checks(dict(grid), values, pde=spec.pde, source=spec.source, alpha=spec.alpha,
                          time_coord=spec.time_coord)
    check = next((c for c in checks if c.name == "PDE residual"), None)
    if check is None or check.value is None:
        raise ResidualNotComputable(f"the {spec.pde} residual could not be computed on this grid: "
                                    f"{getattr(check, 'detail', '') or 'no residual check ran'}")
    return float(check.value)


# -------------------------------------------------------------------------------------------------- the bundle
@dataclass
class TrustReportVerification:
    """What verification established: the report as signed, and both residual numbers with their difference."""

    report: TrustReport
    residual_recorded: float
    residual_recomputed: float
    rel_error: float
    model_class: str
    keyid: str = ""
    created_at: str = ""
    spec: ResidualSpec = field(default_factory=ResidualSpec)

    @property
    def ok(self) -> bool:
        return self.rel_error <= self.spec.rel_tol

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "residual_recorded": self.residual_recorded,
                "residual_recomputed": self.residual_recomputed, "rel_error": self.rel_error,
                "rel_tol": self.spec.rel_tol, "model_class": self.model_class, "keyid": self.keyid,
                "created_at": self.created_at, "decisao": self.report.decisao}


class SignedTrustReport:
    """A ``TrustReport`` plus the model and inputs it describes, signed as one payload and re-verified on load.

    ``create`` computes the residual itself and refuses to sign a report that already records a different one;
    ``verify`` reloads the model and the inputs from the signed bytes, recomputes the residual and compares.
    """

    def __init__(self, *, envelope: Mapping[str, Any], model_blob: bytes, inputs_blob: bytes):
        self.envelope = dict(envelope)
        self.model_blob = bytes(model_blob)
        self.inputs_blob = bytes(inputs_blob)

    # -- creation ---------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, report: TrustReport, *, model: Any, inputs: Mapping[str, Any],
               key: Any, residual: ResidualSpec | None = None) -> SignedTrustReport:
        """Sign ``report`` with the model and the inputs, recording the residual the two of them produce.

        ``key`` is a ``pinneapple_security.signing.SigningKey`` (Ed25519, or HMAC with ``hmac_key()``). The model must
        be a ``torch.nn.Module`` — it is serialised as a state dict and rebuilt by class path (or by
        ``model_factory`` at verification), and its prediction on the signed grid is what the residual is taken on.
        """
        from pinneapple_security.signing import sign_json

        spec = residual or ResidualSpec()
        if not isinstance(report, TrustReport):
            raise TypeError(f"a signed trust report signs a TrustReport, got {type(report).__name__}")
        blob = _state_bytes(model)
        grid = _grid(inputs)
        recorded = residual_value(model, grid, spec)
        in_report = next((c for c in report.checks if c.name == spec.check and c.value is not None), None)
        if in_report is not None and not _close(float(in_report.value), recorded, spec.rel_tol):
            raise ResidualMismatch(
                f"the report records {spec.check} = {float(in_report.value):.12g} but the model and the inputs give "
                f"{recorded:.12g}: not signing a report whose own number the model does not have")
        manifest = {
            "format": BUNDLE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "report": report.to_dict(),
            "model": {"class": _class_path(model), "sha256": _sha256(blob), "n_bytes": len(blob),
                      "n_parameters": int(_n_parameters(model))},
            "inputs": {"sha256": _sha256(_inputs_bytes(grid)), "grid": {k: len(v) for k, v in grid.items()}},
            "residual": {"spec": spec.to_dict(), "value": recorded},
        }
        return cls(envelope=sign_json(manifest, key), model_blob=blob, inputs_blob=_inputs_bytes(grid))

    # -- persistence -------------------------------------------------------------------------------------------
    def dumps(self) -> str:
        """The bundle as one text document: the signed payload, the model and the inputs, nothing else."""
        return json.dumps({"format": BUNDLE_FORMAT, "envelope": self.envelope,
                           "model": _b64(self.model_blob), "inputs": _b64(self.inputs_blob)},
                          indent=1, sort_keys=True) + "\n"

    def save(self, path: str | os.PathLike[str]) -> str:
        """Write the bundle (``.pptrust`` by convention). Returns the path written."""
        text = self.dumps()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return os.fspath(path)

    @classmethod
    def loads(cls, text: str) -> SignedTrustReport:
        """Parse a bundle without verifying it (``verify`` is the only thing that trusts anything)."""
        doc = _json_or_refuse(text)
        if not isinstance(doc, dict) or set(doc) != _FILE_KEYS:
            got = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
            raise SignedReportError(f"not a {BUNDLE_FORMAT} bundle: expected the keys {sorted(_FILE_KEYS)}, got {got}")
        if doc["format"] != BUNDLE_FORMAT:
            raise SignedReportError(f"bundle format {doc['format']!r}, this code reads {BUNDLE_FORMAT!r}")
        return cls(envelope=doc["envelope"], model_blob=_unb64(doc["model"]), inputs_blob=_unb64(doc["inputs"]))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SignedTrustReport:
        with open(path, encoding="utf-8") as f:
            return cls.loads(f.read())

    # -- verification ------------------------------------------------------------------------------------------
    def verify(self, key: Any, *, model_factory: Callable[[], Any] | None = None) -> TrustReportVerification:
        """Verify the signature, the artifacts and the physics, in that order. Returns what was checked.

        ``key`` is the ``VerifyKey`` (``key.public()``) or the signing key. ``model_factory`` rebuilds the model when
        its class needs constructor arguments; otherwise the class path recorded in the payload is imported. Every
        failure raises a :class:`SignedReportError` naming the layer that refused it.
        """
        from pinneapple_security.signing import verify_json

        try:
            manifest = verify_json(self.envelope, key)
        except Exception as exc:                        # noqa: BLE001 - any signature failure is one answer
            raise SignatureFailure(f"the signed payload does not verify: {type(exc).__name__}: {exc}") from exc
        _require_canonical(self.envelope)
        if not isinstance(manifest, dict) or set(manifest) != _PAYLOAD_KEYS:
            got = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
            raise SignatureFailure(f"signed payload has the wrong shape: expected {sorted(_PAYLOAD_KEYS)}, got {got}")
        if manifest["format"] != BUNDLE_FORMAT:
            raise SignatureFailure(f"signed payload format {manifest['format']!r}, this code reads {BUNDLE_FORMAT!r}")
        model_part, residual_part = manifest["model"], manifest["residual"]
        if not (isinstance(model_part, dict) and isinstance(residual_part, dict)
                and isinstance(residual_part.get("spec"), dict) and "value" in residual_part
                and isinstance(manifest.get("report"), dict)):
            raise SignatureFailure("signed payload is missing its model, residual or report block")

        self._require_sha(self.model_blob, model_part, "model")
        self._require_sha(self.inputs_blob, manifest["inputs"], "inputs")

        grid = _signed_grid(self.inputs_blob)
        spec = ResidualSpec.from_dict(residual_part["spec"])
        model = _rebuild(str(model_part.get("class", "")), self.model_blob, model_factory)
        recomputed = residual_value(model, grid, spec)
        recorded = float(residual_part["value"])
        if not _close(recorded, recomputed, spec.rel_tol):
            raise ResidualMismatch(
                f"the signed report records the {spec.pde} residual as {recorded:.12g}; the signed model and inputs "
                f"give {recomputed:.12g} (relative difference {_rel(recorded, recomputed):.3g} > {spec.rel_tol:g}). "
                f"The signature is intact and the bytes are the signed ones: the number does not match the physics")

        report = TrustReport.from_dict(manifest["report"])
        in_report = next((c for c in report.checks if c.name == spec.check and c.value is not None), None)
        if in_report is not None and not _close(float(in_report.value), recomputed, spec.rel_tol):
            raise ResidualMismatch(f"the report's own {spec.check} check says {float(in_report.value):.12g}, the "
                                   f"recomputation says {recomputed:.12g}")
        return TrustReportVerification(report=report, residual_recorded=recorded, residual_recomputed=recomputed,
                                       rel_error=_rel(recorded, recomputed),
                                       model_class=str(model_part.get("class", "")),
                                       keyid=_keyid(self.envelope), created_at=str(manifest["created_at"]),
                                       spec=spec)

    @staticmethod
    def _require_sha(blob: bytes, declared: Mapping[str, Any], what: str) -> None:
        want = str((declared or {}).get("sha256") or "")
        got = _sha256(blob)
        if not want or got != want:
            raise ArtifactMismatch(f"{what}: the bytes in this bundle are not the ones that were signed "
                                   f"(signed sha256 {(want or '<none>')[:16]}…, found {got[:16]}…, {len(blob)} bytes)")


# -------------------------------------------------------------------------------------------------- helpers
def _json_or_refuse(text: Any, what: str = "the bundle") -> Any:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="strict")
    try:
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignedReportError(f"{what} is not readable JSON ({exc}); nothing here is trusted") from exc


def _signed_grid(blob: bytes) -> dict[str, list[float]]:
    """The grid out of the signed inputs, with every refusal in this module's own words: the bytes are the signed ones
    by the time this runs, so a shape that is not a grid means the signer wrote something this code cannot read."""
    doc = _json_or_refuse(blob, "the signed inputs")
    if not isinstance(doc, dict) or not isinstance(doc.get("grid"), dict):
        raise SignedReportError(f"the signed inputs are not {{'grid': {{coord: axis}}}}: got {type(doc).__name__}")
    try:
        return _grid(doc["grid"])
    except ValueError as exc:
        raise ResidualNotComputable(f"the signed grid cannot carry the residual: {exc}") from exc


def _state_bytes(model: Any) -> bytes:
    import io

    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"the signed model must be a torch.nn.Module (serialised as a state dict), got "
                        f"{type(model).__name__}")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _rebuild(class_path: str, blob: bytes, factory: Callable[[], Any] | None) -> Any:
    import io

    import torch

    try:
        state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as exc:                            # noqa: BLE001 - a state dict that does not load is a refusal
        raise ArtifactMismatch(f"the signed model bytes are not a state dict: {type(exc).__name__}: {exc}") from exc
    if factory is not None:
        model = factory()
    else:
        target = _import_class(class_path)
        try:
            model = target()
        except TypeError as exc:
            raise SignedReportError(f"{class_path} cannot be built without arguments ({exc}): pass "
                                    f"model_factory=lambda: {class_path.split(':')[-1]}(...) to verify()") from exc
    try:
        model.load_state_dict(state, strict=True)       # a missing or extra key is a refusal, not a partial load
    except Exception as exc:                            # noqa: BLE001 - the signed weights do not fit the class
        raise ArtifactMismatch(f"{class_path} does not take the signed state dict: {type(exc).__name__}: {exc}") from exc
    return model


def _import_class(class_path: str) -> Any:
    module_name, _, qualname = str(class_path).partition(":")
    if not module_name or not qualname:
        raise SignedReportError(f"model class {class_path!r} is not 'module:Class'")
    hint = " (a class defined inside a function cannot be imported by path)" if "<locals>" in qualname else ""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SignedReportError(f"the signed model class {class_path!r} cannot be imported here ({exc}){hint}: pass "
                                f"model_factory to verify()") from exc
    target: Any = module
    for part in qualname.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise SignedReportError(f"the signed model class {class_path!r} does not resolve here "
                                    f"({exc}){hint}: pass model_factory to verify()") from exc
    return target


def _class_path(model: Any) -> str:
    cls = type(model)
    return f"{cls.__module__}:{getattr(cls, '__qualname__', cls.__name__)}"


def _n_parameters(model: Any) -> int:
    try:
        return sum(int(p.numel()) for p in model.parameters())
    except Exception:                                   # noqa: BLE001 - a module without parameters is still a model
        return 0


def _inputs_bytes(grid: Mapping[str, Sequence[float]]) -> bytes:
    """The grid as written, axis order preserved: the axes are the dimensions of the field, so a reordering by a
    serialiser would silently change which derivative is taken along which axis."""
    return json.dumps({"grid": {k: list(v) for k, v in grid.items()}},
                      separators=(",", ":")).encode("utf-8")


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def _unb64(text: Any) -> bytes:
    """Decode base64 and refuse anything but the canonical encoding.

    ``b64decode(validate=True)`` rejects a byte outside the alphabet, but it ignores the unused bits of the last
    character before the padding: a change there decodes to the same bytes. Requiring the canonical form is what
    makes the bundle byte-exact — a file whose text changed is refused even when the change decoded to nothing.
    """
    raw = str(text)
    try:
        decoded = base64.b64decode(raw.encode("ascii"), validate=True)
    except Exception as exc:                            # noqa: BLE001 - a corrupted blob is a refusal, not a crash
        raise ArtifactMismatch(f"a signed artifact is not valid base64: {type(exc).__name__}: {exc}") from exc
    if _b64(decoded) != raw:
        raise ArtifactMismatch("a signed artifact is not in the canonical base64 that was written: its text changed "
                               "after signing")
    return decoded


def _close(a: float, b: float, rel_tol: float) -> bool:
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= rel_tol * scale


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _keyid(envelope: Mapping[str, Any]) -> str:
    sigs = envelope.get("signatures") or []
    return str(sigs[0].get("keyid", "")) if sigs else ""


def _require_canonical(envelope: Mapping[str, Any]) -> None:
    """The envelope's own base64 must be the canonical encoding of what it carries (see :func:`_unb64`)."""
    fields = [envelope.get("payload")] + [s.get("sig") for s in (envelope.get("signatures") or [])]
    for text in fields:
        _unb64(text)
