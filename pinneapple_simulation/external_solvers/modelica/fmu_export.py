"""Export a validated surrogate as an FMU: a package another simulator loads and co-simulates.

The library already reads FMUs (:mod:`fmi_runner`, :mod:`fmu_inspect`); this is the reverse path. A surrogate that
passed its validation becomes an FMI 2.0 Co-Simulation archive that Simulink, OpenModelica, Dymola or fmpy can load,
with its validity envelope carried inside the package and in the model description, so the tool on the other side
receives the numbers *and* the range the numbers are allowed to answer.

What the package contains
-------------------------

``modelDescription.xml``
    FMI 2.0, one ``CoSimulation`` interface. Every input is a ``causality="input"`` Real with its validated
    ``min``/``max`` as attributes; every output is a ``causality="output"`` Real; ``envelope_ok`` is a Boolean
    output that reads 0 when the inputs handed to the slave are outside the validated box. The envelope, the
    validation decision and the hashes of the surrogate's coefficient tables are in ``VendorAnnotations`` (the slot
    FMI 2.0 reserves for exactly this), so an importer that only parses the XML still sees the range.
``sources/surrogate_fmu.c``
    the whole slave: the FMI 2.0 Co-Simulation entry points with the surrogate's coefficients baked in as
    ``static const double`` tables. No FMI header is needed -- the ABI types are declared in the file -- so `cc` is
    the only thing a rebuild takes.
``resources/validated_surrogate.json``
    the same coefficients, the envelope and the evidence as data, for a rebuild on another platform, for a reader
    that does not compile, and for whoever has to check that the binary and the validation are the same model.
``binaries/<platform>/<identifier>.<ext>``
    the compiled slave, when this machine has a compiler that can produce it (:func:`export_fmu` with
    ``build=True``).

What is *not* produced, and why
-------------------------------

- **A binary for another platform than the one that built the package.** An FMU ships one binary per target; the
  package written here carries the binary of the machine that ran it. Windows and Linux binaries need a compiler (or
  a cross toolchain) on that platform: run the same export there, or rebuild from ``sources/`` and ``resources/``.
- **FMI 3.0.** The 3.0 interface changes the value-reference model, the step API and the binary platform tags;
  emitting both here would double the surface for the benefit of the newest tools only. FMI 2.0 is what Simulink and
  the Modelica tools load, and 3.0 can wait for a real request.
- **State or dynamics of its own.** A surrogate is an algebraic map from inputs to outputs; the slave recomputes it
  at every ``fmi2DoStep`` and holds no state beyond the last inputs. An FMU that integrated something would be
  claiming a model of the physics that this one does not have.
- **A step that refuses to extrapolate.** When the inputs leave the validated box the slave sets ``envelope_ok`` to
  false and logs a warning, and the step still returns ``fmi2OK``. FMI 2.0 has no "refuse this input" that does not
  end somebody's simulation, and a boundary case a tenth of a percent outside the box should not kill a run that is
  otherwise fine; the flag, the log line and the ``min``/``max`` on the variables are what a caller acts on.

The one gate worth naming: :func:`export_fmu` refuses a surrogate whose evidence does not read ``validated`` (or
``approved``). Shipping a model that failed, or that is still under review, into another engineer's Simulink is how a
surrogate gets used outside the envelope it was checked in, and the refusal is cheaper than that.

    >>> from pinneapple_simulation.external_solvers.modelica.fmu_export import (
    ...     Port, RbfForm, SurrogateExport, export_fmu)
    >>> form = RbfForm.fit(design, values)                       # the surrogate, in a form that compiles
    >>> export = SurrogateExport(model=form, inputs=(Port("u", "m/s", 0.0, 30.0),),
    ...                          outputs=(Port("drag", "N"),), evidence=assessment.to_dict())
    >>> print(export_fmu(export, "drag.fmu").summary())          # refuses unless evidence reads "validated"
"""
from __future__ import annotations

import hashlib
import json
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

__all__ = ["FMU_EXPORT_VERSION", "VALIDATED_DECISIONS", "Port", "Envelope", "RbfForm", "MlpForm",
           "SurrogateExport", "ExportedFMU", "UnvalidatedSurrogate", "export_fmu", "platform_binary_tag",
           "compiler_command"]

#: Version of this exporter, written into the model description and the manifest.
FMU_EXPORT_VERSION = "1.0"

#: Evidence decisions that let a surrogate leave the library. ``ValidationCase`` writes ``"validated"``; the trust
#: card writes ``"APPROVED"``, which is the same verdict in the other vocabulary.
VALIDATED_DECISIONS = ("validated", "approved")

_IDENTIFIER = "surrogate"

#: Value references: inputs are ``0 .. n_in-1`` (the column order of the tables), outputs start at 100, and the
#: envelope flag has one of its own so a reader never has to guess which Boolean is which.
_OUTPUT_REFERENCE = 100
_ENVELOPE_REFERENCE = 200


class UnvalidatedSurrogate(RuntimeError):
    """The evidence attached to a surrogate does not say it was validated, so it is not exported."""


# ---------------------------------------------------------------------------
# the interface: what the surrogate may be asked
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    """One input or output of the surrogate, with the range it is defined on.

    ``minimum``/``maximum`` are the validated range for an input (written as the FMI ``min``/``max`` attributes) and
    are left empty for an output, where they would be a claim about the result rather than a limit on the question."""

    name: str
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    start: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"port name {self.name!r} is not an identifier: FMI 2.0 structured naming expects "
                             "letters, digits and underscores, starting with a letter or an underscore")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"port {self.name!r}: minimum {self.minimum} is above maximum {self.maximum}")
        if self.minimum is not None and self.maximum is not None:
            if not (float(self.minimum) <= float(self.start) <= float(self.maximum)):
                raise ValueError(f"port {self.name!r}: start {self.start} is outside its validated range "
                                 f"[{self.minimum}, {self.maximum}]")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "minimum": self.minimum, "maximum": self.maximum,
                "start": float(self.start), "description": self.description}


@dataclass(frozen=True)
class Envelope:
    """The box the surrogate was validated in: one :class:`Port` per input and output."""

    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError("an envelope needs at least one input and one output")
        names = [p.name for p in (*self.inputs, *self.outputs)]
        if len(set(names)) != len(names):
            raise ValueError(f"port names repeat: {names}")
        for port in self.inputs:
            if port.minimum is None or port.maximum is None:
                raise ValueError(f"input {port.name!r} has no validated range: an FMI variable without min/max "
                                 "tells the tool on the other side nothing about where the surrogate is valid")

    def contains(self, design) -> np.ndarray:
        """A boolean per row of ``design``: every input inside its validated range."""
        design = np.atleast_2d(np.asarray(design, dtype=np.float64))
        if design.shape[1] != len(self.inputs):
            raise ValueError(f"{design.shape[1]} inputs against {len(self.inputs)} declared input(s)")
        inside = np.ones(design.shape[0], dtype=bool)
        for column, port in enumerate(self.inputs):
            inside &= (design[:, column] >= float(port.minimum)) & (design[:, column] <= float(port.maximum))
        return inside

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": [p.to_dict() for p in self.inputs], "outputs": [p.to_dict() for p in self.outputs]}


# ---------------------------------------------------------------------------
# the surrogate, in a form that compiles
# ---------------------------------------------------------------------------

class _Form:
    """What both lowerings have in common: shapes, the scaled evaluation, and the C body they emit."""

    input_scale: np.ndarray
    output_scale: np.ndarray

    @property
    def n_inputs(self) -> int:                      # pragma: no cover - overridden by the shape it reads
        raise NotImplementedError

    @property
    def n_outputs(self) -> int:                     # pragma: no cover
        raise NotImplementedError

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(x, dtype=np.float64)) / self.input_scale

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError                     # pragma: no cover

    def _c_preamble(self) -> str:
        raise NotImplementedError                     # pragma: no cover

    def _c_predict(self) -> str:
        raise NotImplementedError                     # pragma: no cover

    def digest(self) -> str:
        """SHA-256 over every number that defines the map -- the identity of the surrogate, quoted in the package."""
        h = hashlib.sha256()
        for array in self._tables():
            h.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return h.hexdigest()

    def _tables(self) -> list[np.ndarray]:
        raise NotImplementedError                     # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError                     # pragma: no cover


@dataclass(frozen=True)
class RbfForm(_Form):
    """A kernel expansion: ``y = output_scale * (bias + sum_j exp(-|x - c_j|^2 / 2l^2) w_j)``.

    This is the form a kernel-ridge surrogate, a Gaussian-process posterior mean, or the correction of a
    co-kriging model takes once its hyperparameters are fixed: a weighted sum of radial bumps, which is a handful of
    arrays and the same handful of lines in C. ``centers`` and ``length_scale`` are in the *scaled* input space (the
    inputs divided by ``input_scale``), which is how a design in metres and one in Mach number end up on the same
    footing -- the same convention :mod:`pinneapple_data.multifidelity` uses."""

    centers: np.ndarray                     # (m, d) scaled
    weights: np.ndarray                     # (m, k)
    length_scale: float
    bias: np.ndarray | None = None          # (k,) or None
    input_scale: np.ndarray | None = None   # (d,) or None
    output_scale: np.ndarray | None = None  # (k,) or None

    def __post_init__(self) -> None:
        centers = np.atleast_2d(np.asarray(self.centers, dtype=np.float64))
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights[:, None]
        if centers.ndim != 2 or weights.ndim != 2:
            raise ValueError(f"centers must be (m, d) and weights (m, k): {centers.shape}, {weights.shape}")
        if centers.shape[0] != weights.shape[0]:
            raise ValueError(f"{centers.shape[0]} centers against {weights.shape[0]} weight rows")
        if not np.isfinite(self.length_scale) or float(self.length_scale) <= 0.0:
            raise ValueError(f"length scale {self.length_scale} is not a positive number")
        k = weights.shape[1]
        bias = np.zeros(k) if self.bias is None else np.asarray(self.bias, dtype=np.float64).reshape(-1)
        if bias.size != k:
            raise ValueError(f"{bias.size} biases against {k} outputs")
        input_scale = np.ones(centers.shape[1]) if self.input_scale is None else np.asarray(
            self.input_scale, dtype=np.float64).reshape(-1)
        output_scale = np.ones(k) if self.output_scale is None else np.asarray(
            self.output_scale, dtype=np.float64).reshape(-1)
        if input_scale.size != centers.shape[1]:
            raise ValueError(f"{input_scale.size} input scales against {centers.shape[1]} columns")
        if output_scale.size != k:
            raise ValueError(f"{output_scale.size} output scales against {k} outputs")
        if np.any(input_scale == 0.0):
            raise ValueError("an input scale of zero cannot divide the inputs")
        for name, array in (("centers", centers), ("weights", weights), ("bias", bias),
                            ("input_scale", input_scale), ("output_scale", output_scale)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} has a non-finite entry: a table with NaN in it makes every prediction NaN")
        for name, array in (("centers", centers), ("weights", weights), ("bias", bias),
                            ("input_scale", input_scale), ("output_scale", output_scale)):
            object.__setattr__(self, name, array)

    @property
    def n_inputs(self) -> int:
        return int(self.centers.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.weights.shape[1])

    def evaluate(self, x) -> np.ndarray:
        x = self._scaled(x)
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"{x.shape[1]} inputs against a surrogate of {self.n_inputs}")
        squared = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=-1)
        phi = np.exp(-0.5 * squared / float(self.length_scale) ** 2)
        return (phi @ self.weights + self.bias) * self.output_scale

    @classmethod
    def fit(cls, design, values, *, ridge: float = 1e-8, length_scale: float | None = None,
            input_scale=None, output_scale=None) -> RbfForm:
        """Kernel ridge with an RBF kernel, in closed form -- the estimator of
        :class:`pinneapple_data.multifidelity.KernelRidge`, returned in the shape that compiles.

        The default length scale is the median pairwise distance of the training designs: no tuning, which matters
        when the surrogate was fitted on eight expensive runs."""
        design = np.atleast_2d(np.asarray(design, dtype=np.float64))
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if design.shape[0] != values.shape[0]:
            raise ValueError(f"{design.shape[0]} designs against {values.shape[0]} values")
        if design.shape[0] < 2:
            raise ValueError("one design is not enough to fit: the kernel has no distance to measure")
        scale = np.ones(design.shape[1]) if input_scale is None else np.asarray(
            input_scale, dtype=np.float64).reshape(-1)
        scaled = design / scale
        if length_scale is None:
            differences = scaled[:, None, :] - scaled[None, :, :]
            distances = np.sqrt((differences ** 2).sum(axis=-1))
            off_diagonal = distances[~np.eye(distances.shape[0], dtype=bool)]
            candidate = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
            length_scale = candidate if candidate > 0.0 else 1.0
        squared = ((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=-1)
        kernel = np.exp(-0.5 * squared / float(length_scale) ** 2)
        mean = values.mean(axis=0)
        weights = np.linalg.solve(kernel + float(ridge) * np.eye(scaled.shape[0]), values - mean)
        return cls(centers=scaled, weights=weights, length_scale=float(length_scale), bias=mean,
                   input_scale=scale,
                   output_scale=np.ones(values.shape[1]) if output_scale is None else output_scale)

    def _tables(self) -> list[np.ndarray]:
        return [self.centers, self.weights, self.bias, self.input_scale, self.output_scale]

    def _c_preamble(self) -> str:
        m, d = self.centers.shape
        return "\n".join([
            f"#define N_CENTERS {m}",
            f"#define LENGTH_SCALE {self.length_scale!r}",
            _c_array("IN_SCALE", self.input_scale),
            _c_array("OUT_SCALE", self.output_scale),
            _c_array("CENTERS", self.centers.reshape(-1)),
            _c_array("WEIGHT", self.weights.reshape(-1)),
            _c_array("BIAS", self.bias),
            "",
            "/* y[k] = OUT_SCALE[k] * (BIAS[k] + sum_j phi_j * WEIGHT[j][k]) */",
            "static void pinn_surrogate_predict(const double *x, double *y) {",
            "    double xs[N_IN];",
            "    int i, j, k;",
            "    for (i = 0; i < N_IN; ++i) xs[i] = x[i] / IN_SCALE[i];",
            "    for (k = 0; k < N_OUT; ++k) y[k] = BIAS[k];",
            "    for (j = 0; j < N_CENTERS; ++j) {",
            "        double r2 = 0.0, phi;",
            "        for (i = 0; i < N_IN; ++i) { double d = xs[i] - CENTERS[j * N_IN + i]; r2 += d * d; }",
            "        phi = exp(-0.5 * r2 / (LENGTH_SCALE * LENGTH_SCALE));",
            "        for (k = 0; k < N_OUT; ++k) y[k] += phi * WEIGHT[j * N_OUT + k];",
            "    }",
            "    for (k = 0; k < N_OUT; ++k) y[k] *= OUT_SCALE[k];",
            "}",
            "",
        ])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "rbf", "length_scale": float(self.length_scale),
                "centers": self.centers.tolist(), "weights": self.weights.tolist(),
                "bias": self.bias.tolist(), "input_scale": self.input_scale.tolist(),
                "output_scale": self.output_scale.tolist(), "sha256": self.digest()}


@dataclass(frozen=True)
class MlpForm(_Form):
    """A fully connected network with ``tanh`` on every hidden layer and a linear output layer.

    The parameters are plain arrays, so the weights of a small trained network -- the kind that fits on a handful of
    simulation runs -- become a few dozen lines of C instead of a dependency on a deep-learning runtime inside
    somebody else's simulator."""

    weights: tuple[np.ndarray, ...]         # layer i: (fan_in, fan_out)
    biases: tuple[np.ndarray, ...]          # layer i: (fan_out,)
    input_scale: np.ndarray | None = None
    output_scale: np.ndarray | None = None

    def __post_init__(self) -> None:
        weights = tuple(np.asarray(w, dtype=np.float64) for w in self.weights)
        biases = tuple(np.asarray(b, dtype=np.float64).reshape(-1) for b in self.biases)
        if not weights or len(weights) != len(biases):
            raise ValueError(f"{len(weights)} weight matrices against {len(biases)} bias vectors")
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            if w.ndim != 2:
                raise ValueError(f"layer {i} is {w.shape}, not (fan_in, fan_out)")
            if w.shape[1] != b.size:
                raise ValueError(f"layer {i}: {w.shape[1]} columns against {b.size} biases")
            if i and weights[i - 1].shape[1] != w.shape[0]:
                raise ValueError(f"layer {i} takes {w.shape[0]} inputs, layer {i - 1} gives "
                                 f"{weights[i - 1].shape[1]}")
            for name, array in (("weights", w), ("biases", b)):
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"layer {i} {name} has a non-finite entry")
        input_scale = np.ones(weights[0].shape[0]) if self.input_scale is None else np.asarray(
            self.input_scale, dtype=np.float64).reshape(-1)
        output_scale = np.ones(weights[-1].shape[1]) if self.output_scale is None else np.asarray(
            self.output_scale, dtype=np.float64).reshape(-1)
        if input_scale.size != weights[0].shape[0]:
            raise ValueError(f"{input_scale.size} input scales against {weights[0].shape[0]} inputs")
        if output_scale.size != weights[-1].shape[1]:
            raise ValueError(f"{output_scale.size} output scales against {weights[-1].shape[1]} outputs")
        if np.any(input_scale == 0.0):
            raise ValueError("an input scale of zero cannot divide the inputs")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "input_scale", input_scale)
        object.__setattr__(self, "output_scale", output_scale)

    @property
    def n_inputs(self) -> int:
        return int(self.weights[0].shape[0])

    @property
    def n_outputs(self) -> int:
        return int(self.weights[-1].shape[1])

    def evaluate(self, x) -> np.ndarray:
        a = self._scaled(x)
        if a.shape[1] != self.n_inputs:
            raise ValueError(f"{a.shape[1]} inputs against a surrogate of {self.n_inputs}")
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            a = a @ w + b
            if i < len(self.weights) - 1:
                a = np.tanh(a)
        return a * self.output_scale

    def _tables(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases, self.input_scale, self.output_scale]

    def _c_preamble(self) -> str:
        width = max([self.n_inputs, self.n_outputs, *(w.shape[1] for w in self.weights)])
        lines = [f"#define N_LAYERS {len(self.weights)}", f"#define WIDTH {width}",
                 _c_array("IN_SCALE", self.input_scale), _c_array("OUT_SCALE", self.output_scale)]
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            lines.append(_c_array(f"W{i}", w.reshape(-1)))
            lines.append(_c_array(f"B{i}", b))
        lines += ["", "/* tanh on every hidden layer, linear on the last one */",
                  "static void pinn_surrogate_predict(const double *x, double *y) {",
                  "    double a[WIDTH], b[WIDTH];",
                  "    double *cur = a, *nxt = b, *swap;",
                  "    int i, j, fan_in, fan_out;",
                  "    for (i = 0; i < N_IN; ++i) a[i] = x[i] / IN_SCALE[i];",
                  "    fan_in = N_IN;"]
        for i, w in enumerate(self.weights):
            fan_out = w.shape[1]
            activation = "tanh(s)" if i < len(self.weights) - 1 else "s"
            lines += [
                f"    for (j = 0; j < {fan_out}; ++j) {{",
                f"        double s = B{i}[j];",
                f"        for (i = 0; i < fan_in; ++i) s += W{i}[i * {fan_out} + j] * cur[i];",
                f"        nxt[j] = {activation};",
                "    }",
                "    swap = cur; cur = nxt; nxt = swap;",
                f"    fan_in = {fan_out};",
            ]
        lines += ["    for (j = 0; j < N_OUT; ++j) y[j] = cur[j] * OUT_SCALE[j];",
                  "}", ""]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "mlp", "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "input_scale": self.input_scale.tolist(), "output_scale": self.output_scale.tolist(),
                "sha256": self.digest()}


def _c_array(name: str, values, per_line: int = 4) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    rows = [", ".join(_c_number(v) for v in flat[i:i + per_line]) for i in range(0, flat.size, per_line)]
    body = (",\n    ".join(rows))
    return f"static const double {name}[{flat.size}] = {{\n    {body}\n}};"


def _c_number(value: float) -> str:
    """A C literal for the exact double: ``repr`` of a float round-trips, and the suffix keeps it a double."""
    text = repr(float(value))
    if text in ("inf", "-inf", "nan"):               # refused earlier; the guard keeps the C valid
        raise ValueError(f"{value} has no C literal")
    return text if ("." in text or "e" in text or "E" in text) else text + ".0"


# ---------------------------------------------------------------------------
# the evidence
# ---------------------------------------------------------------------------

def _decision_of(evidence) -> str:
    """The verdict in ``evidence``: a mapping with ``decision``/``status``, or an object that carries the attribute.

    Both vocabularies the library already writes are read -- ``ValidationCase`` says ``"validated"``, the trust card
    says ``"APPROVED"`` -- and anything else is returned verbatim so the refusal can quote it."""
    if evidence is None:
        return ""
    if isinstance(evidence, Mapping):
        for key in ("decision", "status", "verdict"):
            if key in evidence:
                return str(evidence[key]).strip()
        return ""
    for attribute in ("decision", "status", "verdict"):
        value = getattr(evidence, attribute, None)
        if value is not None:
            return str(value).strip()
    return ""


def _require_validated(evidence) -> dict[str, Any]:
    """Normalise the evidence, and refuse a surrogate whose verdict is not a validation."""
    decision = _decision_of(evidence)
    if decision.lower() not in VALIDATED_DECISIONS:
        raise UnvalidatedSurrogate(
            "the evidence attached to this surrogate reads "
            f"{decision or 'nothing'!r}, not one of {VALIDATED_DECISIONS}: an FMU leaves the library only after its "
            "validation says so, because the file that arrives in another simulator carries no verdict with it "
            "beyond the one written here")
    if isinstance(evidence, Mapping):
        body = dict(evidence)
    elif hasattr(evidence, "to_dict"):
        body = dict(evidence.to_dict())
    else:
        body = {name: getattr(evidence, name) for name in ("decision", "status", "reason")
                if getattr(evidence, name, None) is not None}
    body["decision"] = decision
    return body


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateExport:
    """A validated surrogate, its interface, and the evidence that lets it out.

    ``model`` is one of the compiling forms (:class:`RbfForm`, :class:`MlpForm`); ``inputs``/``outputs`` are the
    ports, in the column order the model's tables use -- the exporter checks the counts, because a port list that is
    out of order would name the numbers wrongly on the other side and nothing inside the FMU could tell."""

    model: RbfForm | MlpForm
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    evidence: Any = field(default=None)
    name: str = "Surrogate"
    description: str = ""
    start_time: float = 0.0
    stop_time: float = 1.0
    step_size: float | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not hasattr(self.model, "evaluate"):
            raise TypeError(f"{type(self.model).__name__} is not a lowering of a surrogate: build an RbfForm or an "
                            "MlpForm from it first")
        if len(inputs) != self.model.n_inputs:
            raise ValueError(f"{len(inputs)} input port(s) against a surrogate of {self.model.n_inputs} input(s)")
        if len(outputs) != self.model.n_outputs:
            raise ValueError(f"{len(outputs)} output port(s) against a surrogate of {self.model.n_outputs} output(s)")
        if not self.name.isidentifier():
            raise ValueError(f"model name {self.name!r} is not an identifier")
        if self.step_size is not None and self.step_size <= 0.0:
            raise ValueError(f"step size {self.step_size} is not positive")
        if self.stop_time <= self.start_time:
            raise ValueError(f"stop time {self.stop_time} is not after start time {self.start_time}")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "evidence", _require_validated(self.evidence))

    @property
    def envelope(self) -> Envelope:
        return Envelope(inputs=self.inputs, outputs=self.outputs)

    def guid(self) -> str:
        """A GUID that changes with the coefficients, the interface or the envelope: an FMU's identity."""
        body = {"name": self.name, "model": self.model.to_dict(), "ports": self.envelope.to_dict(),
                "version": FMU_EXPORT_VERSION}
        return "{" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32] + "}"

    def predict(self, x) -> np.ndarray:
        """The surrogate's own numbers, the ones the compiled slave has to reproduce."""
        prediction = np.atleast_2d(self.model.evaluate(x))
        outside = ~self.envelope.contains(x)
        return prediction, outside


@dataclass
class ExportedFMU:
    """What :func:`export_fmu` wrote, and what it could not."""

    path: str
    model_name: str
    guid: str
    binary: str | None = None                    # path inside the archive
    built: bool = False
    platform: str = ""
    problems: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [f"{self.path}: FMU for {self.model_name} (FMI 2.0 Co-Simulation), guid {self.guid}"]
        lines.append(f"  binary: {self.binary}" if self.built else
                     f"  binary: none for {self.platform} -- {self.problems[0] if self.problems else 'not built'}")
        lines.append(f"  envelope: {len(self.manifest.get('envelope', {}).get('inputs', []))} input(s), "
                     f"{len(self.manifest.get('envelope', {}).get('outputs', []))} output(s), "
                     f"decision {self.manifest.get('evidence', {}).get('decision')!r}")
        lines += [f"  problem: {p}" for p in self.problems[1:] if not self.built]
        return "\n".join(lines)


def platform_binary_tag() -> tuple[str, str]:
    """The FMI 2.0 binary folder and extension for this machine: ``("darwin64", ".dylib")`` on macOS.

    FMI 2.0 names one folder per target and has no Apple-Silicon name of its own, so an arm64 Mac gets ``darwin64``
    -- which is what fmpy and the Modelica tools look in -- and the architecture is recorded in the annotations.
    """
    machine = _platform.machine().lower()
    if sys.platform == "darwin":
        return "darwin64", ".dylib"
    if sys.platform.startswith("linux"):
        return ("linux32" if machine in ("i386", "i686") else "linux64"), ".so"
    if sys.platform in ("win32", "cygwin"):
        return ("win32" if _platform.architecture()[0] == "32bit" else "win64"), ".dll"
    raise ValueError(f"{sys.platform} is not an FMI 2.0 platform name")


def compiler_command(source: Path, binary: Path, compiler: str | None = None) -> list[str] | None:
    """The command that turns ``source`` into a shared library, or ``None`` when this platform has no recipe here."""
    cc = compiler or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return None
    if sys.platform in ("win32", "cygwin"):
        return None                                  # an MSVC or mingw recipe is a different command line; not guessed
    if sys.platform == "darwin":
        return [cc, "-dynamiclib", "-O2", "-fPIC", "-std=c99", "-o", str(binary), str(source), "-lm"]
    return [cc, "-shared", "-O2", "-fPIC", "-std=c99", "-o", str(binary), str(source), "-lm"]


def export_fmu(export: SurrogateExport, path: str | Path, *, build: bool = True,
               compiler: str | None = None) -> ExportedFMU:
    """Write the FMU package for ``export``.

    ``build=False`` writes the description, the C source and the manifest without compiling, which is what a machine
    without a C compiler can do, and what a reader who only wants the metadata needs."""
    path = Path(path)
    if path.suffix.lower() != ".fmu":
        raise ValueError(f"{path.name}: an FMU archive is a .fmu file")
    tag, extension = platform_binary_tag()
    identifier = _IDENTIFIER
    binary_inside = f"binaries/{tag}/{identifier}{extension}"
    description = _model_description(export, identifier)
    source = _c_source(export, identifier)
    manifest = _manifest(export, tag, extension)
    result = ExportedFMU(path=str(path), model_name=export.name, guid=export.guid(), platform=tag,
                         manifest=manifest)

    binary_bytes: bytes | None = None
    if build:
        with tempfile.TemporaryDirectory(prefix="pinneapple-fmu-") as folder:
            source_path = Path(folder) / f"{identifier}.c"
            source_path.write_text(source)
            binary_path = Path(folder) / f"{identifier}{extension}"
            command = compiler_command(source_path, binary_path, compiler)
            if command is None:
                result.problems.append(f"no C compiler on {sys.platform} for a recipe this module knows: the package "
                                       "carries sources/ and resources/, build them on the target and add the binary "
                                       "under " + binary_inside)
            else:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
                if completed.returncode != 0 or not binary_path.is_file():
                    result.problems.append(f"the compiler refused the slave ({' '.join(command)}): "
                                           f"{completed.stderr.strip()[:400]}")
                else:
                    binary_bytes = binary_path.read_bytes()
                    result.built = True
                    result.binary = binary_inside
    else:
        result.problems.append("build=False: the archive carries sources/ and resources/ but no binary")

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("modelDescription.xml", description)
        if binary_bytes is not None:
            archive.writestr(binary_inside, binary_bytes)
        archive.writestr("sources/surrogate_fmu.c", source)
        archive.writestr("resources/validated_surrogate.json", json.dumps(manifest, indent=2, sort_keys=True))
    return result


def _manifest(export: SurrogateExport, tag: str, extension: str) -> dict[str, Any]:
    return {
        "schema": "pinneapple.surrogate_fmu/1",
        "exporter": {"version": FMU_EXPORT_VERSION, "platform": tag, "architecture": _platform.machine().lower()},
        "fmi": {"version": "2.0", "interface": "CoSimulation", "modelIdentifier": _IDENTIFIER},
        "model": {"name": export.name, "guid": export.guid(), "description": export.description,
                  "form": export.model.to_dict()},
        "envelope": export.envelope.to_dict(),
        "evidence": export.evidence,
        "produces": {"sources": "sources/surrogate_fmu.c",
                     "binary": f"binaries/{tag}/{_IDENTIFIER}{extension}"},
    }


def _model_description(export: SurrogateExport, identifier: str) -> str:
    """The FMI 2.0 model description, with the envelope in the variables and in the vendor annotations."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<fmiModelDescription fmiVersion="2.0" modelName="{escape(export.name)}" guid="{export.guid()}" '
        f'generationTool="PINNeAPPLE {FMU_EXPORT_VERSION}" variableNamingConvention="structured" '
        'numberOfEventIndicators="0">',
        f'  <CoSimulation modelIdentifier="{identifier}" canHandleVariableCommunicationStepSize="true" '
        'canInterpolateInputs="false" maxOutputDerivativeOrder="0" canRunAsynchronuously="false"/>',
        f'  <DefaultExperiment startTime="{export.start_time!r}" stopTime="{export.stop_time!r}"'
        + (f' stepSize="{export.step_size!r}"' if export.step_size is not None else "")
        + (f' tolerance="{export.tolerance!r}"' if export.tolerance is not None else "")
        + "/>",
    ]
    units = sorted({p.unit for p in (*export.inputs, *export.outputs) if p.unit})
    if units:
        lines.append("  <UnitDefinitions>")
        lines.extend(f'    <Unit name="{escape(unit)}"/>' for unit in units)
        lines.append("  </UnitDefinitions>")
    lines.append("  <ModelVariables>")
    for column, port in enumerate(export.inputs):
        lines.append("    " + _scalar_variable(port, column, causality="input", continuous=True))
    for column, port in enumerate(export.outputs):
        lines.append("    " + _scalar_variable(port, _OUTPUT_REFERENCE + column, causality="output",
                                               continuous=True))
    lines.append("    " + _scalar_variable(Port("envelope_ok", description="1 when the inputs are inside the "
                                                "validated envelope, 0 when they are not"), _ENVELOPE_REFERENCE,
                                           causality="output", continuous=False))
    lines.append("  </ModelVariables>")
    lines.append("  <ModelStructure>")
    lines.append("    <Outputs>")
    for position in range(len(export.inputs) + 1, len(export.inputs) + len(export.outputs) + 2):
        lines.append(f'      <Unknown index="{position}"/>')
    lines.append("    </Outputs>")
    lines.append("  </ModelStructure>")
    envelope = {
        "inputs": [{"name": p.name, "minimum": p.minimum, "maximum": p.maximum, "unit": p.unit}
                   for p in export.inputs],
        "outputs": [{"name": p.name, "unit": p.unit} for p in export.outputs],
        "decision": export.evidence.get("decision"),
        "surrogate_sha256": export.model.digest(),
        "validated_surrogate": "resources/validated_surrogate.json",
    }
    lines += [
        "  <VendorAnnotations>",
        '    <Tool name="org.pinneapple.surrogate">',
        f'      <ValidatedEnvelope>{escape(json.dumps(envelope, sort_keys=True))}</ValidatedEnvelope>',
        "    </Tool>",
        "  </VendorAnnotations>",
        "</fmiModelDescription>",
    ]
    return "\n".join(lines) + "\n"


def _scalar_variable(port: Port, reference: int, *, causality: str, continuous: bool) -> str:
    """One ``ScalarVariable``: the Real of a port with its validated range, or the Boolean envelope flag."""
    attributes = [f'name="{port.name}"', f'valueReference="{reference}"', f'causality="{causality}"',
                  f'variability="{"continuous" if continuous else "discrete"}"']
    if port.description:
        attributes.append(f'description="{escape(port.description)}"')
    if not continuous:
        return f'<ScalarVariable {" ".join(attributes)}><Boolean start="true"/></ScalarVariable>'
    body = [f'start="{float(port.start)!r}"']
    if port.unit:
        body.append(f'unit="{escape(port.unit)}"')
    if causality == "input":
        body.append(f'min="{float(port.minimum)!r}"')
        body.append(f'max="{float(port.maximum)!r}"')
    return f'<ScalarVariable {" ".join(attributes)}><Real {" ".join(body)}/></ScalarVariable>'


_FMI_PREAMBLE = r"""/* FMI 2.0 Co-Simulation slave generated by PINNeAPPLE %(version)s from a validated surrogate.
 *
 * The surrogate's coefficients are the static tables below: this file needs a C compiler and libm, nothing else
 * (the FMI 2.0 ABI types are declared here instead of in a header, so `cc -shared` is the whole build).
 * Inputs are the first N_IN value references, outputs the next N_OUT starting at %(out_ref)d, and the envelope
 * flag the Boolean %(envelope_ref)d.
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define PINN_EXPORT __declspec(dllexport)
#else
#define PINN_EXPORT __attribute__((visibility("default")))
#endif

typedef void *fmi2Component;
typedef void *fmi2ComponentEnvironment;
typedef void *fmi2FMUstate;
typedef unsigned int fmi2ValueReference;
typedef double fmi2Real;
typedef int fmi2Integer;
typedef int fmi2Boolean;
typedef char fmi2Byte;
typedef const char *fmi2String;
typedef enum { fmi2OK = 0, fmi2Warning = 1, fmi2Discard = 2, fmi2Error = 3, fmi2Fatal = 4, fmi2Pending = 5 }
    fmi2Status;
typedef enum { fmi2ModelExchange = 0, fmi2CoSimulation = 1 } fmi2Type;
typedef enum { fmi2DoStepStatus = 0, fmi2PendingStatus = 1, fmi2LastSuccessfulTime = 2, fmi2Terminated = 3 }
    fmi2StatusKind;
typedef void (*fmi2CallbackLogger)(fmi2ComponentEnvironment, fmi2String, fmi2Status, fmi2String, fmi2String, ...);
typedef void *(*fmi2CallbackAllocateMemory)(size_t, size_t);
typedef void (*fmi2CallbackFreeMemory)(void *);
typedef void (*fmi2StepFinished)(fmi2ComponentEnvironment, fmi2Status);
typedef struct {
    fmi2CallbackLogger logger;
    fmi2CallbackAllocateMemory allocateMemory;
    fmi2CallbackFreeMemory freeMemory;
    fmi2StepFinished stepFinished;
    fmi2ComponentEnvironment componentEnvironment;
} fmi2CallbackFunctions;

#define N_IN %(n_in)d
#define N_OUT %(n_out)d
#define VR_OUT %(out_ref)d
#define VR_ENVELOPE %(envelope_ref)d
#define GUID "%(guid)s"
#define MODEL_NAME "%(name)s"
"""

_FMI_BODY = r"""
typedef struct {
    fmi2Real inputs[N_IN];
    fmi2Real outputs[N_OUT];
    fmi2Real time;
    fmi2Boolean envelope_ok;
    fmi2Boolean initialized;
    fmi2Boolean terminated;
} PinnState;

typedef struct {
    PinnState state;
    fmi2CallbackFunctions callbacks;
    fmi2Boolean logging;
} PinnComponent;

static void pinn_log(PinnComponent *c, fmi2Status status, const char *category, const char *message) {
    if (c != NULL && c->logging && c->callbacks.logger != NULL) {
        c->callbacks.logger(c->callbacks.componentEnvironment, MODEL_NAME, status, category, message);
    }
}

static fmi2Boolean pinn_in_envelope(const fmi2Real *x) {
    int i;
    for (i = 0; i < N_IN; ++i) {
        if (!(x[i] >= IN_MIN[i] && x[i] <= IN_MAX[i])) return 0;
    }
    return 1;
}

static void pinn_update(PinnComponent *c) {
    c->state.envelope_ok = pinn_in_envelope(c->state.inputs);
    pinn_surrogate_predict(c->state.inputs, c->state.outputs);
}

static void pinn_reset(PinnComponent *c) {
    int i, k;
    for (i = 0; i < N_IN; ++i) c->state.inputs[i] = IN_START[i];
    for (k = 0; k < N_OUT; ++k) c->state.outputs[k] = 0.0;
    c->state.time = 0.0;
    c->state.envelope_ok = 1;
    c->state.initialized = 0;
    c->state.terminated = 0;
}

PINN_EXPORT const char *fmi2GetTypesPlatform(void) { return "default"; }
PINN_EXPORT const char *fmi2GetVersion(void) { return "2.0"; }

PINN_EXPORT fmi2Status fmi2SetDebugLogging(fmi2Component component, fmi2Boolean loggingOn, size_t nCategories,
                                           const fmi2String categories[]) {
    PinnComponent *c = (PinnComponent *)component;
    (void)nCategories;
    (void)categories;
    if (c == NULL) return fmi2Error;
    c->logging = loggingOn;
    return fmi2OK;
}

PINN_EXPORT fmi2Component fmi2Instantiate(fmi2String instanceName, fmi2Type fmuType, fmi2String fmuGUID,
                                          fmi2String fmuResourceLocation, const fmi2CallbackFunctions *functions,
                                          fmi2Boolean visible, fmi2Boolean loggingOn) {
    PinnComponent *c;
    (void)instanceName;
    (void)fmuResourceLocation;
    (void)visible;
    if (fmuGUID == NULL || strcmp(fmuGUID, GUID) != 0) {
        if (functions != NULL && functions->logger != NULL) {
            functions->logger(functions->componentEnvironment, MODEL_NAME, fmi2Error, "instantiation",
                              "wrong GUID: the model description and the binary are not the same FMU");
        }
        return NULL;
    }
    if (fmuType != fmi2CoSimulation) {
        if (functions != NULL && functions->logger != NULL) {
            functions->logger(functions->componentEnvironment, MODEL_NAME, fmi2Error, "instantiation",
                              "this FMU is Co-Simulation only");
        }
        return NULL;
    }
    c = (PinnComponent *)calloc(1, sizeof(PinnComponent));
    if (c == NULL) return NULL;
    if (functions != NULL) c->callbacks = *functions;
    c->logging = loggingOn;
    pinn_reset(c);
    return (fmi2Component)c;
}

PINN_EXPORT void fmi2FreeInstance(fmi2Component component) { free(component); }

PINN_EXPORT fmi2Status fmi2SetupExperiment(fmi2Component component, fmi2Boolean toleranceDefined,
                                           fmi2Real tolerance, fmi2Real startTime, fmi2Boolean stopTimeDefined,
                                           fmi2Real stopTime) {
    PinnComponent *c = (PinnComponent *)component;
    (void)toleranceDefined;
    (void)tolerance;
    (void)stopTimeDefined;
    (void)stopTime;
    if (c == NULL) return fmi2Error;
    c->state.time = startTime;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2EnterInitializationMode(fmi2Component component) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL) return fmi2Error;
    c->state.initialized = 0;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2ExitInitializationMode(fmi2Component component) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL) return fmi2Error;
    c->state.initialized = 1;
    pinn_update(c);
    if (!c->state.envelope_ok) {
        pinn_log(c, fmi2Warning, "envelope",
                 "the inputs leave the validated envelope: the outputs are an extrapolation");
    }
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2Terminate(fmi2Component component) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL) return fmi2Error;
    c->state.terminated = 1;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2Reset(fmi2Component component) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL) return fmi2Error;
    pinn_reset(c);
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2GetReal(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                   fmi2Real value[]) {
    PinnComponent *c = (PinnComponent *)component;
    size_t i;
    if (c == NULL) return fmi2Error;
    for (i = 0; i < nvr; ++i) {
        if (vr[i] >= VR_OUT && vr[i] < VR_OUT + N_OUT) {
            value[i] = c->state.outputs[vr[i] - VR_OUT];
        } else if (vr[i] < N_IN) {
            value[i] = c->state.inputs[vr[i]];
        } else {
            pinn_log(c, fmi2Error, "get", "unknown Real value reference");
            return fmi2Error;
        }
    }
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2SetReal(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                   const fmi2Real value[]) {
    PinnComponent *c = (PinnComponent *)component;
    size_t i;
    if (c == NULL) return fmi2Error;
    for (i = 0; i < nvr; ++i) {
        if (vr[i] < N_IN) {
            c->state.inputs[vr[i]] = value[i];
        } else {
            pinn_log(c, fmi2Error, "set", "only the inputs are settable in this surrogate");
            return fmi2Error;
        }
    }
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2GetInteger(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                      fmi2Integer value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)value;
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2SetInteger(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                      const fmi2Integer value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)value;
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2GetBoolean(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                      fmi2Boolean value[]) {
    PinnComponent *c = (PinnComponent *)component;
    size_t i;
    if (c == NULL) return fmi2Error;
    for (i = 0; i < nvr; ++i) {
        if (vr[i] != VR_ENVELOPE) {
            pinn_log(c, fmi2Error, "get", "unknown Boolean value reference");
            return fmi2Error;
        }
        value[i] = c->state.envelope_ok;
    }
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2SetBoolean(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                      const fmi2Boolean value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)value;
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2GetString(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                     fmi2String value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)value;
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2SetString(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                     const fmi2String value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)value;
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2GetFMUstate(fmi2Component component, fmi2FMUstate *FMUstate) {
    PinnComponent *c = (PinnComponent *)component;
    PinnState *copy;
    if (c == NULL || FMUstate == NULL) return fmi2Error;
    copy = (PinnState *)malloc(sizeof(PinnState));
    if (copy == NULL) return fmi2Error;
    *copy = c->state;
    *FMUstate = (fmi2FMUstate)copy;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2SetFMUstate(fmi2Component component, fmi2FMUstate FMUstate) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL || FMUstate == NULL) return fmi2Error;
    c->state = *(const PinnState *)FMUstate;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2FreeFMUstate(fmi2Component component, fmi2FMUstate *FMUstate) {
    (void)component;
    if (FMUstate == NULL) return fmi2Error;
    free(*FMUstate);
    *FMUstate = NULL;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2SerializedFMUstateSize(fmi2Component component, fmi2FMUstate FMUstate, size_t *size) {
    (void)component;
    if (FMUstate == NULL || size == NULL) return fmi2Error;
    *size = sizeof(PinnState);
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2SerializeFMUstate(fmi2Component component, fmi2FMUstate FMUstate, fmi2Byte serializedState[],
                                             size_t size) {
    (void)component;
    if (FMUstate == NULL || serializedState == NULL || size < sizeof(PinnState)) return fmi2Error;
    memcpy(serializedState, FMUstate, sizeof(PinnState));
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2DeSerializeFMUstate(fmi2Component component, const fmi2Byte serializedState[],
                                               size_t size, fmi2FMUstate *FMUstate) {
    PinnState *copy;
    (void)component;
    if (serializedState == NULL || size < sizeof(PinnState) || FMUstate == NULL) return fmi2Error;
    copy = (PinnState *)malloc(sizeof(PinnState));
    if (copy == NULL) return fmi2Error;
    memcpy(copy, serializedState, sizeof(PinnState));
    *FMUstate = (fmi2FMUstate)copy;
    return fmi2OK;
}

PINN_EXPORT fmi2Status fmi2GetDirectionalDerivative(fmi2Component component, const fmi2ValueReference vUnknown_ref[],
                                                    size_t nUnknown, const fmi2ValueReference vKnown_ref[],
                                                    size_t nKnown, const fmi2Real dvKnown[], fmi2Real dvUnknown[]) {
    PinnComponent *c = (PinnComponent *)component;
    (void)vUnknown_ref;
    (void)nUnknown;
    (void)vKnown_ref;
    (void)nKnown;
    (void)dvKnown;
    (void)dvUnknown;
    pinn_log(c, fmi2Warning, "derivative", "directional derivatives are not exported: this slave is algebraic");
    return fmi2Error;
}

PINN_EXPORT fmi2Status fmi2SetRealInputDerivatives(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                                   const fmi2Integer order[], const fmi2Real value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)order;
    (void)value;
    return fmi2Error;   /* canInterpolateInputs="false" in the model description: this is not called */
}

PINN_EXPORT fmi2Status fmi2GetRealOutputDerivatives(fmi2Component component, const fmi2ValueReference vr[], size_t nvr,
                                                    const fmi2Integer order[], fmi2Real value[]) {
    (void)component;
    (void)vr;
    (void)nvr;
    (void)order;
    (void)value;
    return fmi2Error;   /* maxOutputDerivativeOrder="0" */
}

PINN_EXPORT fmi2Status fmi2DoStep(fmi2Component component, fmi2Real currentCommunicationPoint,
                                  fmi2Real communicationStepSize, fmi2Boolean noSetFMUStatePriorToCurrentPoint) {
    PinnComponent *c = (PinnComponent *)component;
    (void)noSetFMUStatePriorToCurrentPoint;
    if (c == NULL) return fmi2Error;
    if (!c->state.initialized) {
        pinn_log(c, fmi2Error, "step", "fmi2DoStep before fmi2ExitInitializationMode");
        return fmi2Error;
    }
    if (c->state.terminated) {
        pinn_log(c, fmi2Error, "step", "fmi2DoStep after fmi2Terminate");
        return fmi2Error;
    }
    c->state.time = currentCommunicationPoint + communicationStepSize;
    pinn_update(c);
    if (!c->state.envelope_ok) {
        pinn_log(c, fmi2Warning, "envelope", "inputs outside the validated envelope: output is an extrapolation");
    }
    return fmi2OK;   /* the flag carries the verdict; a step that returned an error would end the caller's run */
}

PINN_EXPORT fmi2Status fmi2CancelStep(fmi2Component component) {
    (void)component;
    return fmi2Error;   /* an algebraic step is instantaneous: nothing to cancel */
}

PINN_EXPORT fmi2Status fmi2GetStatus(fmi2Component component, const fmi2StatusKind s, fmi2Status *value) {
    PinnComponent *c = (PinnComponent *)component;
    (void)c;
    if (value == NULL) return fmi2Error;
    if (s == fmi2DoStepStatus) {
        *value = fmi2OK;
        return fmi2OK;
    }
    return fmi2Discard;
}

PINN_EXPORT fmi2Status fmi2GetRealStatus(fmi2Component component, const fmi2StatusKind s, fmi2Real *value) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL || value == NULL) return fmi2Error;
    if (s == fmi2LastSuccessfulTime) {
        *value = c->state.time;
        return fmi2OK;
    }
    return fmi2Discard;
}

PINN_EXPORT fmi2Status fmi2GetIntegerStatus(fmi2Component component, const fmi2StatusKind s, fmi2Integer *value) {
    (void)component;
    (void)s;
    if (value == NULL) return fmi2Error;
    return fmi2Discard;
}

PINN_EXPORT fmi2Status fmi2GetBooleanStatus(fmi2Component component, const fmi2StatusKind s, fmi2Boolean *value) {
    PinnComponent *c = (PinnComponent *)component;
    if (c == NULL || value == NULL) return fmi2Error;
    if (s == fmi2Terminated) {
        *value = c->state.terminated;
        return fmi2OK;
    }
    return fmi2Discard;
}

PINN_EXPORT fmi2Status fmi2GetStringStatus(fmi2Component component, const fmi2StatusKind s, fmi2String *value) {
    (void)component;
    (void)s;
    if (value == NULL) return fmi2Error;
    return fmi2Discard;
}
"""


def _c_source(export: SurrogateExport, identifier: str) -> str:
    """The slave: the FMI 2.0 ABI with the surrogate's tables and its envelope compiled in."""
    header = _FMI_PREAMBLE % {"version": FMU_EXPORT_VERSION, "n_in": export.model.n_inputs,
                              "n_out": export.model.n_outputs, "out_ref": _OUTPUT_REFERENCE,
                              "envelope_ref": _ENVELOPE_REFERENCE, "guid": export.guid(),
                              "name": export.name}
    bounds = [_c_array("IN_MIN", [p.minimum for p in export.inputs]),
              _c_array("IN_MAX", [p.maximum for p in export.inputs]),
              _c_array("IN_START", [p.start for p in export.inputs])]
    return "\n".join([header, *bounds, export.model._c_preamble(), _FMI_BODY])
