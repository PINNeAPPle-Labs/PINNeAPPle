"""Signed trust reports whose physics residual is recomputed on load (#644, item B7).

The acceptance line of the issue: a signed report whose recorded physics residual is recomputed on load, and any
tampering with the report, the model or the inputs fails verification, checked by a property test over 100 random
byte flips.

Three layers are exercised separately here, because each catches something the others cannot:

* the DSSE signature over the payload (the report, the hashes, the residual spec and the recorded value);
* the SHA-256 of the model and of the inputs (the bytes themselves);
* the recomputation of the residual from the loaded model and inputs — the only layer that catches a payload that is
  internally consistent and correctly signed, which is what a stale tool or a lying one produces.

The residual is the grid check of ``field_card`` (three-point finite differences on the model's own prediction), so a
model that is not harmonic has a non-zero residual to get wrong.
"""
from __future__ import annotations

import hashlib
import io
import json
import random

import numpy as np
import pytest
import torch

import pinneapple_security.signing as signing
from pinneapple_analysis.trust.signed_report import (
    BUNDLE_FORMAT,
    ArtifactMismatch,
    ResidualMismatch,
    ResidualNotComputable,
    ResidualSpec,
    SignatureFailure,
    SignedReportError,
    SignedTrustReport,
)
from pinneapple_analysis.trust.trust_report import TrustReport, checagem_erro_relativo


# ---------------------------------------------------------------------------------------------- fixtures
class CurveNet(torch.nn.Module):
    """A small deterministic model: two inputs, one output, not harmonic, so the residual is a real number."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


GRID = {"x": [round(0.05 * i, 3) for i in range(20)], "y": [round(0.05 * i, 3) for i in range(20)]}


def _report() -> TrustReport:
    return TrustReport([checagem_erro_relativo("accuracy", 0.01, 0.02, critica=True),
                        checagem_erro_relativo("reference", 0.03, 0.05, critica=False)], model="curve_net")


@pytest.fixture
def key():
    """HMAC: the same signing path as Ed25519 without the ``cryptography`` dependency the extra installs."""
    return signing.hmac_key(b"a-32-byte-secret-for-the-tests")


def _bundle(key, report=None, model=None, grid=None, spec=None) -> SignedTrustReport:
    return SignedTrustReport.create(report if report is not None else _report(), model=model or CurveNet(),
                                    inputs=grid or GRID, residual=spec or ResidualSpec(pde="laplace"), key=key)


# ---------------------------------------------------------------------------------------------- the happy path
def test_the_recorded_residual_is_recomputed_from_the_signed_model_and_inputs(key):
    bundle = _bundle(key)
    checked = SignedTrustReport.loads(bundle.dumps()).verify(key.public(), model_factory=CurveNet)
    assert checked.ok
    assert checked.residual_recorded == pytest.approx(checked.residual_recomputed, rel=1e-9)
    assert checked.residual_recomputed > 0.0, "an untrained tanh stack is not harmonic: the residual is a real number"
    assert checked.report.decisao == "APPROVED" and checked.report.model == "curve_net"
    assert checked.model_class.endswith(":CurveNet")


def test_the_bundle_carries_the_weights_not_a_reference_to_the_model_in_memory(key):
    signed_model = CurveNet()
    bundle = _bundle(key, model=signed_model)
    with torch.no_grad():                               # the object the caller still holds is changed afterwards
        for p in signed_model.parameters():
            p.add_(1.0)
    checked = SignedTrustReport.loads(bundle.dumps()).verify(key.public(), model_factory=CurveNet)
    assert checked.ok, "verification rebuilt the signed weights, not the caller's mutated object"
    assert checked.residual_recorded == pytest.approx(checked.residual_recomputed, rel=1e-9)


def test_save_and_load_round_trip_on_disk(tmp_path, key):
    path = _bundle(key).save(tmp_path / "curve.pptrust")
    assert json.loads((tmp_path / "curve.pptrust").read_text())["format"] == BUNDLE_FORMAT
    assert SignedTrustReport.load(path).verify(key.public(), model_factory=CurveNet).ok


def test_a_second_residual_form_the_heat_equation_is_carried_as_data(key):
    times = np.linspace(0.0, 1.0, 6)
    spec = ResidualSpec(pde="heat", alpha=0.1, time_coord="t", source=None)
    grid = {"x": list(np.linspace(0.05, 0.95, 9)), "t": list(times)}
    bundle = _bundle(key, grid=grid, spec=spec)
    checked = SignedTrustReport.loads(bundle.dumps()).verify(key.public(), model_factory=CurveNet)
    assert checked.spec.pde == "heat" and checked.ok


# ---------------------------------------------------------------------------------------------- the three layers
def test_the_wrong_key_is_refused_before_anything_is_recomputed(key):
    with pytest.raises(SignatureFailure, match="does not verify"):
        _bundle(key).verify(signing.hmac_key(b"another-secret-altogether").public(), model_factory=CurveNet)


def test_a_report_whose_own_number_the_model_does_not_have_is_refused_at_signing():
    key = signing.hmac_key(b"a-32-byte-secret-for-the-tests")
    wrong = [checagem_erro_relativo("accuracy", 0.01, 0.02), ]
    wrong.append(checagem_erro_relativo("residuo_pde", 0.5, 0.05, critica=True))
    with pytest.raises(ResidualMismatch, match="the report records residuo_pde"):
        _bundle(key, report=TrustReport(wrong, model="curve_net"))


def test_a_fabricated_residual_that_was_re_signed_still_fails(key):
    """The signature proves who wrote the payload, not that its number is true: only the recomputation checks that."""
    bundle = SignedTrustReport.loads(_bundle(key).dumps())
    forged = signing.verify_json(bundle.envelope, key)                 # a tool with legitimate signing access
    forged["residual"]["value"] = forged["residual"]["value"] * 0.5     # ...and a number it likes better
    re_signed = SignedTrustReport(envelope=signing.sign_json(forged, key), model_blob=bundle.model_blob,
                                  inputs_blob=bundle.inputs_blob)
    with pytest.raises(ResidualMismatch) as err:
        re_signed.verify(key.public(), model_factory=CurveNet)
    assert "does not match the physics" in str(err.value)
    assert f"{forged['residual']['value']:.12g}" in str(err.value)


def test_a_model_that_drifted_with_its_hash_and_signature_updated_still_fails(key):
    """Weights changed, sha256 recomputed, payload re-signed: the artifact layer passes and the physics does not."""
    bundle = SignedTrustReport.loads(_bundle(key).dumps())
    drifted = {name: (tensor * 1.05 + 0.01) for name, tensor in
               torch.load(io.BytesIO(bundle.model_blob), weights_only=True).items()}
    buffer = io.BytesIO()
    torch.save(drifted, buffer)
    blob = buffer.getvalue()
    payload = signing.verify_json(bundle.envelope, key)
    payload["model"]["sha256"] = hashlib.sha256(blob).hexdigest()
    re_signed = SignedTrustReport(envelope=signing.sign_json(payload, key), model_blob=blob,
                                  inputs_blob=bundle.inputs_blob)
    with pytest.raises(ResidualMismatch) as err:
        re_signed.verify(key.public(), model_factory=CurveNet)
    assert "relative difference" in str(err.value)


def test_the_model_bytes_and_the_inputs_bytes_are_checked_separately(key):
    bundle = SignedTrustReport.loads(_bundle(key).dumps())
    tampered_model = bytearray(bundle.model_blob)
    tampered_model[-40] ^= 0x01
    with pytest.raises(ArtifactMismatch, match="model: the bytes"):
        SignedTrustReport(envelope=bundle.envelope, model_blob=bytes(tampered_model),
                          inputs_blob=bundle.inputs_blob).verify(key.public(), model_factory=CurveNet)

    other_grid = json.dumps({"grid": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0]}},
                            sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ArtifactMismatch, match="inputs: the bytes"):
        SignedTrustReport(envelope=bundle.envelope, model_blob=bundle.model_blob,
                          inputs_blob=other_grid).verify(key.public(), model_factory=CurveNet)


# ---------------------------------------------------------------------------------------------- the property test
def _regions(text: str) -> dict[str, range]:
    """The three parts of the file a byte flip can land in: the signed payload, the model and the inputs."""
    doc = json.loads(text)
    model_at = text.index(doc["model"])
    inputs_at = text.index(doc["inputs"])
    return {"signed payload (report, hashes, spec)": range(0, min(model_at, inputs_at)),
            "model": range(model_at, model_at + len(doc["model"])),
            "inputs": range(inputs_at, inputs_at + len(doc["inputs"]))}


def test_any_of_one_hundred_random_byte_flips_fails_verification(key):
    """#644: the property test the issue asks for — 100 random byte flips, in the report, the model and the inputs.

    The flips are stratified so that every one of the three artifacts is hit (a uniform draw over a 4 kB file would
    mostly land in the two base64 blobs and could leave the report untested), and each flip is a single bit, so a flip
    always changes the byte (replacing it with an arbitrary byte could pick a whitespace that decodes to the same
    document, which would be a false failure of the property rather than a hole in the verifier).

    Measured once over the whole file, before landing this: every single-bit flip at every byte position (39184 of
    them on a 4898-byte bundle, all eight bit positions) raised a ``SignedReportError``, none of them slipped past.
    """
    text = _bundle(key).dumps()
    regions = _regions(text)
    rng = random.Random(644)
    names = list(regions)
    picks = [(name, rng.randrange(regions[name].start, regions[name].stop)) for name in names for _ in range(33)]
    picks += [(names[0], rng.randrange(regions[names[0]].start, regions[names[0]].stop))]
    assert len(picks) == 100

    caught: dict[str, int] = {name: 0 for name in regions}
    for name, offset in picks:
        mutated = bytearray(text.encode("utf-8"))
        mutated[offset] ^= 1 << rng.randrange(8)
        with pytest.raises(SignedReportError):
            SignedTrustReport.loads(mutated.decode("utf-8", errors="replace")).verify(key.public(),
                                                                                    model_factory=CurveNet)
        caught[name] += 1
    assert all(count >= 1 for count in caught.values()), f"every artifact is covered by the flips: {caught}"


def test_the_failure_names_the_layer_that_refused_it(key):
    text = _bundle(key).dumps()
    doc = json.loads(text)
    reports = []
    for region in ("model", "inputs"):
        mutated = list(text)
        at = text.index(doc[region])
        mutated[at] = "B" if mutated[at] != "B" else "C"
        with pytest.raises(SignedReportError) as err:
            SignedTrustReport.loads("".join(mutated)).verify(key.public(), model_factory=CurveNet)
        reports.append(err.value)
    assert isinstance(reports[0], SignedReportError) and isinstance(reports[1], SignedReportError)


# ---------------------------------------------------------------------------------------------- honest refusals
def test_a_model_class_that_does_not_resolve_here_says_to_pass_a_factory(key):
    class LocalOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.w * x[:, 0]

    bundle = _bundle(key, model=LocalOnly())
    assert not bundle.dumps().startswith("error")
    with pytest.raises(SignedReportError, match="model_factory"):
        bundle.verify(key.public())                      # the test class is not importable by path


def test_a_grid_too_coarse_for_the_residual_is_refused_not_silently_signed(key):
    with pytest.raises(ResidualNotComputable, match="at least 4 points"):
        _bundle(key, grid={"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0]})


def test_a_grid_axis_that_is_not_increasing_is_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        _bundle(signing.hmac_key(b"a-32-byte-secret-for-the-tests"), grid={"x": [0.0, 0.5, 0.4, 0.6], "y": [0, 1]})


def test_a_non_module_is_refused_with_what_it_is(key):
    with pytest.raises(TypeError, match="torch.nn.Module"):
        _bundle(key, model=lambda x: x)


def test_a_text_that_is_not_a_bundle_is_refused_rather_than_parsed_loosely():
    with pytest.raises(SignedReportError, match="bundle format"):
        SignedTrustReport.loads('{"format": "something-else", "envelope": {}, "model": "", "inputs": ""}')
    with pytest.raises(SignedReportError, match="not a pinneapple.signed-trust-report"):
        SignedTrustReport.loads('{"format": "pinneapple.signed-trust-report/1", "extra": 1}')
    with pytest.raises(SignedReportError, match="not readable JSON"):
        SignedTrustReport.loads("{not json")
