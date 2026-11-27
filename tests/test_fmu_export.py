"""Exporting a validated surrogate as an FMU: the package, its metadata, and the compiled slave running through the
FMI 2.0 ABI.

The three things the issue asks for, each with the same shape as the rest of the suite: the package is opened and
read (nothing here trusts the writer), the metadata is checked against what the envelope says, and -- when this
machine has a C compiler -- the binary is loaded with ``ctypes`` and driven through the same call sequence fmpy,
Simulink or OpenModelica would use, comparing its numbers with the Python surrogate's. The FMI 2.0 standard's
required function set comes from :mod:`fmu_inspect`, which holds it for the import direction, so the two paths are
checked against one copy of the standard.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.modelica.fmu_export import (
    Envelope,
    MlpForm,
    Port,
    RbfForm,
    SurrogateExport,
    UnvalidatedSurrogate,
    compiler_command,
    export_fmu,
    platform_binary_tag,
)
from pinneapple_simulation.external_solvers.modelica.fmu_inspect import (
    REQUIRED_FUNCTIONS,
    inspect_fmu,
)

NEEDS_COMPILER = pytest.mark.skipif(compiler_command(Path("x.c"), Path("x.so")) is None,
                                    reason="no C compiler with a recipe this module knows")


# ---------------------------------------------------------------------------
# the surrogate under test: two inputs, two outputs, an envelope
# ---------------------------------------------------------------------------

def _form(seed: int = 0, n: int = 12) -> RbfForm:
    rng = np.random.default_rng(seed)
    design = rng.uniform([0.0, 0.0], [30.0, 1.0], size=(n, 2))
    values = np.column_stack([0.1 * design[:, 0] ** 2 + design[:, 1] + 1.0, 0.3 * design[:, 0] - design[:, 1]])
    return RbfForm.fit(design, values, input_scale=np.array([30.0, 1.0]))


def _export(evidence=None) -> SurrogateExport:
    return SurrogateExport(
        model=_form(),
        inputs=(Port("u", "m/s", 0.0, 30.0, start=10.0), Port("alpha", "rad", 0.0, 1.0)),
        outputs=(Port("cd"), Port("cl", description="lift coefficient")),
        evidence={"decision": "validated", "case": "VC-demo"} if evidence is None else evidence,
        name="DragSurrogate", description="drag and lift of a plate, measured range only",
        stop_time=10.0, step_size=0.1)


def _mlp_export() -> SurrogateExport:
    rng = np.random.default_rng(7)
    weights = (rng.normal(size=(2, 5)) * 0.5, rng.normal(size=(5, 3)) * 0.5, rng.normal(size=(3, 1)))
    biases = (rng.normal(size=5) * 0.1, rng.normal(size=3) * 0.1, np.array([2.0]))
    form = MlpForm(weights=weights, biases=biases, input_scale=np.array([30.0, 1.0]))
    return SurrogateExport(model=form, inputs=(Port("u", "m/s", 0.0, 30.0), Port("alpha", "rad", 0.0, 1.0)),
                           outputs=(Port("cd"),), evidence={"decision": "validated"}, name="MeshSurrogate")


# ---------------------------------------------------------------------------
# the package: description, sources, manifest
# ---------------------------------------------------------------------------

def test_the_description_carries_the_envelope_and_the_evidence(tmp_path):
    fmu = export_fmu(_export(), tmp_path / "drag.fmu", build=False).path
    with zipfile.ZipFile(fmu) as archive:
        root = ET.fromstring(archive.read("modelDescription.xml"))
        manifest = json.loads(archive.read("resources/validated_surrogate.json"))
        assert "sources/surrogate_fmu.c" in archive.namelist()

    assert root.get("fmiVersion") == "2.0"
    assert root.get("modelName") == "DragSurrogate"
    assert root.get("guid") == _export().guid()
    (cosim,) = root.findall("CoSimulation")
    assert cosim.get("modelIdentifier") == "surrogate"
    assert root.find("DefaultExperiment").get("stopTime") == "10.0"

    variables = {v.get("name"): v for v in root.find("ModelVariables")}
    assert set(variables) == {"u", "alpha", "cd", "cl", "envelope_ok"}
    u = variables["u"]
    assert u.get("causality") == "input" and u.get("variability") == "continuous"
    (real,) = u.findall("Real")
    assert (float(real.get("min")), float(real.get("max"))) == (0.0, 30.0)     # the validated range, in the FMI slot
    assert real.get("unit") == "m/s" and float(real.get("start")) == 10.0
    assert variables["cd"].get("causality") == "output"
    assert variables["envelope_ok"].get("causality") == "output" and variables["envelope_ok"].find("Boolean") is not None
    assert [unit.get("name") for unit in root.findall("UnitDefinitions/Unit")] == ["m/s", "rad"]

    # ModelStructure refers to the 1-based index of the variables; the outputs are the last three
    outputs = [int(unknown.get("index")) for unknown in root.findall("ModelStructure/Outputs/Unknown")]
    ordered = list(root.find("ModelVariables"))
    assert outputs == [3, 4, 5] and [ordered[i - 1].get("name") for i in outputs] == ["cd", "cl", "envelope_ok"]

    # the envelope and the verdict a parser that ignores VendorAnnotations cannot see
    annotations = json.loads(root.find("VendorAnnotations/Tool/ValidatedEnvelope").text)
    assert annotations["decision"] == "validated"
    assert annotations["inputs"] == [{"name": "u", "minimum": 0.0, "maximum": 30.0, "unit": "m/s"},
                                     {"name": "alpha", "minimum": 0.0, "maximum": 1.0, "unit": "rad"}]
    assert annotations["surrogate_sha256"] == _form().digest() == manifest["model"]["form"]["sha256"]
    assert manifest["evidence"]["case"] == "VC-demo"
    assert manifest["fmi"] == {"version": "2.0", "interface": "CoSimulation", "modelIdentifier": "surrogate"}


def test_the_package_passes_the_librarys_own_preflight(tmp_path):
    """The importer's static checks are the first reviewer of what the exporter writes."""
    fmu = export_fmu(_export(), tmp_path / "drag.fmu", build=False).path
    report = inspect_fmu(fmu)
    assert report.problems == [], report.summary()
    assert report.interfaces == {"CS": "surrogate"}
    assert report.n_variables == 5


def test_build_false_ships_the_sources_and_says_why_there_is_no_binary(tmp_path):
    result = export_fmu(_export(), tmp_path / "drag.fmu", build=False)
    assert not result.built and result.binary is None and not result.ok
    assert "build=False" in result.problems[0]
    assert "binary: none" in result.summary()
    with zipfile.ZipFile(result.path) as archive:
        assert not [name for name in archive.namelist() if name.startswith("binaries/")]
        source = archive.read("sources/surrogate_fmu.c").decode()
    assert "fmi2DoStep" in source and "fmi2Instantiate" in source
    assert "IN_MIN" in source and "IN_MAX" in source        # the envelope is compiled in, not only written down


def test_the_guid_follows_the_coefficients_and_the_interface(tmp_path):
    first = _export().guid()
    assert first == _export().guid()                                        # deterministic
    rng = np.random.default_rng(99)
    other = _form(seed=3)
    changed = SurrogateExport(model=other, inputs=(Port("u", "m/s", 0.0, 30.0), Port("alpha", "rad", 0.0, 1.0)),
                              outputs=(Port("cd"), Port("cl")), evidence={"decision": "validated"},
                              name="DragSurrogate")
    assert changed.guid() != first
    narrower = SurrogateExport(model=_form(), inputs=(Port("u", "m/s", 0.0, 20.0), Port("alpha", "rad", 0.0, 1.0)),
                               outputs=(Port("cd"), Port("cl")), evidence={"decision": "validated"},
                               name="DragSurrogate")
    assert narrower.guid() != first                                          # a different envelope is a different FMU
    assert rng is not None


# ---------------------------------------------------------------------------
# the gate: only a validated surrogate leaves
# ---------------------------------------------------------------------------

def test_a_surrogate_whose_evidence_is_not_a_validation_is_refused():
    for evidence in ({"decision": "review required", "reason": "accuracy 8 % over the tolerance"},
                     {"status": "failed"}, {}, {"outcome": "validated"}):
        with pytest.raises(UnvalidatedSurrogate, match="not one of"):
            _export(evidence)


def test_no_evidence_at_all_is_refused():
    with pytest.raises(UnvalidatedSurrogate, match="'nothing'"):
        SurrogateExport(model=_form(), inputs=(Port("u", "m/s", 0.0, 30.0), Port("alpha", "rad", 0.0, 1.0)),
                        outputs=(Port("cd"), Port("cl")))


def test_the_two_vocabularies_of_the_library_are_both_read():
    """``ValidationCase`` writes "validated"; the trust card writes "APPROVED"."""
    assert _export({"decision": "validated"}).evidence["decision"] == "validated"
    assert _export({"decision": "APPROVED"}).evidence["decision"] == "APPROVED"
    assert _export({"status": "Validated"}).evidence["decision"] == "Validated"


def test_an_object_with_a_decision_attribute_is_accepted():
    class Assessment:
        status = "validated"
        reason = "every critical quantity covered"

        def to_dict(self):
            return {"status": self.status, "reason": self.reason, "findings": []}

    export = _export(Assessment())
    assert export.evidence["decision"] == "validated" and export.evidence["findings"] == []


# ---------------------------------------------------------------------------
# the interface has to agree with the tables
# ---------------------------------------------------------------------------

def test_a_port_list_that_does_not_match_the_tables_is_refused():
    with pytest.raises(ValueError, match="input port"):
        SurrogateExport(model=_form(), inputs=(Port("u", "m/s", 0.0, 30.0),),
                        outputs=(Port("cd"), Port("cl")), evidence={"decision": "validated"})
    with pytest.raises(ValueError, match="output port"):
        SurrogateExport(model=_form(), inputs=(Port("u", "m/s", 0.0, 30.0), Port("alpha", "rad", 0.0, 1.0)),
                        outputs=(Port("cd"),), evidence={"decision": "validated"})
    with pytest.raises(ValueError, match="not an identifier"):
        Port("u [m/s]", minimum=0.0, maximum=1.0)
    with pytest.raises(ValueError, match="outside its validated range"):
        Port("u", minimum=0.0, maximum=1.0, start=2.0)
    with pytest.raises(ValueError, match="no validated range"):
        Envelope(inputs=(Port("u"),), outputs=(Port("cd"),))


def test_the_envelope_is_the_box_it_declares():
    envelope = _export().envelope
    assert envelope.contains([[0.0, 0.0], [30.0, 1.0], [15.0, 0.5]]).all()
    assert envelope.contains([[30.5, 0.5], [15.0, -0.1]]).tolist() == [False, False]
    with pytest.raises(ValueError, match="against 2 declared input"):
        envelope.contains([[1.0, 2.0, 3.0]])


# ---------------------------------------------------------------------------
# the compiled slave, through the FMI 2.0 ABI
# ---------------------------------------------------------------------------

class Slave:
    """The FMI 2.0 Co-Simulation call sequence, in ``ctypes`` -- what an importer does, written out.

    Every function used here declares its ``argtypes``: without them ``ctypes`` truncates the component pointer to a C
    ``int`` and the process dies, which is a property of the test, not of the slave.
    """

    V, D, CINT, U = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_uint

    def __init__(self, fmu: Path, folder: Path):
        with zipfile.ZipFile(fmu) as archive:
            (self.binary,) = [n for n in archive.namelist() if n.startswith("binaries/")]
            archive.extract(self.binary, folder)
        self.lib = ctypes.CDLL(str(folder / self.binary))
        self.lib.fmi2Instantiate.restype = self.V
        self.lib.fmi2GetVersion.restype = ctypes.c_char_p
        self.lib.fmi2GetTypesPlatform.restype = ctypes.c_char_p
        self.lib.fmi2Instantiate.argtypes = [ctypes.c_char_p, self.CINT, ctypes.c_char_p, ctypes.c_char_p, self.V,
                                             self.CINT, self.CINT]
        simple = {"fmi2EnterInitializationMode": [self.V], "fmi2ExitInitializationMode": [self.V],
                  "fmi2Terminate": [self.V], "fmi2Reset": [self.V], "fmi2FreeInstance": [self.V],
                  "fmi2SetupExperiment": [self.V, self.CINT, self.D, self.D, self.CINT, self.D]}
        for name, argtypes in simple.items():
            getattr(self.lib, name).argtypes = argtypes
        self.lib.fmi2SetReal.argtypes = [self.V, ctypes.POINTER(self.U), ctypes.c_size_t, ctypes.POINTER(self.D)]
        self.lib.fmi2GetReal.argtypes = [self.V, ctypes.POINTER(self.U), ctypes.c_size_t, ctypes.POINTER(self.D)]
        self.lib.fmi2GetBoolean.argtypes = [self.V, ctypes.POINTER(self.U), ctypes.c_size_t, ctypes.POINTER(self.CINT)]
        self.lib.fmi2DoStep.argtypes = [self.V, self.D, self.D, self.CINT]
        self.component = None

    def instantiate(self, guid: str | None = None) -> bool:
        self.component = self.lib.fmi2Instantiate(b"slave", 1, (guid or "{wrong}").encode(), b"", None, 1, 0)
        return self.component is not None

    def start(self, guid: str, start: float = 0.0, stop: float = 1.0) -> None:
        assert self.instantiate(guid)
        assert self.lib.fmi2SetupExperiment(self.component, 0, 0.0, start, 1, stop) == 0
        assert self.lib.fmi2EnterInitializationMode(self.component) == 0
        assert self.lib.fmi2ExitInitializationMode(self.component) == 0

    def step(self, inputs, at: float, size: float) -> tuple[np.ndarray, bool]:
        references = (self.U * len(inputs))(*range(len(inputs)))
        values = (self.D * len(inputs))(*[float(v) for v in inputs])
        assert self.lib.fmi2SetReal(self.component, references, len(inputs), values) == 0
        assert self.lib.fmi2DoStep(self.component, at, size, 1) == 0
        return self.outputs()

    def outputs(self) -> tuple[np.ndarray, bool]:
        references = (self.U * 2)(100, 101)
        values = (self.D * 2)()
        assert self.lib.fmi2GetReal(self.component, references, 2, values) == 0
        flag = (self.CINT * 1)()
        assert self.lib.fmi2GetBoolean(self.component, (self.U * 1)(200), 1, flag) == 0
        return np.array(list(values)), bool(flag[0])

    def close(self) -> None:
        if self.component is not None:
            assert self.lib.fmi2Terminate(self.component) == 0
            self.lib.fmi2FreeInstance(self.component)
            self.component = None


@NEEDS_COMPILER
def test_the_binary_exports_every_function_the_standard_requires(tmp_path):
    """``fmu_inspect`` parses ELF; on Mach-O and PE it reports "unchecked". Resolving the symbols through ``ctypes``
    is the same question asked with the dynamic loader, and it answers on every platform."""
    result = export_fmu(_export(), tmp_path / "drag.fmu")
    assert result.built and result.ok, result.summary()
    slave = Slave(Path(result.path), tmp_path)
    required = set(REQUIRED_FUNCTIONS["2"]["common"]) | set(REQUIRED_FUNCTIONS["2"]["CS"])
    missing = [name for name in sorted(required) if not hasattr(slave.lib, name)]
    assert missing == []
    assert slave.lib.fmi2GetVersion() == b"2.0"
    assert slave.lib.fmi2GetTypesPlatform() == b"default"


@NEEDS_COMPILER
def test_the_compiled_slave_reproduces_the_surrogate(tmp_path):
    export = _export()
    result = export_fmu(export, tmp_path / "drag.fmu")
    assert result.built, result.summary()
    slave = Slave(Path(result.path), tmp_path)
    slave.start(export.guid(), stop=10.0)
    for row, at in zip([[10.0, 0.0], [3.0, 0.25], [27.5, 0.9]], [0.0, 0.1, 0.2], strict=True):
        measured, inside = slave.step(row, at, 0.1)
        expected = export.predict([row])[0][0]
        assert inside
        assert np.allclose(measured, expected, rtol=1e-12, atol=1e-12), (row, measured, expected)
    slave.close()


@NEEDS_COMPILER
def test_the_envelope_flag_is_what_says_the_inputs_left_the_box(tmp_path):
    export = _export()
    slave = Slave(Path(export_fmu(export, tmp_path / "drag.fmu").path), tmp_path)
    slave.start(export.guid(), stop=10.0)
    _, inside = slave.step([15.0, 0.5], 0.0, 0.1)
    assert inside
    for outside in ([30.5, 0.5], [15.0, 1.2], [-0.5, 0.5]):
        measured, flag = slave.step(outside, 0.1, 0.1)
        assert flag is False, outside
        # the numbers still come out -- an algebraic map has no undefined point -- and the flag is the warning
        assert np.all(np.isfinite(measured))
        assert np.allclose(measured, export.predict([outside])[0][0], rtol=1e-12)
    slave.close()


@NEEDS_COMPILER
def test_the_slave_refuses_a_guid_that_is_not_its_own(tmp_path):
    export = _export()
    slave = Slave(Path(export_fmu(export, tmp_path / "drag.fmu").path), tmp_path)
    assert slave.instantiate("{00000000-0000-0000-0000-000000000000}") is False
    assert slave.instantiate(export.guid()) is True
    slave.close()


@NEEDS_COMPILER
def test_a_network_surrogate_compiles_to_the_same_numbers(tmp_path):
    export = _mlp_export()
    result = export_fmu(export, tmp_path / "mesh.fmu")
    assert result.built, result.summary()
    with zipfile.ZipFile(result.path) as archive:
        source = archive.read("sources/surrogate_fmu.c").decode()
    assert "tanh(s)" in source and "N_LAYERS 3" in source
    slave = Slave(Path(result.path), tmp_path)
    slave.start(export.guid())
    references = (Slave.U * 2)(0, 1)
    values = (Slave.D * 2)(12.0, 0.4)
    assert slave.lib.fmi2SetReal(slave.component, references, 2, values) == 0
    assert slave.lib.fmi2DoStep(slave.component, 0.0, 0.1, 1) == 0
    measured = (Slave.D * 1)()
    assert slave.lib.fmi2GetReal(slave.component, (Slave.U * 1)(100), 1, measured) == 0
    expected = export.predict([[12.0, 0.4]])[0][0][0]
    assert measured[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    slave.close()


@NEEDS_COMPILER
def test_the_binary_lives_in_the_platform_folder_the_tools_look_in(tmp_path):
    tag, extension = platform_binary_tag()
    result = export_fmu(_export(), tmp_path / "drag.fmu")
    with zipfile.ZipFile(result.path) as archive:
        names = archive.namelist()
        assert result.binary == f"binaries/{tag}/surrogate{extension}"
        binary = archive.read(result.binary)
    assert len(binary) > 1000
    assert hashlib.sha256(binary).hexdigest()                    # the bytes are there, and they are bytes
    assert any(name.startswith("binaries/") for name in names)
    assert shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
