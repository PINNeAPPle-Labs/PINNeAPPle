"""FMU preflight: check that a compiled FMU can actually be loaded *before* a
sweep or a twin depends on it.

``fmi_runner`` / ``sweep`` hand an ``.fmu`` to fmpy and find out it is broken
only when instantiation fails, often minutes into a campaign. This module
opens the archive statically (nothing is loaded or executed) and reports:

- FMI version, model name, ``modelIdentifier`` per interface (ME / CS / SE);
- which platform binaries are shipped (``binaries/<platform>/<id>.<ext>``);
- for ELF64 shared objects (Linux), the **exported** symbol table, compared
  with the functions the FMI standard requires for the declared interfaces;
- cheap modelDescription consistency checks (unique value references,
  outputs listed in ``ModelStructure`` are declared ``causality="output"``).

Function lists: taken from the FMI standard headers ``fmi2FunctionTypes.h`` /
``fmi3FunctionTypes.h`` (Modelica Association, BSD-2-Clause), grouped by the
interface they belong to. The FMI specification requires an FMU to export
every function of each interface type it declares.

How this was built: REA (``inspect-artifact``, ``inspect-binary-layout``) was
run on the binaries of the Modelica Reference-FMUs to confirm what a correct
FMU exports; REA's deep native analysis (``open_binary``) needs a Hopper /
Ghidra / IDA provider, which was not available, so function bodies were not
analysed. PE (Windows ``.dll``) and Mach-O (``.dylib``) symbol tables are not
parsed here and are reported as ``unchecked``.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FMUReport", "inspect_fmu", "elf_exported_functions", "REQUIRED_FUNCTIONS"]

_FMI2_ME = {"fmi2EnterEventMode", "fmi2NewDiscreteStates", "fmi2EnterContinuousTimeMode",
            "fmi2CompletedIntegratorStep", "fmi2SetTime", "fmi2SetContinuousStates", "fmi2GetDerivatives",
            "fmi2GetEventIndicators", "fmi2GetContinuousStates", "fmi2GetNominalsOfContinuousStates"}
_FMI2_CS = {"fmi2SetRealInputDerivatives", "fmi2GetRealOutputDerivatives", "fmi2DoStep", "fmi2CancelStep",
            "fmi2GetStatus", "fmi2GetRealStatus", "fmi2GetIntegerStatus", "fmi2GetBooleanStatus",
            "fmi2GetStringStatus"}
_FMI2_COMMON = {"fmi2GetTypesPlatform", "fmi2GetVersion", "fmi2SetDebugLogging", "fmi2Instantiate",
                "fmi2FreeInstance", "fmi2SetupExperiment", "fmi2EnterInitializationMode",
                "fmi2ExitInitializationMode", "fmi2Terminate", "fmi2Reset", "fmi2GetReal", "fmi2GetInteger",
                "fmi2GetBoolean", "fmi2GetString", "fmi2SetReal", "fmi2SetInteger", "fmi2SetBoolean",
                "fmi2SetString", "fmi2GetFMUstate", "fmi2SetFMUstate", "fmi2FreeFMUstate",
                "fmi2SerializedFMUstateSize", "fmi2SerializeFMUstate", "fmi2DeSerializeFMUstate",
                "fmi2GetDirectionalDerivative"}

_FMI3_ME = {"fmi3InstantiateModelExchange", "fmi3EnterContinuousTimeMode", "fmi3CompletedIntegratorStep",
            "fmi3SetTime", "fmi3SetContinuousStates", "fmi3GetContinuousStateDerivatives",
            "fmi3GetEventIndicators", "fmi3GetContinuousStates", "fmi3GetNominalsOfContinuousStates",
            "fmi3GetNumberOfEventIndicators", "fmi3GetNumberOfContinuousStates"}
_FMI3_CS = {"fmi3InstantiateCoSimulation", "fmi3EnterStepMode", "fmi3GetOutputDerivatives", "fmi3DoStep"}
_FMI3_SE = {"fmi3InstantiateScheduledExecution", "fmi3ActivateModelPartition"}
_T = ("Float32", "Float64", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64",
      "Boolean", "String", "Binary", "Clock")
_FMI3_COMMON = ({"fmi3GetVersion", "fmi3SetDebugLogging", "fmi3FreeInstance", "fmi3EnterInitializationMode",
                 "fmi3ExitInitializationMode", "fmi3EnterEventMode", "fmi3Terminate", "fmi3Reset",
                 "fmi3GetNumberOfVariableDependencies", "fmi3GetVariableDependencies", "fmi3GetFMUState",
                 "fmi3SetFMUState", "fmi3FreeFMUState", "fmi3SerializedFMUStateSize", "fmi3SerializeFMUState",
                 "fmi3DeserializeFMUState", "fmi3GetDirectionalDerivative", "fmi3GetAdjointDerivative",
                 "fmi3EnterConfigurationMode", "fmi3ExitConfigurationMode", "fmi3GetIntervalDecimal",
                 "fmi3GetIntervalFraction", "fmi3GetShiftDecimal", "fmi3GetShiftFraction",
                 "fmi3SetIntervalDecimal", "fmi3SetIntervalFraction", "fmi3SetShiftDecimal",
                 "fmi3SetShiftFraction", "fmi3EvaluateDiscreteStates", "fmi3UpdateDiscreteStates"}
                | {f"fmi3Get{t}" for t in _T} | {f"fmi3Set{t}" for t in _T})

#: ``REQUIRED_FUNCTIONS[version][interface]`` (interface: "common", "ME", "CS", "SE").
REQUIRED_FUNCTIONS: dict[str, dict[str, set[str]]] = {
    "2": {"common": _FMI2_COMMON, "ME": _FMI2_ME, "CS": _FMI2_CS},
    "3": {"common": _FMI3_COMMON, "ME": _FMI3_ME, "CS": _FMI3_CS, "SE": _FMI3_SE},
}

_IFACE_TAGS = {"ModelExchange": "ME", "CoSimulation": "CS", "ScheduledExecution": "SE"}


@dataclass
class BinaryCheck:
    platform: str
    path: str
    format: str                       # "elf64", "pe", "macho", "unknown"
    exported: list[str] | None     # None when the format is not parsed
    missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.exported is None:
            return "unchecked"
        return "ok" if not self.missing else "missing-functions"


@dataclass
class FMUReport:
    path: str
    fmi_version: str
    model_name: str
    interfaces: dict[str, str]        # "ME"/"CS"/"SE" -> modelIdentifier
    binaries: list[BinaryCheck] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    n_variables: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and all(b.status != "missing-functions" for b in self.binaries)

    def summary(self) -> str:
        lines = [f"{Path(self.path).name}: FMI {self.fmi_version}, model {self.model_name!r}, "
                 f"interfaces {sorted(self.interfaces)}, {self.n_variables} variables"]
        for b in self.binaries:
            extra = f" missing {b.missing[:5]}{'...' if len(b.missing) > 5 else ''}" if b.missing else ""
            lines.append(f"  {b.platform}/{Path(b.path).name}: {b.status}{extra}")
        lines += [f"  problem: {p}" for p in self.problems]
        return "\n".join(lines)


def elf_exported_functions(data: bytes) -> list[str]:
    """Names of defined, global/weak, default-visibility FUNC symbols in the
    ``.dynsym`` table of a little-endian ELF64 object. Raises ValueError for
    anything else."""
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only little-endian ELF64 is supported")
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        (_, sh_type, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize) = struct.unpack_from(
            "<IIQQQQIIQQ", data, off)
        sections.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))
    SHT_DYNSYM = 11
    out: list[str] = []
    for sh_type, off, size, link, entsize in sections:
        if sh_type != SHT_DYNSYM:
            continue
        str_off, str_size = sections[link][1], sections[link][2]
        strtab = data[str_off: str_off + str_size]
        for k in range(size // (entsize or 24)):
            st_name, st_info, st_other, st_shndx, _, _ = struct.unpack_from("<IBBHQQ", data, off + k * 24)
            bind, typ, vis = st_info >> 4, st_info & 0xF, st_other & 0x3
            if st_shndx == 0 or typ not in (2, 10) or bind not in (1, 2) or vis != 0:  # UNDEF / not FUNC|IFUNC / local / hidden
                continue
            end = strtab.find(b"\x00", st_name)
            out.append(strtab[st_name:end].decode("utf-8", "replace"))
    return sorted(set(out))


def _format(blob: bytes) -> str:
    if blob[:4] == b"\x7fELF":
        return "elf64" if blob[4] == 2 else "elf32"
    if blob[:2] == b"MZ":
        return "pe"
    if blob[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
        return "macho"
    return "unknown"


def inspect_fmu(path: str | Path) -> FMUReport:
    """Static inspection of an ``.fmu`` archive (see module docstring)."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "modelDescription.xml" not in names:
            raise ValueError(f"{path.name}: no modelDescription.xml")
        root = ET.fromstring(z.read("modelDescription.xml"))
        version = (root.get("fmiVersion") or "").split(".")[0]
        if version not in REQUIRED_FUNCTIONS:
            raise ValueError(f"{path.name}: unsupported fmiVersion {root.get('fmiVersion')!r}")
        ifaces = {_IFACE_TAGS[c.tag]: c.get("modelIdentifier", "") for c in root if c.tag in _IFACE_TAGS}
        rep = FMUReport(str(path), root.get("fmiVersion", ""), root.get("modelName", ""), ifaces)
        if not ifaces:
            rep.problems.append("no ModelExchange/CoSimulation/ScheduledExecution element")

        # --- variables & structure -------------------------------------
        mv = root.find("ModelVariables")
        variables = list(mv) if mv is not None else []
        rep.n_variables = len(variables)
        vrs: dict[str, int] = {}
        causality_by_index: dict[int, str] = {}
        for i, v in enumerate(variables, start=1):
            key = v.get("valueReference")
            if version == "3":   # FMI 3: references are per variable type, unique overall
                vrs[key] = vrs.get(key, 0) + 1
            else:                # FMI 2: unique per base type
                base = next(iter(v), None)
                k2 = f"{base.tag if base is not None else '?'}:{key}"
                vrs[k2] = vrs.get(k2, 0) + 1
            # FMI 2 ModelStructure refers to the 1-based variable index,
            # FMI 3 to the valueReference.
            ref_key = int(key) if version == "3" else i
            causality_by_index[ref_key] = v.get("causality", "local")
        dup = [k for k, n in vrs.items() if n > 1]
        if dup:
            rep.problems.append(f"duplicate valueReference(s): {dup[:5]}")
        ms = root.find("ModelStructure")
        if ms is not None:
            for out in ms.findall("Outputs/Unknown") + ms.findall("Output"):
                ref = int(out.get("index") or out.get("valueReference") or 0)
                if causality_by_index.get(ref) not in (None, "output"):
                    rep.problems.append(f"ModelStructure output {ref} is not a causality='output' variable")

        # --- binaries ------------------------------------------------------
        req = set(REQUIRED_FUNCTIONS[version]["common"])
        for tag in ifaces:
            req |= REQUIRED_FUNCTIONS[version].get(tag, set())
        model_ids = set(ifaces.values())
        found_any = False
        for n in names:
            parts = n.split("/")
            if len(parts) != 3 or parts[0] != "binaries" or not parts[2]:
                continue
            stem = parts[2].rsplit(".", 1)[0]
            if stem not in model_ids:
                continue
            found_any = True
            blob = z.read(n)
            fmt = _format(blob)
            exported: list[str] | None = None
            if fmt == "elf64":
                try:
                    exported = elf_exported_functions(blob)
                except ValueError as exc:
                    rep.problems.append(f"{n}: {exc}")
            missing = sorted(req - set(exported)) if exported is not None else []
            rep.binaries.append(BinaryCheck(parts[1], n, fmt, exported, missing))
        if not found_any and "sources/" not in {n[:8] for n in names}:
            rep.problems.append("no binary for any modelIdentifier and no sources/")
    return rep
