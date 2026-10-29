"""Stage B informed: the LLM's structured claims, and why they are not evidence.

The specification requires that a generated artifact arrive with more than
code: the assumptions it was built on, its dependencies, the units it uses,
the equations it believes it implemented, and its known limitations. This
module captures exactly that, and then does the part that matters -- it
confronts those declarations with the specification and with reality:

* declared units must parse, and must agree with the specification's units
  for the same quantity;
* declared dependencies must be in the static import allowlist, so the
  artifact will not claim a dependency the validator will reject;
* an assumption the specification does not contain is recorded as an
  **unverified extra assumption**, not silently merged into the spec;
* missing declarations are reported as missing.

The language model's own description is never treated as evidence that the
code is correct. It is an input to stage C/E, and every claim in it that can
be checked mechanically is checked. This is the difference between "the
model says it used SI units" and "the units in the artifact are SI".
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .report import Finding, Status
from .spec import PhysicalSpecification
from .static import DEFAULT_ALLOWED_ROOTS, StaticResult, validate_json_artifact, validate_python_artifact
from .units import UnitError, are_compatible, parse_unit

GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["code", "assumptions", "dependencies", "units", "equations", "limitations"],
    "properties": {
        "code": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "units": {"type": "object"},
        "equations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}

_SYSTEM_PROMPT = """You are writing a Python artifact for PINNeAPPle, a physics-AI library.
Respond with ONE JSON object and nothing else, of exactly this shape:

{"code": "<python source>",
 "assumptions": ["<assumption the code relies on>", ...],
 "dependencies": ["<imported module>", ...],
 "units": {"<variable>": "<unit>", ...},
 "equations": ["<equation actually implemented>", ...],
 "limitations": ["<what this artifact does not establish>", ...]}

Rules:
- "code" must only use modules in the ALLOWED MODULES list given to you, and
  only APIs that exist in the installed PINNeAPPle version.
- "units" must use explicit units (m, mm, s, year, kg, K, Pa, ...). Never omit
  the unit of a physical quantity.
- "assumptions" must list only what the code actually rests on. If you had to
  guess a value, say so here rather than presenting it as measured.
- "limitations" must state what the artifact does NOT demonstrate.
- Do not claim validation, certification or agreement with a real system.
"""


@dataclass
class GeneratedArtifact:
    """The LLM's proposal plus its structured claims. Untrusted by construction."""

    code: str
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)
    equations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    elapsed_s: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_sha256_source": self.code,
            "assumptions": list(self.assumptions),
            "dependencies": list(self.dependencies),
            "units": dict(self.units),
            "equations": list(self.equations),
            "limitations": list(self.limitations),
            "provider": self.provider, "model": self.model,
            "elapsed_s": round(self.elapsed_s, 4),
        }


def build_prompt(spec: PhysicalSpecification) -> str:
    """A prompt carrying the specification, not a paraphrase of it."""
    return (
        "SPECIFICATION (JSON):\n"
        f"{json.dumps(spec.to_dict(), indent=2, sort_keys=True, default=str)}\n\n"
        f"ALLOWED MODULES:\n{json.dumps(sorted(DEFAULT_ALLOWED_ROOTS))}\n\n"
        "Write the artifact that implements this specification and return the JSON envelope."
    )


def parse_generation_response(raw: str, *, provider: str = "", model: str = "",
                              elapsed_s: float = 0.0) -> Tuple[GeneratedArtifact, StaticResult]:
    """Parse and schema-check an LLM response. Never raises on malformed JSON."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        result = StaticResult(findings=[Finding(
            id="generation.json", status=Status.FAIL, severity="error",
            message=f"the model did not return valid JSON: {exc}",
            evidence={"raw_head": raw[:400]},
            justification="without the structured envelope there are no claims to check")])
        return GeneratedArtifact(code="", raw_response=raw), result

    schema_result = validate_json_artifact(payload, GENERATION_SCHEMA, name="generation envelope")
    artifact = GeneratedArtifact(
        code=str(payload.get("code", "")),
        assumptions=[str(x) for x in payload.get("assumptions", []) or []],
        dependencies=[str(x) for x in payload.get("dependencies", []) or []],
        units={str(k): str(v) for k, v in (payload.get("units") or {}).items()},
        equations=[str(x) for x in payload.get("equations", []) or []],
        limitations=[str(x) for x in payload.get("limitations", []) or []],
        provider=provider, model=model, elapsed_s=elapsed_s, raw_response=raw,
    )
    return artifact, schema_result


def check_claims(artifact: GeneratedArtifact,
                 spec: PhysicalSpecification) -> List[Finding]:
    """Confront the model's declarations with the specification and the unit table.

    Returns findings only; no side effects, no network, no model call.
    """
    findings: List[Finding] = []

    if not artifact.code.strip():
        findings.append(Finding(id="generation.empty", status=Status.FAIL, severity="error",
                                message="the model returned no code"))

    # 1. dependencies must be something the static validator will accept
    for dependency in artifact.dependencies:
        root = dependency.split(".")[0]
        if root not in DEFAULT_ALLOWED_ROOTS and not root.startswith("pinneapple_"):
            findings.append(Finding(
                id="generation.dependency", status=Status.FAIL, severity="error",
                message=f"declared dependency '{dependency}' is outside the allowed module set",
                evidence={"dependency": dependency},
                justification="the static validator would reject this import; catching it here names "
                            "the problem before execution"))

    # 2. every declared unit must parse, and match the spec's unit for the same name
    for name, unit in artifact.units.items():
        try:
            parse_unit(unit)
        except UnitError as exc:
            findings.append(Finding(id="generation.unit", status=Status.FAIL, severity="error",
                                    message=f"declared unit for '{name}' is invalid: {exc}",
                                    evidence={"name": name, "unit": unit}))
            continue
        declared = spec.parameters.get(name)
        if declared is not None and not are_compatible(unit, declared.unit):
            findings.append(Finding(
                id="generation.unit_mismatch", status=Status.FAIL, severity="error",
                message=f"the model says '{name}' is in '{unit}' but the specification declares "
                        f"'{declared.unit}'",
                evidence={"name": name, "claimed": unit, "spec": declared.unit}))

    # 3. assumptions the spec does not contain are unverified extras
    known = {a.strip().lower() for a in spec.assumptions}
    for assumption in artifact.assumptions:
        if assumption.strip().lower() not in known:
            findings.append(Finding(
                id="generation.extra_assumption", status=Status.NOT_RUN, severity="warning",
                message=f"the model declares an assumption the specification does not contain: "
                        f"'{assumption}'",
                evidence={"assumption": assumption},
                justification="it is kept as an explicit, unverified assumption rather than merged "
                            "into the specification"))

    # 4. silent gaps
    if not artifact.units:
        findings.append(Finding(id="generation.no_units", status=Status.NOT_RUN, severity="warning",
                                message="the model declared no units"))
    if not artifact.limitations:
        findings.append(Finding(id="generation.no_limitations", status=Status.NOT_RUN,
                                severity="warning",
                                message="the model declared no limitations -- an artifact that claims "
                                        "none is claiming more than it can establish"))
    if not artifact.equations:
        findings.append(Finding(id="generation.no_equations", status=Status.NOT_RUN,
                                severity="warning",
                                message="the model declared no equations"))

    if not any(f.blocks for f in findings):
        findings.append(Finding(
            id="generation.claims", status=Status.PASS, severity="info",
            message=f"{len(artifact.units)} unit(s), {len(artifact.dependencies)} dependency(ies) and "
                    f"{len(artifact.assumptions)} assumption(s) checked against the specification; "
                    "this verifies the declarations, not the code"))
    return findings


def validate_generated(artifact: GeneratedArtifact, spec: PhysicalSpecification,
                       *, filename: str = "artifact.py") -> Tuple[List[Finding], Any]:
    """Claim checks + the full static validation of the returned code."""
    findings = check_claims(artifact, spec)
    static = validate_python_artifact(artifact.code, filename=filename)
    findings.extend(static.findings)
    return findings, static


def generate_artifact(spec: PhysicalSpecification, *,
                      provider: str = "anthropic",
                      model: Optional[str] = None,
                      api_key: Optional[str] = None,
                      conversation_store: Any = None) -> GeneratedArtifact:
    """Ask a provider for an artifact + envelope, through the existing dispatch.

    Uses :func:`pinneapple_llm._dispatch.call_llm`, so every provider already
    supported (``anthropic``, ``openai``, ``ollama``) works unchanged and the
    call is logged to the conversation store when one is given.
    """
    from .._dispatch import call_llm

    started = time.monotonic()
    raw = call_llm(
        build_prompt(spec), provider=provider, model=model, api_key=api_key,
        system=_SYSTEM_PROMPT, json_mode=True, module="qa.generate_artifact",
        conversation_store=conversation_store,
    )
    elapsed = time.monotonic() - started
    artifact, _schema = parse_generation_response(
        raw, provider=provider, model=model or "", elapsed_s=elapsed)
    return artifact
