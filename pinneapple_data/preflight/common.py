"""Findings and the pre-flight report shared by the OpenFOAM and CalculiX rule sets."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SECTIONS = [("mesh", "Mesh"), ("materials", "Material properties"), ("boundary_conditions", "Boundary conditions"),
            ("initial_conditions", "Initial conditions"), ("solver", "Solver settings")]
SEVERITY = {"fail": 3, "warn": 2, "info": 1, "pass": 0}


@dataclass
class Finding:
    section: str                      # one of SECTIONS
    status: str                       # pass | info | warn | fail
    title: str                        # short: "Missing thermal conductivity"
    detail: str = ""                  # explanation: what is wrong and what the solver will do
    fix: str = ""                     # how to correct it
    file: Optional[str] = None
    line: Optional[int] = None
    entity: Optional[str] = None      # patch / set / material / field the finding is about
    rule: str = ""                    # stable id for the rule
    evidence: Optional[str] = None    # what the solver does with it (from our runs)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass
class Report:
    solver: str
    findings: List[Finding] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add(self, *a, **k) -> Finding:
        f = Finding(*a, **k)
        self.findings.append(f)
        return f

    def to_dict(self) -> Dict[str, Any]:
        worst = max((SEVERITY[f.status] for f in self.findings), default=0)
        overall = {3: "FAIL", 2: "WARNING"}.get(worst, "PASS")
        sections = []
        for key, label in SECTIONS:
            fs = [f for f in self.findings if f.section == key]
            w = max((SEVERITY[f.status] for f in fs), default=-1)
            sections.append({"key": key, "label": label, "status": {3: "fail", 2: "warn", 1: "pass", 0: "pass", -1: "na"}[w],
                             "checks": len(fs), "problems": sum(f.status in ("warn", "fail") for f in fs)})
        order = sorted(self.findings, key=lambda f: (-SEVERITY[f.status], [s for s, _ in SECTIONS].index(f.section)))
        return {"status": overall, "solver": self.solver, "sections": sections,
                "findings": [f.to_dict() for f in order], "info": self.info,
                "counts": {s: sum(f.status == s for f in self.findings) for s in ("fail", "warn", "info", "pass")}}


def line_of(text: Optional[str], pattern: str, flags: int = re.M) -> Optional[int]:
    if not text:
        return None
    m = re.search(pattern, text, flags)
    if not m:
        return None
    pos = m.start() + len(m.group(0)) - len(m.group(0).lstrip())     # skip leading whitespace/newlines
    return text.count("\n", 0, pos) + 1


def num(v: Any) -> Optional[float]:
    """Last number in an OpenFOAM entry ('nu [0 2 -1 0 0 0 0] 1e-05' -> 1e-05, 'uniform 0' -> 0)."""
    if v is None:
        return None
    s = str(v)
    m = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s.split("]")[-1])
    return float(m[-1]) if m else None


def vec(v: Any) -> Optional[List[float]]:
    m = re.search(r"\(\s*([^()]*)\)", str(v or ""))
    if not m:
        return None
    try:
        return [float(x) for x in m.group(1).split()]
    except ValueError:
        return None
