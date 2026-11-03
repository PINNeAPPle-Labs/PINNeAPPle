"""The rules that turn checks into APPROVED / REVIEW / REJECT, set by the user (#1187).

The library ships a default (:data:`DEFAULT`); a user (or a team) sets their own and makes it their default:

    from pinneapple_veriphysics.policy import GradePolicy, save_policy, load_policy
    mine = GradePolicy(name="aero-team", residual_tol=0.01, reference_tol=0.02, inconclusive="reject")
    save_policy(mine)                    # ~/.config/pinneapple/veriphysics_policy.json: the default from now on
    load_policy()                        # -> mine;  load_policy("pinneapple-default") -> the library's

Which policy applies, first match wins: the one passed to the call (a :class:`GradePolicy`, a dict of its fields
or a JSON path), the file named by ``PINNEAPPLE_VERIPHYSICS_POLICY``, the user's saved default, the library default.
Every decision records the policy it used (its name and every field), so a grade can be reproduced.

Fields:

* tolerances of the model-grade checks (``residual_tol``, ``condition_tol``, ``reference_tol``,
  ``robustness_band``) and the finite-difference ``eps``;
* ``critical``: checks whose failure is REJECT; any other failure is REVIEW (``fail_noncritical="reject"`` makes
  every failure REJECT);
* ``inconclusive``: what a check that could not run does: ``"review"`` (the default), ``"ignore"`` or ``"reject"``;
  ``required``: checks that must have run (inconclusive there is REVIEW at least, whatever ``inconclusive`` says);
* ``min_coverage`` / ``min_score``: the share of checks that ran and the weighted share that passed
  (:class:`pinneapple_analysis.trust.trust_report.TrustReport` reads these too).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ENV = "PINNEAPPLE_VERIPHYSICS_POLICY"


def user_policy_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "pinneapple" / "veriphysics_policy.json"


@dataclass(frozen=True)
class GradePolicy:
    name: str = "pinneapple-default"
    residual_tol: float = 0.05
    condition_tol: float = 0.02
    reference_tol: float = 0.05
    robustness_band: tuple[float, float] = (3.0, 30.0)
    critical: tuple[str, ...] = ("grade.pde_residual", "grade.conditions", "grade.reference",
                                 "grade.applicability")
    fail_noncritical: str = "review"         # "review" | "reject"
    inconclusive: str = "review"             # "review" | "ignore" | "reject"
    required: tuple[str, ...] = ()
    min_coverage: float = 0.6
    min_score: float = 0.8
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.inconclusive not in ("review", "ignore", "reject"):
            raise ValueError("inconclusive is 'review', 'ignore' or 'reject'")
        if self.fail_noncritical not in ("review", "reject"):
            raise ValueError("fail_noncritical is 'review' or 'reject'")
        for k in ("residual_tol", "condition_tol", "reference_tol", "min_coverage", "min_score"):
            if float(getattr(self, k)) < 0:
                raise ValueError(f"{k} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["robustness_band"] = list(self.robustness_band)
        d["critical"], d["required"] = list(self.critical), list(self.required)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], base: GradePolicy | None = None) -> GradePolicy:
        """A policy from ``d``; fields it does not name come from ``base`` (default: the library's)."""
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown policy fields {bad}; known: {sorted(known)}")
        cur = (base or DEFAULT).to_dict()
        cur.update(d)
        cur["robustness_band"] = tuple(cur["robustness_band"])
        cur["critical"], cur["required"] = tuple(cur["critical"]), tuple(cur["required"])
        return cls(**cur)


DEFAULT = GradePolicy()


def save_policy(policy: GradePolicy, path: str | os.PathLike | None = None) -> str:
    """Write ``policy`` as JSON; without ``path`` it becomes the user's default."""
    p = Path(path) if path else user_policy_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policy.to_dict(), indent=1) + "\n")
    return str(p)


def load_policy(policy: GradePolicy | dict | str | os.PathLike | None = None) -> GradePolicy:
    """The policy that applies (see the module docstring). A dict is applied over the user's default, so a call
    can change one tolerance and keep the rest of the user's rules; ``"pinneapple-default"`` is the library's."""
    if isinstance(policy, GradePolicy):
        return policy
    if policy == DEFAULT.name:
        return DEFAULT
    if isinstance(policy, (str, os.PathLike)):
        return GradePolicy.from_dict(json.loads(Path(policy).read_text()))
    base = DEFAULT
    env = os.environ.get(ENV)
    if env:
        base = GradePolicy.from_dict(json.loads(Path(env).read_text()))
    elif user_policy_path().is_file():
        base = GradePolicy.from_dict(json.loads(user_policy_path().read_text()))
    if isinstance(policy, dict):
        return GradePolicy.from_dict(policy, base)
    return base


def decide(checks: list[dict[str, Any]], policy: GradePolicy) -> tuple[str, list[str]]:
    """APPROVED / REVIEW / REJECT and the reasons, from ``checks`` (``{code, status}``) under ``policy``."""
    reasons: list[str] = []
    level = 0                                       # 0 approved, 1 review, 2 reject
    for c in checks:
        st, code = c["status"], c["code"]
        if st == "fail":
            crit = code in policy.critical or policy.fail_noncritical == "reject"
            level = max(level, 2 if crit else 1)
            reasons.append(f"{code} failed" + (" (critical)" if crit else ""))
        elif st == "inconclusive":
            if code in policy.required:
                level = max(level, 2 if policy.inconclusive == "reject" else 1)
                reasons.append(f"{code} did not run and the policy requires it")
            elif policy.inconclusive != "ignore":
                level = max(level, 2 if policy.inconclusive == "reject" else 1)
                reasons.append(f"{code} did not run")
    ran = [c for c in checks if c["status"] != "inconclusive"]
    cov = len(ran) / len(checks) if checks else 0.0
    score = sum(c["status"] == "pass" for c in ran) / len(ran) if ran else 0.0
    if cov < policy.min_coverage:
        level = max(level, 1)
        reasons.append(f"coverage {cov:.0%} below the policy's {policy.min_coverage:.0%}")
    if ran and score < policy.min_score and level < 2:
        level = max(level, 1)
        reasons.append(f"score {score:.2f} below the policy's {policy.min_score:.2f}")
    return ("APPROVED", "REVIEW", "REJECT")[level], reasons
