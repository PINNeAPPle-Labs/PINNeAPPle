"""``pp.zoo.anchors``: the published model family, grouped by the engineering problem it answers.

The zoo is a catalogue of everything that passed the gate; an *anchor* is the opposite view -- one named,
productised entry per problem, pointing at the model(s) that answer it, and carrying the numbers that already
exist in their model cards. Nothing here invents a metric: every value is read from ``manifest.json`` and the
model card, so the family page cannot drift from what ships (``tests/test_zoo_anchors.py`` checks it).

    import pinneapple as pp
    pp.zoo.anchors.list()                          # ['anchor-aero', 'anchor-contact', ...]
    a = pp.zoo.anchors.info("anchor-aero")         # name, domain, members, metrics, validity, trust
    print(pp.zoo.anchors.card("anchor-aero"))      # one Markdown page per anchor
    print(pp.zoo.anchors.catalogue())              # the whole family, for docs/modelos
    data = pp.zoo.anchors.public()                 # JSON for the site's /modelos page

An anchor whose model has no approved card, no held-out metric or an incomplete validity range is a problem,
not a product: :func:`problems` lists those, and the test fails on a non-empty list.
"""
from __future__ import annotations

import builtins
import json
from dataclasses import dataclass, field
from typing import Any

from . import _artifacts as art

__all__ = ["FAMILY", "ANCHORS", "list", "info", "card", "catalogue", "public", "dumps", "problems", "AnchorInfo"]

#: The family as it is published. ``version`` moves when a member is added or replaced, never when a metric moves.
FAMILY: dict[str, Any] = {
    "name": "PINNeAPPle Anchor",
    "short": "PPA",
    "version": "1.0.0",
    "promise": ("One named model per engineering problem, published with the error it was measured at, the range "
                "it holds in and the checks it passed -- so the answer arrives with the condition under which it "
                "is true."),
}

#: One entry per problem. ``models`` are zoo model names; the anchor's numbers come from their cards.
ANCHORS: dict[str, dict[str, Any]] = {
    "anchor-aero": {
        "name": "Anchor-Aero",
        "domain": "Aerodynamics",
        "problem": "Lift, drag and pitching moment of a 2-D section from its shape and angle of attack",
        "replaces": ("A RANS sweep per design point (meshing + solve) during shape exploration, at Re 4e6, "
                     "attached to mildly separated flow"),
        "models": ["airfoil_cst_rans_mlp"],
        "guard_metric": "test/Cl_mae",
    },
    "anchor-thermal-forced": {
        "name": "Anchor-Thermal-Forced",
        "domain": "Thermal",
        "problem": "Base-to-air thermal resistance and pressure drop of a plate-fin heat sink in ducted forced convection",
        "replaces": "A conjugate-heat-transfer solve per fin geometry during heat-sink sizing",
        "models": ["heatsink_forced_mlp"],
        "guard_metric": "test/R_mean_abs_pct",
    },
    "anchor-thermal-natural": {
        "name": "Anchor-Thermal-Natural",
        "domain": "Thermal",
        "problem": "Base-to-air thermal resistance of a plate-fin heat sink in natural convection",
        "replaces": "A buoyancy-driven CFD solve per fin geometry where no fan is available",
        "models": ["heatsink_natural_mlp"],
        "guard_metric": "test/R_mean_abs_pct",
    },
    "anchor-contact": {
        "name": "Anchor-Contact",
        "domain": "Terramechanics",
        "problem": "Wheel-soil contact forces of a rigid grousered wheel on lunar regolith simulant",
        "replaces": "A discrete-element wheel-soil simulation per slip and load condition during mobility studies",
        "models": ["rover_contact_grc1_viper"],
        "guard_metric": "test/Fz_rmse_N",
    },
}


@dataclass
class AnchorInfo:
    """One anchor, resolved against the shipped models: what it answers, where it holds, how well it did."""

    id: str
    name: str
    domain: str
    problem: str
    replaces: str
    guard_metric: str
    members: list[dict[str, Any]] = field(default_factory=list)

    @property
    def models(self) -> builtins.list[str]:
        return [m["model"] for m in self.members]

    @property
    def trust(self) -> dict[str, str]:
        """The trust decision of each member, by model name (``APPROVED`` / ``REVIEW`` / ``REJECT``)."""
        return {m["model"]: m["trust"] for m in self.members}

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        """Held-out metrics of each member, by model name -- the numbers the card was issued with."""
        return {m["model"]: m["metrics"] for m in self.members}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "domain": self.domain, "problem": self.problem,
                "replaces": self.replaces, "guard_metric": self.guard_metric, "members": self.members}


def list() -> builtins.list[str]:  # noqa: A001 - mirrors pp.zoo.list()
    """Anchor ids, sorted."""
    return sorted(ANCHORS)


def _member(name: str) -> dict[str, Any]:
    from . import info as _model_card

    items = art.manifest()["models"]
    if name not in items:
        raise KeyError(f"anchor refers to {name!r}, which is not in the zoo manifest "
                       f"(known models: {', '.join(sorted(items))})")
    entry = items[name]
    card = _model_card(name)
    return {
        "model": name,
        "version": entry["version"],
        "task": entry.get("task", ""),
        "trust": card.trust_decision,
        "metrics": {k: float(card.validation_metrics[k]) for k in entry.get("guard_metrics", [])
                    if k in card.validation_metrics},
        "acceptance": dict(entry.get("acceptance", {})),
        "validity_range": dict(card.validity_range),
        "limitations": builtins.list(card.known_limitations),
        "license": card.license,
        "hf_repo": entry.get("hf_repo", ""),
        "revision": entry.get("revision", ""),
        "citation": card.citation,
    }


def info(anchor_id: str) -> AnchorInfo:
    """The anchor resolved against the shipped models. Raises ``KeyError`` for an unknown id or a member that
    is not in the manifest -- a dangling anchor is a bug, not a warning."""
    if anchor_id not in ANCHORS:
        raise KeyError(f"unknown anchor {anchor_id!r} (known: {', '.join(list())})")
    spec = ANCHORS[anchor_id]
    return AnchorInfo(id=anchor_id, name=spec["name"], domain=spec["domain"], problem=spec["problem"],
                      replaces=spec["replaces"], guard_metric=spec["guard_metric"],
                      members=[_member(m) for m in spec["models"]])


def problems() -> builtins.list[str]:
    """Why an anchor is not publishable yet (empty list = every anchor has an approved card with its numbers)."""
    found = []
    for anchor_id in list():
        a = info(anchor_id)
        for m in a.members:
            if m["trust"] != "APPROVED":
                found.append(f"{a.name}: {m['model']} has trust decision {m['trust'] or 'none'!r}")
            if not m["metrics"]:
                found.append(f"{a.name}: {m['model']} has no held-out metric in its card")
            elif a.guard_metric not in m["metrics"]:
                found.append(f"{a.name}: {m['model']} does not report the guard metric {a.guard_metric}")
            if not m["validity_range"]:
                found.append(f"{a.name}: {m['model']} has no validity range")
    return found


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}" if value.is_integer() or abs(value) >= 1e4 else f"{value:.3g}"
    if isinstance(value, (builtins.list, tuple)) and len(value) == 2:
        return " – ".join(_fmt(v) for v in value)
    return str(value)


def _is_range(value: Any) -> bool:
    return (isinstance(value, (builtins.list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value))


def card(anchor_id: str) -> str:
    """One anchor as a Markdown page: the problem, what it replaces, the numbers, the range, what it does not do."""
    a = info(anchor_id)
    lines = [f"## {a.name}", "",
             f"**Domain.** {a.domain}", "",
             f"**Answers.** {a.problem}.", "",
             f"**Replaces.** {a.replaces}.", ""]
    for m in a.members:
        metrics = ", ".join(f"`{k}` {_fmt(v)}" for k, v in m["metrics"].items()) or "no held-out metric"
        lines.append(f"**Model.** `{m['model']}` v{m['version']} -- trust decision **{m['trust'] or 'none'}**, "
                     f"held-out {metrics}.")
        lines.append("")
        ranges = {k: v for k, v in m["validity_range"].items() if _is_range(v)}
        notes = {k: v for k, v in m["validity_range"].items() if not _is_range(v) and v != ""}
        if ranges:
            lines.append("**Holds in.** " + ", ".join(f"`{k}` {_fmt(v)}" for k, v in ranges.items()) + ".")
            lines.append("")
        for k, v in notes.items():
            lines.append(f"**`{k}`.** {_fmt(v)}")
            lines.append("")
        if m["limitations"]:
            lines.append("**Does not hold for:**")
            lines.append("")
            lines += [f"- {lim}" for lim in m["limitations"]]
            lines.append("")
        if m["hf_repo"]:
            lines.append(f"**Weights.** [`{m['hf_repo']}`](https://huggingface.co/{m['hf_repo']}/tree/"
                         f"{m['revision']}) ({m['license']}).")
            lines.append("")
    lines += [f"Load it with `pp.zoo.load(\"{a.models[0]}\")`; the error above is the one measured against the "
              f"reference named in the model card (`pp.zoo.info(...)`), not an average over cases.", ""]
    return "\n".join(lines)


def catalogue() -> str:
    """The whole family as one Markdown page -- the source of the site's ``/modelos`` page."""
    lines = [f"# {FAMILY['name']} models", "", FAMILY["promise"], "",
             f"Family version {FAMILY['version']}. Every number below is the one the model card was issued with; "
             "the range and the checks are in each card.", "",
             "| Anchor | Domain | Answers | Model | Held-out | Trust |", "|---|---|---|---|---|---|"]
    for anchor_id in list():
        a = info(anchor_id)
        for m in a.members:
            metrics = ", ".join(f"`{k.split('/')[-1]}` {_fmt(v)}" for k, v in m["metrics"].items())
            lines.append(f"| **{a.name}** | {a.domain} | {a.problem} | `{m['model']}` | {metrics} | {m['trust']} |")
    lines += ["", f"{len(list())} anchors over {len({m for a in (info(i) for i in list()) for m in a.models})} "
                  "models. Metric values are `<split>/<metric>` in the card; a lower value is better.", ""]
    return "\n".join(lines)


def public() -> dict[str, Any]:
    """The family as JSON for the site: no weights, no Hub download, only what is already in the cards."""
    return {
        "family": dict(FAMILY),
        "anchors": [info(anchor_id).to_dict() for anchor_id in list()],
        "problems": problems(),
    }


def dumps(**kwargs: Any) -> str:
    """``public()`` as JSON text, ready to be written next to the site."""
    return json.dumps(public(), indent=2, ensure_ascii=False, **kwargs)
