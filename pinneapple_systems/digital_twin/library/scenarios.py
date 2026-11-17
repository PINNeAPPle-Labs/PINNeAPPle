"""Named scenarios, the manifest they roll up to, and the status of every twin.

Why a scenario is a first-class object
--------------------------------------
A parameter set that says something is worth keeping. "The well after two years of decline",
"the furnace on high pulverised coal", "the hall in a cold climate" are the cases a study is
actually about, and retyping them as keyword arguments is how two people end up comparing two
different runs and calling it a disagreement.

A :class:`Scenario` is a slug, a name, a handful of parameter overrides and a line saying where
the numbers came from. It runs through the same :meth:`ReferenceModel.simulate` as anything else,
so a scenario can never bypass the range checks, and it saves to JSON so a scenario can be sent to
somebody.

Scenarios that are meant to fail
--------------------------------
Four of the eleven cases below trip an invariant on purpose: a 3:1 ratio in a single compression
stage discharges at 143 degC, a 69 K lift in a single-stage ammonia machine discharges above 150
degC, a heavier booster runs out of landing propellant, and one cold-climate data hall has an
economiser that costs more than the chiller it replaces. Those are **findings**, not failures of
the library: the model is doing its job by refusing the case. They carry ``expect="finding"``, and
:func:`check` is strict in both directions — a finding that stops being reported has changed the
physics, which is a regression just as much as a passing case that starts failing.

The manifest
------------
:func:`manifest` is the registry's status page as data: every twin, its fidelity, its named
scenarios, and — for each — the verdict, the worst balance residual against its own tolerance and
the parameter that was deliberately extrapolated. It is what a CI job reads and what a reviewer
looks at instead of eleven terminal scrollbacks.

What this is not
----------------
This is the *registry* half of the shared-twin-specification issue (#1802): discoverability, named
scenarios and result status. The typed unit algebra, the component graph with ports, the solver
adapter protocol, the cross-fidelity consistency checks and the VeriPhysics applicability gate are
**not** here, and the issue is not closed by this module. What is here is the part the library can
carry today without pretending.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import LibraryError, ParameterError, TwinRun
from .registry import TWINS, build, catalogue

__all__ = [
    "Scenario",
    "SCENARIOS",
    "scenario_names",
    "run_scenario",
    "manifest",
    "save_scenarios",
    "load_scenarios",
]


@dataclass(frozen=True)
class Scenario:
    """A named parameter set: what it is, what it changes, and where the numbers came from."""

    slug: str
    title: str
    values: dict[str, float] = field(default_factory=dict)
    note: str = ""
    source: str = ""
    #: ``"pass"`` for a case that has to run clean; ``"finding"`` for a case that is meant to trip an
    #: invariant, because the operating point is infeasible and the model should say so.
    expect: str = "pass"

    def __post_init__(self) -> None:
        if self.slug not in TWINS:
            raise LibraryError(f"scenario {self.title!r} names {self.slug!r}, which is not a twin; "
                               f"the library has {sorted(TWINS)}")
        unknown = set(self.values) - set(TWINS[self.slug].parameters)
        if unknown:
            raise ParameterError(f"scenario {self.title!r}: {self.slug} has no parameter(s) {sorted(unknown)}")
        if self.expect not in ("pass", "finding"):
            raise ParameterError(f"scenario {self.title!r}: expect must be 'pass' or 'finding', not {self.expect!r}")

    @property
    def name(self) -> str:
        """``<twin>/<scenario>``: unique in the registry and readable in a report."""
        return f"{self.slug}/{self.title}"

    def run(self, *, strict: bool = True) -> TwinRun:
        """Run the twin with this scenario's overrides applied to its declared defaults."""
        return build(self.slug).simulate(dict(self.values), strict=strict)

    def as_dict(self) -> dict[str, Any]:
        return {"slug": self.slug, "title": self.title, "values": dict(self.values),
                "note": self.note, "source": self.source, "expect": self.expect}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(slug=data["slug"], title=data["title"], values=dict(data.get("values") or {}),
                   note=data.get("note", ""), source=data.get("source", ""),
                   expect=data.get("expect", "pass"))


#: The named scenarios of the library. Each one is a case the model was built to answer, and each
#: one is expected to pass: a scenario that fails its own invariants is a finding, and the manifest
#: says so rather than hiding it.
SCENARIOS: dict[str, Scenario] = {}


def _register(*scenarios: Scenario) -> None:
    for scenario in scenarios:
        if scenario.name in SCENARIOS:
            raise LibraryError(f"scenario {scenario.name!r} is declared twice")
        SCENARIOS[scenario.name] = scenario


_register(
    Scenario("fpso_subsea", "mature_well",
             {"production_days": 330.0, "productivity_index": 1.1e-9, "water_cut": 0.42,
              "reservoir_pressure": 27.5e6},
             note="the same well after a year on stream: pressure down, water cut up, productivity falling",
             source="declared scenario, not a reservoir study"),
    Scenario("blast_furnace", "high_pci",
             {"coal_rate": 200.0, "coke_rate": 300.0, "blast_temperature": 1253.15},
             note="more pulverised coal and less coke, at the same hot metal: the gas gets leaner",
             source="declared scenario inside the model's range"),
    Scenario("chemical_plant", "low_pressure",
             {"pressure": 40.0e5, "reactor_inlet_temperature": 460.0},
             note="half the pressure: the equilibrium conversion falls and the recycle has to work harder",
             source="declared scenario inside the model's range"),
    Scenario("reusable_booster", "heavier_stage", {"dry_mass": 38_000.0, "meco_velocity": 2_050.0},
             note="a heavier stage at a faster staging point runs out of landing propellant: the reserve, not "
                  "the ascent, is what limits the case",
             source="declared scenario inside the model's range", expect="finding"),
    Scenario("lunar_base", "six_crew",
             {"crew": 6.0, "array_area": 1_800.0},
             note="six crew instead of four, with the array and the storage grown to match",
             source="declared scenario inside the model's range"),
    Scenario("pipeline_network", "low_delivery_pressure",
             {"delivery_pressure": 4.5e6, "station_ratio": 1.45},
             note="a lower offtake pressure: more gas moves, and the line packs less of it",
             source="declared scenario inside the model's range"),
    Scenario("compression_station", "single_stage", {"stages": 1.0},
             note="the same 3:1 ratio in one stage discharges at about 143 degC, above the 125 degC machine "
                  "limit: this is why the station inter-cools",
             source="declared scenario inside the model's range", expect="finding"),
    Scenario("green_hydrogen_plant", "high_current_density",
             {"current_density": 2.2, "cell_voltage": 1.95},
             note="a stack pushed harder: more hydrogen per cell, and a worse specific energy",
             source="declared scenario inside the model's range"),
    Scenario("desalination_plant", "high_recovery",
             {"recovery": 0.55, "net_driving_pressure": 16.0},
             note="squeezing more permeate out of the same feed: the brine concentrates and the energy rises",
             source="declared scenario inside the model's range"),
    Scenario("industrial_refrigeration", "frozen_storage",
             {"load_temperature": -30.0, "cooling_medium_temperature": 32.0},
             note="a freezer instead of a chiller: a 69 K lift in one stage discharges above 150 degC, which is "
                  "where a single-stage ammonia machine stops and a two-stage one starts",
             source="declared scenario inside the model's range", expect="finding"),
    Scenario("data_center_cooling", "cold_climate", {"wet_bulb": 12.0, "free_cooling_hours": 6_500.0},
             note="a cold site: with only a 4 K lift the chiller costs less than the declared economiser share, "
                  "so a fixed free-cooling fraction makes the annual PUE *worse* than the design one — the "
                  "economiser has to be sized against the plant it replaces, not against a share of IT",
             source="declared scenario inside the model's range", expect="finding"),
)


def scenario_names(slug: str | None = None) -> list[str]:
    """Every registered scenario, or the ones of one twin, in name order."""
    return sorted(name for name in SCENARIOS if slug is None or name.startswith(f"{slug}/"))


def run_scenario(name: str, *, strict: bool = True) -> TwinRun:
    """Run a scenario by name: ``run_scenario("pipeline_network/low_delivery_pressure")``."""
    try:
        scenario = SCENARIOS[name]
    except KeyError as exc:
        raise LibraryError(f"no scenario {name!r}; the registry has {sorted(SCENARIOS)}") from exc
    return scenario.run(strict=strict)


def _status(run: TwinRun) -> dict[str, Any]:
    """One run reduced to what a reviewer needs: the verdict and what the worst balance was."""
    worst = max(run.balances, key=lambda b: abs(b.residual) / b.tolerance, default=None)
    return {
        "ok": run.ok,
        "balanced": run.balanced,
        "invariants_hold": run.invariants_hold,
        "n_balances": len(run.balances),
        "n_checks": len(run.checks),
        "worst_balance": None if worst is None else {
            "name": worst.name,
            "residual": worst.residual,
            "tolerance": worst.tolerance,
            "fraction_of_tolerance": abs(worst.residual) / worst.tolerance,
        },
        "failed_checks": [c.name for c in run.checks if not c.passed],
        "extrapolated": list(run.extrapolated),
    }


def manifest(*, include_defaults: bool = True, include_scenarios: bool = True) -> dict[str, Any]:
    """The registry's status page as data: every twin, its scenarios and the verdict of each run.

    A run that fails is recorded and *not* raised: a manifest is a report on the library, and a
    library with a red entry is exactly what it exists to show. Use :func:`check` to turn the
    manifest into an exit code.
    """
    entries = []
    for entry in catalogue():
        system: dict[str, Any] = {
            "slug": entry.slug, "title": entry.title, "fidelity": entry.fidelity,
            "status": entry.status, "n_parameters": len(entry.model.parameters),
            "physics": entry.physics,
        }
        if include_defaults:
            system["default"] = _status(build(entry.slug).simulate())
        if include_scenarios:
            system["scenarios"] = {
                name: {**_status(SCENARIOS[name].run()), "note": SCENARIOS[name].note,
                       "expect": SCENARIOS[name].expect}
                for name in scenario_names(entry.slug)}
        entries.append(system)
    statuses = [s["default"] for s in entries if "default" in s]
    statuses += [v for s in entries if "scenarios" in s for v in s["scenarios"].values()]
    findings = [v for v in statuses if v.get("expect") == "finding"]
    return {
        "schema": "pinneapple.digital_twin.manifest/1",
        "created": time.time(),
        "systems": entries,
        "summary": {
            "systems": len(entries),
            "scenarios": len(SCENARIOS),
            "runs": len(statuses),
            "passing": sum(1 for s in statuses if s["ok"]),
            "findings_expected": len(findings),
            "findings_reported": sum(1 for s in findings if not s["ok"]),
            "failing": [s["failed_checks"] for s in statuses
                        if not s["ok"] and s.get("expect") != "finding"],
            "worst_fraction_of_tolerance": max(
                (s["worst_balance"]["fraction_of_tolerance"] for s in statuses if s["worst_balance"]),
                default=0.0),
        },
    }


def check() -> list[str]:
    """Every reason the manifest is not green, as sentences. Empty means the library is green.

    Strict in both directions: a case that had to pass and did not is a problem, and a case that was
    meant to report a finding and now passes is one too — the second is how a regression shows up if
    the physics somehow becomes more permissive than the operating case allows.
    """
    problems = []
    for system in manifest()["systems"]:
        for name, status in [("default", system.get("default") or {})] + list(
                (system.get("scenarios") or {}).items()):
            expected = status.get("expect", "pass")
            failed = status.get("failed_checks") or ["a balance"]
            if expected == "pass" and not status.get("ok", False):
                problems.append(f"{system['slug']}:{name}: " + ", ".join(failed))
            elif expected == "finding" and status.get("ok", False):
                problems.append(f"{system['slug']}:{name}: expected a finding and got a clean run")
    return problems


def save_scenarios(path: str | Path, scenarios: list[Scenario] | None = None) -> Path:
    """Write scenarios to JSON, so a case can be handed to somebody else."""
    p = Path(path)
    payload = {"schema": "pinneapple.digital_twin.scenarios/1",
               "scenarios": [s.as_dict() for s in (scenarios if scenarios is not None else SCENARIOS.values())]}
    p.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return p


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    """Read scenarios from JSON, rejecting anything the library would refuse to run."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    loaded = [Scenario.from_dict(item) for item in data.get("scenarios", [])]
    return {s.name: s for s in loaded}


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_systems.digital_twin.library.scenarios",
                                 description="Run the library's named scenarios and report their status.")
    ap.add_argument("--out", help="write the manifest JSON here")
    ap.add_argument("--check", action="store_true", help="exit 1 if any run fails")
    ap.add_argument("--list", action="store_true", help="list the scenarios and stop")
    a = ap.parse_args(argv)

    if a.list:
        for name in scenario_names():
            print(f"{name:<44s} {SCENARIOS[name].note}")
        return 0

    data = manifest()
    if a.out:
        Path(a.out).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    summary = data["summary"]
    print(f"{summary['systems']} systems, {summary['scenarios']} scenarios, "
          f"{summary['passing']}/{summary['runs']} runs passing "
          f"({summary['findings_reported']} of {summary['findings_expected']} expected findings reported), "
          f"worst balance at {summary['worst_fraction_of_tolerance']:.2f} of its tolerance")
    problems = check()
    for problem in problems:
        print(f"FAIL {problem}")
    if a.check and problems:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI
    sys.exit(_main())
