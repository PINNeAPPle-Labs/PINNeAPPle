"""A model card generated from an experiment record, validated in the same call (#40).

The hub already refuses a card whose ``validation_metrics`` are empty. What it could not do is *produce* one: the
numbers were typed in by hand from a run somebody remembered. This builds the card from the record the run wrote —
``experiment.json`` and ``result.json`` — so the metrics are the recorded ones, the cost is the measured wall time,
and the validity domain is the box the problem was solved in.

    >>> from pinneapple_hub.experiment_card import card_from_record, generate_and_validate
    >>> card = card_from_record("runs/burgers_pinn")        # doctest: +SKIP
    >>> card.validate(require_cost=True, require_validity=True)   # doctest: +SKIP
    []

What it will not do: fill a field it does not have. A record with no reference solution gets no
``validation_metrics`` (the card is refused for the right reason), and a run with no measured time gets no cost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_card import ModelCard

__all__ = ["GeneratedCard", "card_from_record", "cli", "generate_and_validate"]

#: parameter names a training record uses for the things a reader calls "cost"
_COST_KEYS = ("epochs", "adam_steps", "lbfgs_steps", "n_collocation", "width", "depth", "hidden_dim", "n_layers")


def _flatten_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    """``{"relative_l2": {"u": 1e-3}}`` -> ``{"relative_l2.u": 1e-3}``: the same naming the reproduction uses."""
    out: dict[str, float] = {}
    for name, value in (metrics or {}).items():
        if isinstance(value, dict):
            for field, number in value.items():
                try:
                    out[f"{name}.{field}"] = float(number)
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _parameters(model: Any) -> int | None:
    """Trainable parameters of a saved model, when one was saved next to the record."""
    if not hasattr(model, "parameters"):
        return None
    return int(sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", True)))


def card_from_record(directory: str | Path, *, name: str = "", license: str = "apache-2.0", citation: str = "",
                     description: str = "", task: str = "", extra_limitations: list[str] | None = None,
                     model: Any = None) -> ModelCard:
    """Build the card from a record written by ``Experiment.save`` + ``ExperimentResult.save``.

    Raises ``FileNotFoundError`` when the record is not there and ``ValueError`` when it has no metrics: a card
    without a computed number is what the hub refuses, so building one would only move the refusal downstream.
    """
    root = Path(directory)
    experiment_path, result_path = root / "experiment.json", root / "result.json"
    if not experiment_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"{root} is not an experiment record: it needs experiment.json and result.json")
    experiment = json.loads(experiment_path.read_text())
    result = json.loads(result_path.read_text())
    config = dict(result.get("config") or {})
    metrics = _flatten_metrics(result.get("metrics"))
    if not metrics:
        raise ValueError(f"{root}: the record has no metrics — the run had no reference to score against, so there "
                         "is no computed validation metric to put on a card")

    problem = dict(experiment.get("problem") or {})
    environment = dict(result.get("environment") or {})
    reference = config.get("reference")
    options = dict(config.get("options") or {})
    method = str(config.get("method") or experiment.get("method") or "")
    bounds = {k: list(v) for k, v in (problem.get("domain_bounds") or {}).items()}
    ranges = {k: list(v) for k, v in (problem.get("field_ranges") or {}).items()}

    validity: dict[str, Any] = {f"{k} (bounds)": v for k, v in bounds.items()}
    validity.update({f"{k} (observed range)": v for k, v in ranges.items()})
    if reference:
        validity["compared against"] = f"{reference} solution on {result.get('n_eval_points')} points"

    limitations = [
        "trained and compared inside the bounds above: outside them the error is not measured by this record",
        "the validation metrics are those of one run with the seed in the provenance, not a distribution over runs",
    ]
    if not problem.get("units"):
        limitations.append("the problem declares no units, so the numbers are in the archive's own convention")
    limitations.extend(extra_limitations or [])

    cost: dict[str, Any] = {"wall_time_s": float(result.get("wall_time_s") or 0.0),
                            "n_eval_points": int(result.get("n_eval_points") or 0)}
    cost.update({k: options[k] for k in _COST_KEYS if k in options})
    n_params = _parameters(model)
    if n_params is not None:
        cost["trainable_parameters"] = n_params

    from pinneapple_data.provenance import build_provenance

    provenance = build_provenance(
        solver=f"{method} (pinneapple_physics.Experiment)", seed=int(config.get("seed", 0)),
        data_hash=str(result.get("problem_fingerprint") or ""), commit=str(environment.get("commit") or ""),
        version=str(environment.get("pinneapple") or ""),
        mesh={"domain_bounds": bounds} if bounds else None,
        record=str(root), problem=str(config.get("problem") or ""), reference=str(reference or ""))

    return ModelCard(
        name=name or str(config.get("name") or root.name), description=description or
        f"{method} solution of {config.get('problem') or problem.get('name') or 'a problem'}"
        f"{' scored against ' + str(reference) if reference else ''}",
        task=task, architecture=method, architecture_config={k: options[k] for k in ("hidden_dim", "n_layers",
                                                                                     "width", "depth")
                                                              if k in options},
        equations=[str(problem.get("pde"))] if problem.get("pde") else [],
        domain_bounds=bounds, training_config=options,
        data_lineage={"record": str(root), "problem": str(config.get("problem") or problem.get("name") or ""),
                      "problem_fingerprint": result.get("problem_fingerprint", "")},
        validation_metrics=metrics,
        reference_source=(f"{reference} reference solution of the same problem, {result.get('n_eval_points')} "
                          f"evaluation points" if reference else
                          "the recorded metrics, whose reference this record does not name"),
        validity_range=validity, known_limitations=limitations, cost=cost,
        provenance=provenance, citation=citation, license=license,
        pinneapple_version=str(environment.get("pinneapple") or ""),
        tags=[t for t in (method, "generated-card") if t])


class GeneratedCard:
    """A card and the verdict on it: what was generated, and every reason it would be refused."""

    def __init__(self, card: ModelCard, problems: list[str]) -> None:
        self.card = card
        self.problems = problems

    @property
    def ok(self) -> bool:
        return not self.problems

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.card.save(str(path))
        return path

    def to_markdown(self) -> str:
        head = "" if self.ok else ("\n\n> **Refused:**\n" + "\n".join(f"> - {p}" for p in self.problems))
        return self.card.to_markdown() + head

    def __repr__(self) -> str:
        return f"GeneratedCard({self.card.name!r}, ok={self.ok}, {len(self.problems)} problems)"


def generate_and_validate(directory: str | Path, **kwargs: Any) -> GeneratedCard:
    """The one command: build the card and run the hub's own validation, including cost and validity."""
    card = card_from_record(directory, **kwargs)
    return GeneratedCard(card, card.validate(require_cost=True, require_validity=True))


def cli(argv: list[str] | None = None) -> int:
    """``pinneapple model-card <record>``: generate the card, validate it, and write it.

    Exit codes: ``0`` the card passed, ``1`` it was generated and refused (the reasons are printed), ``2`` no record
    or a bad invocation. A refused card is written anyway when ``--out`` is given — the reasons are the useful part,
    and throwing the draft away would hide which field the record could not fill.
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="pinneapple model-card",
        description="Generate a model card from an experiment record and validate it (metrics, cost and validity "
                    "domain are all required).",
        epilog="Docs: docs/guides/model_cards.md")
    ap.add_argument("record", help="the folder with experiment.json and result.json")
    ap.add_argument("--out", "-o", metavar="FILE", help="write the card (JSON) here")
    ap.add_argument("--markdown", metavar="FILE", help="also write the card as Markdown here")
    ap.add_argument("--name", default="", help="card name (default: the experiment's name)")
    ap.add_argument("--license", default="apache-2.0")
    ap.add_argument("--citation", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--limitation", action="append", default=[],
                    help="an extra known limitation (repeatable)")
    a = ap.parse_args(argv)
    try:
        generated = generate_and_validate(a.record, name=a.name, license=a.license, citation=a.citation,
                                          description=a.description, task=a.task,
                                          extra_limitations=list(a.limitation))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if a.out:
        generated.save(a.out)
    if a.markdown:
        from pathlib import Path
        Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown).write_text(generated.to_markdown())
    if generated.ok:
        print(f"{generated.card.name}: card generated and validated ({len(generated.card.validation_metrics)} "
              f"validation metrics)")
        return 0
    print(f"{generated.card.name}: card refused", file=sys.stderr)
    for problem in generated.problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1
