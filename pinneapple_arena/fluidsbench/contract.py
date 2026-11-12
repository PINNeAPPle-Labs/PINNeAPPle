"""The FluidsBench contract, read from an official checkout instead of copied into this repository.

FluidsBench publishes, for every dataset, a *submission specification* (``benchmark-specs/<id>/submission-spec.json``)
that names the supported splits with their SHA-256, the scoring-support release that owns coordinates, weights and ground
truth, the metric bindings and the composite that the leaderboard ranks on. Those files change with each release, so this
module reads them at run time and records the digest of every file it read: a result package can then say exactly which
contract it was scored against. Nothing here restates a number the benchmark owns.

    from pinneapple_arena.fluidsbench import Checkout

    checkout = Checkout.find()                      # $FLUIDSBENCH_REPO, or an explicit path
    contract = checkout.dataset("airfrans")
    print(contract.status, contract.open_splits(), contract.support_manifest)
    composite = contract.composite.score({"surface_pressure_rel_l2": 6.1, "cd_r2": 0.93, ...})

The checkout is a clone of https://github.com/FluidsBench/fluidsbench-submission (Apache-2.0). Nothing in this package
replaces it: :meth:`Checkout.validate_submission` runs the official validator from that clone when it is asked to.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Checkout", "DatasetContract", "SplitSpec", "ScoreComponent", "ScoreComposite", "ContractError",
           "KNOWN_DATASETS", "REPO_URL", "DEFAULT_ENV_VAR"]

REPO_URL = "https://github.com/FluidsBench/fluidsbench-submission"
DEFAULT_ENV_VAR = "FLUIDSBENCH_REPO"

#: The datasets the benchmark publishes (the website and ``benchmark-specs/README.md``), by id. Whether one accepts
#: submissions, and which splits are open, is read from its specification, never assumed here.
KNOWN_DATASETS = {
    "airfrans": "AirfRANS",
    "ahmedml": "AhmedML",
    "drivaerml": "DrivAerML",
    "drivaernetplusplus": "DrivAerNet++",
    "windsorml": "WindsorML",
    "blendednet": "BlendedNet",
    "hiliftaeroml": "HiLiftAeroML",
    "rotor37": "Rotor37",
    "vki-ls59": "VKI-LS59",
}

_TRANSFORMS = frozenset({"bounded_error", "bounded_quality", "physics_null_skill"})


class ContractError(ValueError):
    """The checkout, a specification, or a split does not have the shape the contract requires."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"{path} is not valid JSON: {error}") from error


@dataclass(frozen=True)
class SplitSpec:
    """One split a dataset supports: which case set scores it, and the digest of its case list."""
    id: str
    label: str
    case_set_id: str
    case_count: int
    index_file: str
    expected_sha256: str
    case_id_status: str = ""

    @property
    def open(self) -> bool:
        """A split whose case ids are still provisional cannot be scored against a frozen support."""
        return self.case_id_status in ("official", "")


@dataclass(frozen=True)
class ScoreComponent:
    """One term of the published composite: its weight, how its metric becomes a score, and the evidence it needs."""
    metric_id: str
    weight: float
    transform: str
    cap: float | None = None
    baseline_error: float | None = None

    def score(self, error: float) -> float:
        """The component score published in the metric reference, for one measured metric value.

        ``bounded_error``: ``clip(100 * (1 - error / cap), 0, 100)``.
        ``bounded_quality``: ``100 * clip(value, 0, 1)`` (the value is a quality, e.g. R², not an error).
        ``physics_null_skill``: ``100 * (1 - error / baseline_error)``, deliberately unclipped, so a method worse than
        the published null keeps negative skill."""
        if self.transform == "bounded_error":
            if not self.cap or self.cap <= 0:
                raise ContractError(f"component {self.metric_id!r} uses bounded_error without a positive cap")
            return float(min(max(100.0 * (1.0 - float(error) / self.cap), 0.0), 100.0))
        if self.transform == "bounded_quality":
            return float(100.0 * min(max(float(error), 0.0), 1.0))
        if self.transform == "physics_null_skill":
            if not self.baseline_error or self.baseline_error <= 0:
                raise ContractError(f"component {self.metric_id!r} has no positive published baseline_error")
            return float(100.0 * (1.0 - float(error) / self.baseline_error))
        raise ContractError(f"unknown composite transform {self.transform!r} for {self.metric_id!r}")


@dataclass(frozen=True)
class ScoreComposite:
    """``overall_score`` as the dataset publishes it: components, their weights, and the declared tolerance."""
    metric_id: str = "overall_score"
    components: tuple[ScoreComponent, ...] = ()
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    tolerance: float = 1e-6
    status: str = "active"

    def __post_init__(self) -> None:
        total = sum(c.weight for c in self.components)
        if self.components and abs(total - 1.0) > 1e-9:
            raise ContractError(f"composite weights must sum to one, got {total!r}")

    def score(self, values: Mapping[str, float]) -> float:
        """Recompute the composite from the component metric values the submission reports."""
        missing = [c.metric_id for c in self.components if c.metric_id not in values]
        if missing:
            raise ContractError(f"composite needs values for {missing}")
        return float(sum(c.weight * c.score(values[c.metric_id]) for c in self.components))

    def group_scores(self, values: Mapping[str, float]) -> dict[str, float]:
        """The intermediate ``field_score`` / ``force_score`` / ``diagnostic_score`` reductions, renormalised."""
        out: dict[str, float] = {}
        for name, members in self.groups.items():
            chosen = [c for c in self.components if c.metric_id in members]
            weight = sum(c.weight for c in chosen)
            if not chosen or weight <= 0:
                continue
            out[name] = float(sum(c.weight * c.score(values[c.metric_id]) for c in chosen) / weight)
        return out


@dataclass(frozen=True)
class DatasetContract:
    """Everything this package needs to know about one dataset, as its own specification states it."""
    dataset_id: str
    dataset_name: str
    dataset_version: str
    status: str
    submissions_open: bool
    evaluation_reference_version: str
    splits: tuple[SplitSpec, ...]
    composite: ScoreComposite
    support_release_id: str
    support_manifest: Path | None
    support_manifest_sha256: str
    required_fields: tuple[Mapping[str, Any], ...]
    ranking: Mapping[str, Any]
    spec_path: Path
    root: Path
    _digests: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ splits
    def split(self, split_id: str) -> SplitSpec:
        for spec in self.splits:
            if spec.id == split_id:
                return spec
        raise ContractError(f"{self.dataset_id} has no split {split_id!r}; it publishes "
                            f"{[s.id for s in self.splits]}")

    def open_splits(self) -> list[str]:
        """The splits a submission can be scored on: the specification open, ids frozen, case set available."""
        if not self.submissions_open:
            return []
        return [s.id for s in self.splits if s.open]

    def split_case_ids(self, split_id: str, verify: bool = True) -> list[str]:
        """The case ids of a split, checked against the digest its specification publishes."""
        spec = self.split(split_id)
        path = (self.spec_path.parent / spec.index_file).resolve()
        if verify:
            attempt = path if path.exists() else next(iter(self.root.glob(f"**/{spec.index_file}")), path)
            if attempt.exists():
                digest = sha256_file(attempt)
                if spec.expected_sha256 and digest != spec.expected_sha256:
                    raise ContractError(f"{self.dataset_id}/{split_id}: split file digest {digest} does not match the "
                                        f"published {spec.expected_sha256}")
                path = attempt
        payload = _read_json(path)
        ids = payload.get("case_ids")
        if not isinstance(ids, list) or not ids:
            raise ContractError(f"{path} has no case_ids")
        if spec.case_count and len(ids) != spec.case_count:
            raise ContractError(f"{path} lists {len(ids)} cases, the specification says {spec.case_count}")
        return [str(i) for i in ids]

    # ------------------------------------------------------------- provenance
    def digests(self) -> dict[str, str]:
        """SHA-256 of every contract file this object was built from, relative to the checkout root."""
        return dict(self._digests)

    @property
    def name(self) -> str:
        return self.dataset_name or KNOWN_DATASETS.get(self.dataset_id, self.dataset_id)


class Checkout:
    """A clone of the FluidsBench submission repository, read-only."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "benchmark-specs").is_dir():
            raise ContractError(f"{self.root} is not a FluidsBench submission checkout "
                                f"(no benchmark-specs/); clone {REPO_URL}")
        self._contracts: dict[str, DatasetContract] = {}

    @classmethod
    def find(cls, root: str | Path | None = None, env_var: str = DEFAULT_ENV_VAR) -> Checkout:
        """``root``, else ``$FLUIDSBENCH_REPO``, else a clear error naming the clone command."""
        if root is not None:
            return cls(root)
        from_env = os.environ.get(env_var)
        if from_env:
            return cls(from_env)
        raise ContractError(f"no FluidsBench checkout: set {env_var} or pass the path; clone it with\n"
                            f"    git clone {REPO_URL}")

    # ------------------------------------------------------------------ read
    def available(self) -> list[str]:
        """The dataset ids with a specification in this checkout."""
        out = []
        for folder in sorted((self.root / "benchmark-specs").iterdir()):
            if (folder / "submission-spec.json").is_file():
                out.append(folder.name)
        return out

    def dataset(self, dataset_id: str) -> DatasetContract:
        if dataset_id in self._contracts:
            return self._contracts[dataset_id]
        spec_path = self.root / "benchmark-specs" / dataset_id / "submission-spec.json"
        if not spec_path.is_file():
            raise ContractError(f"{dataset_id!r} has no submission-spec.json in {self.root}; available: "
                                f"{self.available()}")
        spec = _read_json(spec_path)
        digests = {_rel(spec_path, self.root): sha256_file(spec_path)}

        support = spec.get("scoring_support") or {}
        manifest_rel = support.get("manifest_file")
        # the specification names its files relative to its own folder (benchmark-specs/<id>/); a checkout that stores
        # them relative to the repository root is accepted too, so a rehearsal or a copy is not silently unreadable
        manifest = None
        if manifest_rel:
            for base in (spec_path.parent, self.root):
                candidate = (base / manifest_rel)
                if candidate.is_file():
                    manifest = candidate
                    break
            if manifest is None:
                manifest = spec_path.parent / manifest_rel
        if manifest is not None and manifest.is_file():
            digests[_rel(manifest, self.root)] = sha256_file(manifest)

        splits = []
        for entry in spec.get("splits") or []:
            splits.append(SplitSpec(id=str(entry["id"]), label=str(entry.get("label", entry["id"])),
                                    case_set_id=str(entry.get("case_set_id", "standard")),
                                    case_count=int(entry.get("case_count", 0)),
                                    index_file=str(entry.get("index_file", "")),
                                    expected_sha256=str(entry.get("sha256", "")),
                                    case_id_status=str(entry.get("case_id_status", ""))))

        methods = self.root / "benchmark-specs" / dataset_id / "methodology-contract.json"
        required = ()
        if methods.is_file():
            digests[_rel(methods, self.root)] = sha256_file(methods)
            required = tuple(_read_json(methods).get("required_predicted_fields") or ())

        contract = DatasetContract(
            dataset_id=str(spec.get("dataset_id", dataset_id)),
            dataset_name=str(spec.get("dataset_name", KNOWN_DATASETS.get(dataset_id, dataset_id))),
            dataset_version=str(spec.get("dataset_version", "")),
            status=str(spec.get("status", "")),
            submissions_open=bool(support.get("submissions_open", False)),
            evaluation_reference_version=str(spec.get("evaluation_reference_version", "")),
            splits=tuple(splits),
            composite=_composite(spec.get("overall_score_composite") or {}, spec.get("component_score_groups") or {}),
            support_release_id=str(support.get("release_id", "")),
            support_manifest=manifest,
            support_manifest_sha256=str(support.get("manifest_sha256", "")),
            required_fields=required,
            ranking=dict(spec.get("ranking") or {}),
            spec_path=spec_path,
            root=self.root,
            _digests=digests,
        )
        self._contracts[dataset_id] = contract
        return contract

    # ------------------------------------------------- representative workflow
    def profile_definition(self, dataset_id: str) -> Mapping[str, Any]:
        """The dataset's profile definition, whose file is part of the published contract."""
        spec = _read_json(self.root / "benchmark-specs" / dataset_id / "submission-spec.json")
        return dict(spec.get("profile_definition") or {})

    def validate_submission(self, submission_dir: str | Path, *, timeout: float = 900.0) -> dict:
        """Run the official contributor-stage validator from this checkout on a result directory.

        The validator is the authority; this never reimplements it. Returns ``{"ok", "returncode", "stdout",
        "stderr", "command"}``; ``ok`` is ``returncode == 0``."""
        validator = self.root / "scripts" / "validate_submission.py"
        if not validator.is_file():
            raise ContractError(f"{validator} is missing from the checkout")
        command = [sys.executable, str(validator), "--contributor-stage", str(Path(submission_dir).resolve())]
        done = subprocess.run(command, cwd=str(self.root), capture_output=True, text=True, timeout=timeout)
        return {"ok": done.returncode == 0, "returncode": done.returncode, "stdout": done.stdout,
                "stderr": done.stderr, "command": " ".join(command)}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _groups(payload: Any) -> dict[str, tuple[str, ...]]:
    """``component_score_groups`` as published: ``{"groups": [{"metric_id", "component_metric_ids"}, ...]}``.

    A plain mapping of name → members is accepted too, because a rehearsal or a future revision may write it that way."""
    out: dict[str, tuple[str, ...]] = {}
    if isinstance(payload, Mapping):
        if "groups" in payload:
            payload = payload["groups"]
        elif all(isinstance(v, (list, tuple)) for v in payload.values()):
            return {str(k): tuple(str(m) for m in v) for k, v in payload.items()}
        else:
            return out
    for entry in payload or ():
        if isinstance(entry, Mapping) and "metric_id" in entry:
            members = entry.get("component_metric_ids") or entry.get("components") or ()
            out[str(entry["metric_id"])] = tuple(str(m) for m in members)
    return out


def _composite(payload: Mapping[str, Any], groups: Any) -> ScoreComposite:
    components = []
    for entry in payload.get("components") or []:
        transform = str(entry.get("transform", ""))
        if transform not in _TRANSFORMS:
            raise ContractError(f"composite component {entry.get('metric_id')!r} has unknown transform {transform!r}")
        components.append(ScoreComponent(metric_id=str(entry["metric_id"]), weight=float(entry["weight"]),
                                         transform=transform, cap=entry.get("cap"),
                                         baseline_error=entry.get("baseline_error")))
    return ScoreComposite(metric_id=str(payload.get("metric_id", "overall_score")), components=tuple(components),
                          groups=_groups(groups),
                          tolerance=float(payload.get("tolerance", 1e-6)),
                          status=str(payload.get("status", "active")))
