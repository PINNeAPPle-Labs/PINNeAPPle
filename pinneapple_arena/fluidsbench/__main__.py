"""``python -m pinneapple_arena.fluidsbench`` — inspect a FluidsBench contract, a support release, or a package.

    python -m pinneapple_arena.fluidsbench datasets                     # what the checkout publishes, and its status
    python -m pinneapple_arena.fluidsbench dataset airfrans            # one dataset: splits, composite, required fields
    python -m pinneapple_arena.fluidsbench support manifest.json -c standard
    python -m pinneapple_arena.fluidsbench validate submissions/airfrans/my-model-v1 --official
"""
from __future__ import annotations

import argparse
import sys

from .contract import Checkout, ContractError
from .package import validate_package
from .support import SupportError, load_support_release

__all__ = ["main"]


def _checkout(args) -> Checkout:
    return Checkout.find(getattr(args, "repo", None))


def _datasets(args) -> int:
    checkout = _checkout(args)
    rows = [("dataset", "status", "submissions", "open splits")]
    for dataset_id in checkout.available():
        contract = checkout.dataset(dataset_id)
        rows.append((dataset_id, contract.status, "open" if contract.submissions_open else "closed",
                     ", ".join(contract.open_splits()) or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return 0


def _dataset(args) -> int:
    contract = _checkout(args).dataset(args.dataset_id)
    print(f"{contract.name} ({contract.dataset_id}) {contract.dataset_version} — status {contract.status}")
    print(f"  submissions open: {contract.submissions_open}")
    print(f"  evaluation reference: {contract.evaluation_reference_version}")
    for split in contract.splits:
        print(f"  split {split.id:<20} {split.case_count:>5} cases  case set {split.case_set_id:<18} "
              f"sha256 {split.expected_sha256[:12]}")
    if contract.composite.components:
        print("  overall_score components:")
        for component in contract.composite.components:
            cap = f" cap {component.cap:g}" if component.cap is not None else ""
            print(f"    {component.metric_id:<38} weight {component.weight:<6g} {component.transform}{cap}")
    for entry in contract.required_fields:
        print(f"  required field {entry.get('field_id')} ({entry.get('component_count')} components, "
              f"{entry.get('domain')})")
    print(f"  support release {contract.support_release_id} sha256 {contract.support_manifest_sha256[:16]}")
    print("  files read:")
    for name, digest in contract.digests().items():
        print(f"    {digest[:16]}  {name}")
    return 0


def _support(args) -> int:
    release = load_support_release(args.manifest, args.case_set)
    print(f"release {release.release_id} ({release.dataset_id}) case set {release.case_set_id}: "
          f"{len(release)} cases, digests verified")
    for support_id, definition in release.definitions.items():
        print(f"  support {support_id}: {definition.domain}, {definition.location_mode}/{definition.table_format}, "
              f"{len(definition.bindings)} metric binding(s)")
    if release.case_ids:
        case_id = args.case_id or release.case_ids[0]
        for case in release.supports(case_id):
            print(f"  {case_id} / {case.support_id}: {case.entity_count} entities, "
                  f"{case.coordinates.shape[1]}-D, quantities {sorted(case.targets)}, "
                  f"target shapes { {k: v.shape for k, v in case.targets.items()} }")
    return 0


def _validate(args) -> int:
    checkout = None
    contract = None
    if args.repo or args.official:
        checkout = _checkout(args)
        if args.dataset_id:
            contract = checkout.dataset(args.dataset_id)
    report = validate_package(args.package, contract=contract, checkout=checkout, run_official=args.official)
    print(report)
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pinneapple_arena.fluidsbench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="FluidsBench submission checkout (default $FLUIDSBENCH_REPO)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="the datasets the checkout publishes").set_defaults(func=_datasets)

    dataset = sub.add_parser("dataset", help="one dataset: splits, composite, required fields, files read")
    dataset.add_argument("dataset_id")
    dataset.set_defaults(func=_dataset)

    support = sub.add_parser("support", help="verify a support release chain and describe one case")
    support.add_argument("manifest")
    support.add_argument("-c", "--case-set", default="standard")
    support.add_argument("--case-id")
    support.set_defaults(func=_support)

    validate = sub.add_parser("validate", help="recompute a package's identities, optionally with the official validator")
    validate.add_argument("package")
    validate.add_argument("--dataset-id", help="the dataset the package belongs to, for the composite check")
    validate.add_argument("--official", action="store_true", help="also run scripts/validate_submission.py")
    validate.set_defaults(func=_validate)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ContractError, SupportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
