"""A FluidsBench adapter: the official scoring contract read from its own repository, on PINNeAPPle's trust layer.

FluidsBench (https://fluidsbench.org) evaluates machine-learned surrogates on realistic fluid dynamics — AirfRANS, AhmedML,
DrivAerML, DrivAerNet++, WindsorML and more — with published splits, a benchmark-owned scoring support, and a leaderboard
ranked on a composite of field errors, integrated forces and flow profiles. This package does the part a participant owns,
and does not re-declare the part the benchmark owns:

===========================  ==========================================================================
``Checkout``                 reads ``benchmark-specs/<id>/submission-spec.json`` from a clone of the submission
                             repository: status, open splits with their digests, the scoring-support release, the
                             composite and its caps, the required predicted fields — and hashes every file it read
``load_support_release``     reads and verifies ``manifest → case set → chunk → ground-truth table``; the coordinates,
                             weights and truth are the benchmark's, joined by ``support_id``
``map_to_support``           places a model's own output on those entities (identity, by id, nearest, linear) and reports
                             what it reached: scored/total entities, weight fraction, unmapped, extrapolated
``relative_l2_statistics``   the published reductions, with chunk-safe sufficient statistics and a macro-average across
                             cases — never an average of chunk L2 values
``build_package``            writes the result directory (``submission.json``, ``evaluation-evidence.json``,
                             ``discretization.json`` and its case lines, ``metrics/cases.json``) with its digest chain
``validate_package``         recomputes, from the package's own numbers, the identities the maintainers' validator
                             checks, and runs that official validator from a checkout when asked
``assess_submission``        scope, integrity and applicability as evidence-ledger records, and a submit / review / hold
                             decision: the VeriPhysics side of a submission

    from pinneapple_arena.fluidsbench import Checkout, load_support_release, map_to_support, score_support

    checkout = Checkout.find()                                  # $FLUIDSBENCH_REPO
    contract = checkout.dataset("airfrans")
    release = load_support_release(contract.support_manifest, "standard")
    mapped = map_to_support(release.support(case_id, "two-dimensional-domain-native"), fields=predictions,
                            points=my_mesh)
    score = score_support(release.support(case_id, "two-dimensional-domain-native"), mapped.values,
                          coverage=mapped.coverage.as_dict())

What this is not: a claim that a number computed here is the official one. The official evaluator writes the score and
the maintainers verify the package; this package makes sure that what leaves the machine is complete, declared, and
consistent with the published rules — and says so when it is not.

Experimental (a labs package of PINNeAPPle): its API may change between versions. See ``docs/guides/fluidsbench.md``.
"""
from __future__ import annotations

from .contract import (
                       DEFAULT_ENV_VAR,
                       KNOWN_DATASETS,
                       REPO_URL,
                       Checkout,
                       ContractError,
                       DatasetContract,
                       ScoreComponent,
                       ScoreComposite,
                       SplitSpec,
)
from .mapping import (
                       Coverage,
                       CoverageError,
                       MappedPredictions,
                       MappingError,
                       map_to_support,
                       native_entity_counts,
                       summarize_mapping,
)
from .metrics import (
                       MetricError,
                       Statistics,
                       SupportScore,
                       aggregate,
                       competition_rank,
                       macro_average,
                       r2,
                       reduction_value,
                       relative_l1,
                       relative_l2,
                       relative_l2_from_statistics,
                       relative_l2_statistics,
                       score_release,
                       score_support,
                       weighted_mae,
                       weighted_mse,
                       weighted_rmse,
)
from .package import (
                       Package,
                       ValidationReport,
                       build_package,
                       load_package,
                       to_table,
                       validate_package,
)
from .support import (
                       CaseSupport,
                       Component,
                       MetricBinding,
                       Quantity,
                       SupportDefinition,
                       SupportError,
                       SupportRelease,
                       UnsupportedLocationMode,
                       load_support_release,
                       write_support_release,
)
from .trust import HOLD, REVIEW, SUBMIT, CheckResult, SubmissionTrust, assess_submission

__all__ = [
    # contract
    "Checkout", "DatasetContract", "SplitSpec", "ScoreComposite", "ScoreComponent", "ContractError", "REPO_URL",
    "KNOWN_DATASETS", "DEFAULT_ENV_VAR",
    # support
    "load_support_release", "write_support_release", "SupportRelease", "SupportDefinition", "CaseSupport",
    "Quantity", "Component", "MetricBinding", "SupportError", "UnsupportedLocationMode",
    # mapping
    "map_to_support", "MappedPredictions", "Coverage", "CoverageError", "MappingError", "native_entity_counts",
    "summarize_mapping",
    # metrics
    "relative_l2", "relative_l2_statistics", "relative_l2_from_statistics", "relative_l1", "weighted_mae",
    "weighted_mse", "weighted_rmse", "r2", "reduction_value", "macro_average", "aggregate", "competition_rank",
    "score_support", "score_release", "SupportScore", "Statistics", "MetricError",
    # package
    "build_package", "validate_package", "load_package", "to_table", "Package", "ValidationReport",
    # trust
    "assess_submission", "SubmissionTrust", "CheckResult", "SUBMIT", "REVIEW", "HOLD",
]
