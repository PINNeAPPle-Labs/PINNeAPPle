"""The evidence behind #77: a recovered parameter, its interval, and the coverage that interval was measured to have.

Three questions, in the order a discovery report asks them:

1. **Did we get it back?** The cooling-curve preset (:func:`pinneapple_analysis.inverse_problems.presets.get_preset`,
   ``"cooling_h"``): a 450 J/K aluminium part cooling in still air, thermocouple at ±0.3 °C, unknown ``h``, start and
   air temperature. The data were generated at known values, so the truth is available to check against.
2. **Is that interval calibrated?** The same experiment repeated ``--n`` times at the truth with fresh noise, counting
   how often the 95 % interval contains it — and the same count for the profile-likelihood interval.
3. **What does it look like when it is not?** Two failures a coverage study is meant to catch: data with a bias the
   model cannot represent (the interval is confidently wrong) and two parameters the data only see as a ratio
   (``h/C``: the interval is infinite rather than confident).

    $ python scripts/identifiability_recovery.py            # the three reports
    $ python scripts/identifiability_recovery.py --n 400 --markdown /tmp/recovery.md

Exit code 1 when the well-specified coverage is outside the tolerance: a study whose job is to detect a miscalibrated
interval should not report success when it finds one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinneapple_analysis.inverse_problems.forward import ParameterSet  # noqa: E402
from pinneapple_analysis.inverse_problems.presets import get_preset, synthetic  # noqa: E402
from pinneapple_analysis.inverse_problems.recovery import coverage_study, recover  # noqa: E402

PRESET = "cooling_h"
BIAS = 2.0          # °C: a thermocouple reading that is off by two degrees


def cooling_case(rng: np.random.Generator):
    """One synthetic repetition of the cooling experiment: the preset, its noise redrawn at the truth."""
    built = get_preset(PRESET).build()
    params = ParameterSet(built["parameters"])
    return built["model"], params, synthetic(built["model"], params.all, built["observations"],
                                             int(rng.integers(2**31 - 1)))


def biased_case(rng: np.random.Generator):
    """The same experiment with a 2 °C offset: the model has no term for it, so every prediction is off by that much."""
    model, params, obs = cooling_case(rng)
    return model, params, obs.with_values(obs.values + BIAS)


def ambiguous_case():
    """The same curve with the heat capacity unknown too: the data can only see ``h/C``."""
    built = get_preset("cooling_h_C").build()
    params = ParameterSet(built["parameters"])
    return built["model"], params, built["observations"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="synthetic repetitions for the coverage study")
    ap.add_argument("--profile-n", type=int, default=30, help="repetitions for the profile interval (it refits)")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--markdown", type=Path, default=None, help="write the tables to this file")
    args = ap.parse_args(argv)

    built = get_preset(PRESET).build()
    params = ParameterSet(built["parameters"])
    truth = params.truth()
    result = recover(built["model"], params, built["observations"], level=args.level)

    print(f"## 1. Recovered from the measured cooling curve ({', '.join(f'{k} = {v:g}' for k, v in truth.items())})")
    print(result, "\n")
    print("truth inside every interval:", result.contains(truth), "\n")

    model, ambiguous_params, ambiguous_obs = ambiguous_case()
    ambiguous = recover(model, ambiguous_params, ambiguous_obs, level=args.level)
    print("## 2. When the data cannot separate two parameters")
    print(ambiguous.to_markdown(), "\n")

    linear = coverage_study(cooling_case, n=args.n, level=args.level, seed=args.seed)
    print(f"## 3. Coverage of the {args.level:.0%} interval, {args.n} synthetic repetitions")
    print(linear.to_markdown(), "\n")

    profile = coverage_study(cooling_case, n=args.profile_n, level=args.level, seed=args.seed, interval="profile")
    print(f"### The same counts for the profile-likelihood interval ({args.profile_n} repetitions)")
    print(profile.to_markdown(), "\n")

    biased = coverage_study(biased_case, n=max(40, args.n // 2), level=args.level, seed=args.seed)
    print(f"### A 2 °C bias the model cannot represent ({max(40, args.n // 2)} repetitions)")
    print(biased.to_markdown(), "\n")

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            "# Recovered parameters and their measured intervals\n\n"
            f"Truth of the cooling case: {', '.join(f'`{k}` = {v:g}' for k, v in truth.items())}.\n\n"
            "## The recovery\n\n" + result.to_markdown() + "\n\n"
            "## What the data cannot separate\n\n" + ambiguous.to_markdown() + "\n\n"
            f"## Coverage, {args.n} synthetic repetitions\n\n" + linear.to_markdown() + "\n\n"
            f"## Profile-likelihood interval, {args.profile_n} repetitions\n\n" + profile.to_markdown() + "\n\n"
            f"## A model missing a real effect ({max(40, args.n // 2)} repetitions)\n\n" + biased.to_markdown() + "\n",
            encoding="utf-8")
        print(f"written to {args.markdown}")

    if not linear.ok:
        print(f"the well-specified case is not calibrated ({linear.calibrations})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
