"""``python -m pinneapple_simulation.pipelines.airfrans_benchmark`` — run the campaign from a JSON spec.

    python -m pinneapple_simulation.pipelines.airfrans_benchmark prepare  spec.json work/dir
    python -m pinneapple_simulation.pipelines.airfrans_benchmark train    spec.json work/dir
    python -m pinneapple_simulation.pipelines.airfrans_benchmark predict  spec.json work/dir
    python -m pinneapple_simulation.pipelines.airfrans_benchmark submit   spec.json work/dir
    python -m pinneapple_simulation.pipelines.airfrans_benchmark all      spec.json work/dir

The spec is a JSON object with ``dataset_root``, ``support_root``, ``checkout``, ``submission_id``, ``split_id`` and the
optional fields of :class:`~.campaign.CampaignSpec` and :class:`~.train.TrainingSpec` (``training`` object).
"""
from __future__ import annotations

import argparse
import json

from .campaign import STAGES, campaign_from_dict, run_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pinneapple_simulation.pipelines.airfrans_benchmark",
                                     description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("spec", help="JSON spec of the campaign")
    parser.add_argument("workdir", help="where the store, the runs and the entry live")
    args = parser.parse_args(argv)

    with open(args.spec) as handle:
        payload = json.load(handle)
    submission_id = payload.pop("submission_id", None)
    split_id = payload.pop("split_id", "full")
    if not submission_id:
        raise SystemExit("the spec needs a submission_id (the directory name of the result package)")
    spec = campaign_from_dict(payload)
    stages = STAGES if args.stage == "all" else (args.stage,)
    result = {
        # every stage is resumable: an existing store, run or entry is reused rather than overwritten
    }
    result = run_campaign(spec, args.workdir, submission_id=submission_id, split_id=split_id, stages=stages)
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
