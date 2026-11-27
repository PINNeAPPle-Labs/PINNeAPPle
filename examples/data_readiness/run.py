"""The data-readiness gate, end to end (#1491, #1493): assess, refuse, review, train, and say who accepted what.

Run it with ``python examples/data_readiness/run.py`` from the repository root. It needs three seconds and no network:
a tiny linear model stands in for the real one, because the subject here is the gate and not the model.

The three states are shown in the order a project meets them:

1. a dataset whose operating point is outside the declared validity -- **FAIL**, and the run never starts;
2. a dataset with one warning (a column with no unit) -- **WARNING**, refused until an engineer records that they
   accept that specific finding, for that specific dataset digest;
3. with the record in place, the run starts and carries the reviewer into its own provenance.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import pinneapple as pp
from pinneapple_train.trainer import Trainer, TrainConfig

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def assess(manifest, table):
    report = pp.readiness(manifest, table=table)
    print(f"\n{report.to_markdown()}")
    return report


def main() -> int:
    table = pd.read_csv(HERE / "sweep.csv")
    print(f"dataset: {MANIFEST['name']} v{MANIFEST['version']}, {len(table)} cases, columns {list(table.columns)}")

    # 1. the operating point the model would be asked about is outside what the data covers
    outside = {**MANIFEST, "operating_point": {"Re": 300000.0}}
    refused = assess(outside, table)
    try:
        pp.require_readiness(refused)
    except pp.ReadinessError as exc:
        print(f"\nrefused before training: {str(exc).splitlines()[0]}")
        for finding in exc.findings:
            print(f"  {finding.id}: {finding.detail} fix: {finding.fix}")

    # 2. one warning: the unit of the output is declared nowhere
    warning_manifest = {**MANIFEST, "columns": [{"name": "Re", "role": "input", "unit": "1"},
                                                {"name": "T_max", "role": "output"}]}
    warned = assess(warning_manifest, table)
    store_path = pathlib.Path(tempfile.mkdtemp()) / "reviews.jsonl"
    store = pp.ReviewStore(store_path)
    try:
        pp.require_readiness(warned, review=store)
    except pp.ReadinessError as exc:
        print(f"\nrefused: {str(exc).splitlines()[0]}")

    # 3. an engineer looks at the evidence and accepts that finding, for this digest
    record = store.save(pp.ReviewRecord(
        subject=warned.digest, decision="accept_exception", reviewer="A. Ribeiro", role="thermal",
        notes="the case sheet states K; the column header does not", evidence_seen=[warned.digest],
        scope=[{"finding": "units.missing", "reason": "unit stated in the case sheet, not in the CSV header"}]))
    print(f"\nreview {record.id[:19]}... by {record.reviewer} ({record.role}): accepts {record.overridden()}")

    # 4. and now the run starts, carrying the verdict and the reviewer into its own record
    model = torch.nn.Linear(1, 1)
    # the input is scaled by its own magnitude: Re is 1e4-1e5 and an unscaled linear layer would only show how big the
    # numbers are, not whether the gate works
    re_values = table[["Re [1]"]].to_numpy()
    scaled = (re_values - re_values.mean()) / re_values.std()
    loader = DataLoader(TensorDataset(torch.tensor(scaled, dtype=torch.float32),
                                      torch.tensor(table[["T_max [K]"]].to_numpy(), dtype=torch.float32)),
                        batch_size=3)
    trainer = Trainer(model=model, loss_fn=lambda m, y_hat, batch: torch.nn.functional.mse_loss(y_hat, batch["y"]),
                      readiness=warned, review=store)
    run = trainer.fit(loader, TrainConfig(epochs=100, device="cpu", log_dir=str(HERE / "_out"), run_name="gate",
                                          save_best=False))
    print("\nthe run started, and its provenance says:")
    print(json.dumps(run["readiness"], indent=1, default=str))
    print(f"\nloss {run['history'][0]['train_total']:.4f} -> {run['history'][-1]['train_total']:.4f}")
    print("\nThe gate is the point: an approval is a record, tied to one digest of one dataset, by one person.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
