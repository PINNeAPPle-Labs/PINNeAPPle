"""Research orchestration for KIT: repeated seeds, cross-campaign tests, plots.

Run after the public measurements have been downloaded. The numerical RC and
one-step neural models are scored with observed lagged temperature (teacher
forcing); autonomous RC rollout is reported separately. No fabricated data.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from pinneapple_lab.experiments.real_world_thermal import (
    load_real, rc_baseline, score, train_network,
)
from pinneapple_lab.experiments.kit_continuous_thermal import train_continuous_pinn


def prepare(path, mapping):
    frame, dt, sha = load_real(path, mapping)
    t = frame["temperature"].to_numpy(dtype=float)
    outdoor = frame["outdoor"].to_numpy(dtype=float)
    heat_w = frame["heat_w"].to_numpy(dtype=float)
    x = np.column_stack((t[:-1], outdoor[:-1], heat_w[:-1]/1000))
    return {"frame":frame,"dt":dt,"sha256":sha,"temperature":t,
            "ambient":outdoor,"heat_w":heat_w,"x":x,"y":t[1:]}


def rc_predict(x, coefficients):
    a,b,c = coefficients
    return x[:,0] + a*(x[:,1]-x[:,0]) + b*x[:,2] + c


def rc_rollout(initial_temp, ambient, heat_kw, coefficients):
    """Autonomous RC simulation: never feeds observed future temperatures."""
    a,b,c = coefficients
    out = np.empty(len(ambient)+1)
    out[0] = initial_temp
    for i in range(len(ambient)):
        out[i+1] = out[i]+a*(ambient[i]-out[i])+b*heat_kw[i]+c
    return out


def run(train_csv, test_csv, mapping, outdir, seeds, epochs, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True,exist_ok=True)
    training = prepare(train_csv,mapping)
    testing = prepare(test_csv,mapping)
    # Entire test campaign is held out. Validation is the last 20% of train
    # campaign; calibration and supervised updates use only first 80%.
    n = len(training["y"])
    cut = int(.8*n)
    split = {"train":np.arange(cut), "val":np.arange(cut,n)}
    coefficients, _ = rc_baseline(training["x"],training["y"],split["train"])
    observed = testing["y"]
    preds = {"rc_one_step":rc_predict(testing["x"],coefficients)}
    roll = rc_rollout(testing["temperature"][0],testing["ambient"][:-1],
                      testing["heat_w"][:-1]/1000,coefficients)
    preds["rc_autonomous_rollout"] = roll[1:]
    rows = []
    predictions = {"observed":observed}
    for name, p in preds.items():
        predictions[name]=p
        rows.append({"model":name,"seed":None,**score(observed,p)})
    for seed in seeds:
        for name, physics in (("data_only",False),("rc_regularized",True)):
            prior_train = rc_predict(training["x"],coefficients)
            _,history,_ = train_network(training["x"],training["y"],split,
                                        prior_train,physics,epochs,seed,device)
            # Train_network returns predictions on the supplied x. To prevent
            # test-campaign leakage, a transferable model API is required.
            # Explicitly mark as not evaluated rather than reporting in-sample scores.
            rows.append({"model":name,"seed":seed,"status":"transferable_predictor_pending",
                         "validation_epochs":len(history)})
        # The continuous model is currently trained on the training campaign;
        # its current API does not expose a transferable model checkpoint.
        _,hist,physical = train_continuous_pinn(
            training["frame"]["time"].to_numpy(dtype=float),
            training["ambient"],training["heat_w"],training["temperature"],
            np.arange(cut+1),np.arange(cut+1,len(training["temperature"])),
            seed=seed,epochs=epochs,device=device)
        rows.append({"model":"continuous_ode_pinn","seed":seed,
                     "status":"cross_campaign_inference_pending",
                     "validation_epochs":len(hist),
                     "physical_parameters":physical})
    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir/"metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(outdir/"predictions.csv",index=False)
    provenance = {
        "train_file":str(train_csv),"test_file":str(test_csv),
        "train_sha256":training["sha256"],"test_sha256":testing["sha256"],
        "train_rows":len(training["temperature"]),
        "test_rows":len(testing["temperature"]),
        "split":"first 80% train campaign for fitting, final 20% validation; separate held-out campaign",
        "seeds":seeds,"epochs":epochs,"model_status":"incomplete cross-campaign neural inference",
        "limitations":["RC rollout is autonomous; RC one-step uses observed lag",
                       "Neural model test predictions not yet available; no claim of superiority"],
    }
    (outdir/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(observed,label="observed",linewidth=1)
    ax.plot(preds["rc_one_step"],label="RC one-step",alpha=.8)
    ax.plot(preds["rc_autonomous_rollout"],label="RC autonomous",alpha=.8)
    ax.set(xlabel="Held-out test step",ylabel="Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/"cross_campaign_rc.png",dpi=160)
    plt.close(fig)
    (outdir/"README.md").write_text(
        "# KIT cross-campaign benchmark\n\n"
        "This is a partial experimental report. Neural cross-campaign inference "
        "is not yet implemented; missing values must not be interpreted as failures "
        "or successes. See metrics.csv and provenance.json.\n")
    return provenance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",required=True,type=Path)
    parser.add_argument("--test",required=True,type=Path)
    parser.add_argument("--mapping",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("reports/kit_cross_campaign"))
    parser.add_argument("--seeds",default="0,1,2,3,4")
    parser.add_argument("--epochs",type=int,default=400)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if args.train.resolve()==args.test.resolve():
        parser.error("train and test campaigns must be different files")
    seeds=[int(s) for s in args.seeds.split(",")]
    run(args.train,args.test,json.loads(args.mapping.read_text()),args.output,
        seeds,args.epochs,args.device)


if __name__=="__main__":
    main()
