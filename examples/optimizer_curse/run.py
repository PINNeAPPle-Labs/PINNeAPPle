"""Synthetic, reference-blind optimizer's curse experiment (NOT real CFD)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import RBFInterpolator
from pinneapple_veriphysics.design_harness import DesignVerificationHarness

BASE=np.array([.5,.5,.5])
BOUNDS=np.array([[0.,1.]]*3)

def reference(x):
    x=np.asarray(x,float)
    return (1.-.9*np.sum((x-np.array([.72,.36,.64]))**2,axis=-1)
            +.09*np.sin(13*x[...,0])*np.cos(11*x[...,1])
            +.045*np.sin(16*x[...,2]+4*x[...,0]))

def lhs(n,d,rng):
    x=np.empty((n,d))
    for j in range(d): x[:,j]=(rng.permutation(n)+rng.random(n))/n
    return x

def one(seed,train_size,budget,threshold,out):
    rng=np.random.default_rng(seed)
    train=lhs(train_size,3,rng)
    test=lhs(128,3,rng)
    model=RBFInterpolator(train,reference(train),kernel="thin_plate_spline",smoothing=.025)
    p0=float(model(BASE[None,:])[0])
    f0=float(reference(BASE))
    calibration=lhs(72,3,rng)
    error=float(np.quantile(np.abs(model(calibration)-reference(calibration)),.95))
    candidates=lhs(budget,3,rng)
    predictions=model(candidates)
    rank=np.argsort(-predictions)
    baseline=rank[:min(16,budget)]
    h=DesignVerificationHarness(BOUNDS,train,p0,threshold,
                                ood_distance=.42,uncertainty_multiplier=.15)
    screened=[]
    escalated=0
    for i in rank:
        if len(screened)>=len(baseline): break
        decision=h.check(candidates[i],float(predictions[i]),error,
                         check_id=f"seed{seed}/candidate{int(i)}/screen")
        if decision.status=="provisional": screened.append(int(i))
        elif decision.status=="escalate": escalated+=1
    # FREEZE selection before running either final reference adjudication.
    f_base=reference(candidates[baseline])
    f_screen=reference(candidates[screened]) if screened else np.array([])
    for i in screened:
        h.check(candidates[i],float(predictions[i]),error,reference=reference,
                reference_baseline=f0,check_id=f"seed{seed}/candidate{int(i)}/reference")
    def confirmation(ids,truth):
        if not len(ids): return None
        promised=predictions[ids]-p0>=threshold
        if not np.any(promised): return None
        return float(np.mean((truth-f0)[promised]>=threshold))
    out.mkdir(parents=True,exist_ok=True)
    (out/f"seed_{seed}.json").write_text(json.dumps(h.ledger.to_dict(),indent=2))
    return {"seed":seed,"reference":"manufactured_synthetic_not_CFD",
            "train_size":train_size,"candidate_budget":budget,
            "reference_calls_selection":0,
            "reference_calls_post_selection":len(baseline)+len(screened),
            "escalated_without_reference":escalated,
            "baseline_indices":[int(i) for i in baseline],
            "screened_indices":screened,
            "baseline_confirmation_rate":confirmation(baseline,f_base),
            "screened_confirmation_rate":confirmation(screened,f_screen),
            "baseline_best_actual_gain":float(max(f_base-f0)),
            "screened_best_actual_gain":float(max(f_screen-f0)) if len(f_screen) else None,
            "test_mae":float(np.mean(abs(model(test)-reference(test)))),
            "nominal_calibration_error_95":error,
            "blind_coverage_95":float(np.mean(abs(model(test)-reference(test))<=error))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("/tmp/pinneapple-optimizer-curse"))
    p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--budget",type=int,default=512)
    p.add_argument("--train",type=int,default=48)
    p.add_argument("--threshold",type=float,default=.005)
    a=p.parse_args()
    if a.seeds<1 or a.budget<16 or a.train<8 or a.threshold<0:p.error("invalid budget")
    a.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    rows=[one(s,a.train,a.budget,a.threshold,a.out/"evidence") for s in range(a.seeds)]
    result={"experiment":"optimizer_curse_synthetic_pilot",
            "warning":"NOT CFD; 43%/88%/+37% not replicated",
            "seconds":time.perf_counter()-start,"runs":rows}
    (a.out/"results.json").write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    labels=["baseline","screened"]
    vals=[]
    for key in labels:
        observed=[r[f"{key}_confirmation_rate"] for r in rows if r[f"{key}_confirmation_rate"] is not None]
        vals.append(float(np.mean(observed)) if observed else np.nan)
    ax.bar(labels,vals)
    ax.set_ylim(0,1)
    ax.set_ylabel("Independently confirmed predicted improvements")
    ax.set_title("Synthetic experiment only; not CFD")
    fig.tight_layout()
    fig.savefig(a.out/"confirmation.png",dpi=160)
    plt.close(fig)
    report=("# PINNeAPPle optimizer's curse pilot\n\n"
            "Manufactured synthetic objective; NOT CFD or wind-tunnel validation.\n\n"
            f"Seeds: {a.seeds}; candidates per seed: {a.budget}; train size: {a.train}.\n\n"
            f"Baseline average confirmation: {vals[0]:.1%}.\n\n"
            f"Screened average confirmation: {vals[1]:.1%}.\n\n"
            "Reference used only AFTER both selections freeze; screening multiplier "
            "is a heuristic, NOT calibrated prediction-interval coverage. "
            "Local ledger is tamper-evident, not externally immutable.\n")
    (a.out/"REPORT.md").write_text(report)
    print(report)
if __name__=="__main__": main()
