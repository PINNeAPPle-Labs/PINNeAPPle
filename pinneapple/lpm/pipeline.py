"""Restartable end-to-end solver-to-surrogate LPM campaigns."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import shutil
import traceback
import numpy as np
from .schema import Campaign
from .sampling import designs,assign_splits,fingerprint
from . import openfoam,calculix,fenics
from .native_training import fit_native
from .gates import report

BACKENDS={"openfoam":openfoam,"calculix":calculix,"fenics":fenics}

def orchestrate(config,root,stage="all",resume=True):
    c=Campaign.load(config) if not isinstance(config,Campaign) else config
    c.validate()
    if stage not in {"prepare","simulate","train","all"}:raise ValueError("Unknown stage")
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    digest=fingerprint(c.to_dict());path=root/"manifest.json"
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest["config_hash"]!=digest:raise ValueError("Existing campaign has a different configuration")
    else:
        samples=designs(c);splits=assign_splits(c.samples,c.split,c.seed)
        records=[dict(id=f"{i:05d}",design=d,split=splits[i],status="pending",
                      path=str(root/"cases"/f"{i:05d}"/"sample.npz")) for i,d in enumerate(samples)]
        manifest=dict(name=c.name,config_hash=digest,config=c.to_dict(),
                      python=platform.python_version(),created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      records=records)
        path.write_text(json.dumps(manifest,indent=2))
    records=manifest["records"];backend=BACKENDS[c.solver]
    if stage in {"prepare","simulate","all"}:
        for record in records:
            if resume and record["status"]=="ok" and Path(record["path"]).is_file():continue
            work=root/"cases"/record["id"];work.mkdir(parents=True,exist_ok=True)
            try:
                result=backend.run(work,record["design"],c,execute=stage!="prepare")
                if stage!="prepare":
                    np.savez_compressed(record["path"],x=result["x"],y=result["y"],coords=result["coords"])
                    record.update(status="ok",meta=result["meta"],time=result.get("time"))
                else:record.update(status="prepared",meta=result)
                record.pop("error",None)
            except Exception as exc:
                record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                (work/"error.log").write_text(traceback.format_exc())
            path.write_text(json.dumps(manifest,indent=2,default=str))
        if stage=="prepare":return manifest
    if stage in {"train","all"}:
        incomplete=[r for r in records if r["status"]!="ok" or not Path(r["path"]).is_file()]
        if incomplete:raise RuntimeError(f"{len(incomplete)} simulations missing; refusing incomplete campaign")
        metrics=fit_native(records,c,root/"training")
        review=report(root,records,metrics,c)
        registry=root/"deployment";registry.mkdir(exist_ok=True)
        card=dict(model=c.training.get("model","mlp"),campaign=c.name,
                  dataset_hash=fingerprint([{k:v for k,v in r.items() if k!="path"} for r in records]),
                  metrics=metrics,validation=review,status="approved" if review["publishable"] else "blocked")
        (registry/"model_card.json").write_text(json.dumps(card,indent=2))
        if review["publishable"]:
            shutil.copy2(root/"training"/"model.pt",registry/"approved_model.pt")
        else:
            (registry/"approved_model.pt").unlink(missing_ok=True)
    return manifest

def main():
    parser=argparse.ArgumentParser(description="PINNeAPPLe LPM campaign")
    parser.add_argument("config")
    parser.add_argument("--out",required=True)
    parser.add_argument("--stage",choices=["prepare","simulate","train","all"],default="all")
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    orchestrate(args.config,args.out,args.stage,not args.no_resume)

if __name__=="__main__":main()
