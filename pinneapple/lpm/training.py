"""Sample-disjoint surrogate training, checkpointing and held-out metrics."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from .models import make_model

def _load(records,split):
    arrays=[]
    for record in records:
        if record["status"]=="ok" and record["split"]==split:
            with np.load(record["path"],allow_pickle=False) as d:
                arrays.append((d["x"].astype("float32"),d["y"].astype("float32")))
    if not arrays:raise ValueError(f"No {split} samples available")
    if len({x.shape for x,y in arrays})!=1 or len({y.shape for x,y in arrays})!=1:
        raise ValueError("This reference trainer requires identical topology; remap variable meshes first")
    return np.stack([x for x,y in arrays]),np.stack([y for x,y in arrays])

def _metrics(pred,y):
    err=pred-y
    return dict(rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
                relative_l2=float(np.linalg.norm(err)/(np.linalg.norm(y)+1e-12)),
                max_abs=float(np.max(abs(err))))

def fit(records,config,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)
    tx,ty=_load(records,"train");vx,vy=_load(records,"val");sx,sy=_load(records,"test")
    opts=config.training;kind=opts.get("model","mlp")
    if kind=="fno" and config.domain!="cfd":raise ValueError("2D FNO requires CFD")
    shape=(int(config.numerical["ny"]),int(config.numerical["nx"])) if kind=="fno" else ((int(config.numerical["ny"]),int(config.numerical["nx"])) if config.domain=="cfd" else (int(config.numerical["nz"])+1,int(config.numerical["ny"])+1,int(config.numerical["nx"])+1)) if kind=="graph" else None
    if shape and tx.shape[1]!=int(np.prod(shape)):raise ValueError("Structured grid size mismatch")
    im=tx.reshape(-1,tx.shape[-1]).astype("float64").mean(0)
    isd=tx.reshape(-1,tx.shape[-1]).astype("float64").std(0)
    isd=np.where(isd<1e-12,1.,isd)
    om=ty.reshape(-1,ty.shape[-1]).astype("float64").mean(0)
    osd=ty.reshape(-1,ty.shape[-1]).astype("float64").std(0)
    osd=np.where(osd<1e-20,1.,osd)
    def transform(x,y):
        return torch.from_numpy(((x-im)/isd).astype("float32")),torch.from_numpy(((y-om)/osd).astype("float32"))
    t_x,t_y=transform(tx,ty);v_x,v_y=transform(vx,vy);s_x,s_y=transform(sx,sy)
    device=torch.device(opts.get("device","cpu"))
    model=make_model(kind,tx.shape[-1],ty.shape[-1],opts).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(opts.get("learning_rate",1e-3)),weight_decay=float(opts.get("weight_decay",1e-5)))
    epochs=int(opts.get("epochs",100));batch=int(opts.get("batch_size",4));patience=int(opts.get("patience",30))
    if min(epochs,batch,patience)<1:raise ValueError("epochs, batch_size, patience must be positive")
    best=float("inf");best_epoch=0;bad=0;history=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for ids in torch.randperm(len(t_x)).split(batch):
            x=t_x[ids].to(device);y=t_y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.mean((model(x,shape)-y)**2)
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():val=torch.mean((model(v_x.to(device),shape)-v_y.to(device))**2).item()
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=val))
        if val<best-1e-9:
            best=val;best_epoch=epoch+1;bad=0
            torch.save(dict(state_dict=model.state_dict(),model=kind,in_features=tx.shape[-1],
                out_features=ty.shape[-1],options=opts,grid_shape=shape,input_mean=im,
                input_scale=isd,output_mean=om,output_scale=osd,source_config=config.to_dict()),
                outdir/"model.pt")
        else:
            bad+=1
            if bad>=patience:break
    # Only load a checkpoint created locally during this invocation.
    state=torch.load(outdir/"model.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["state_dict"]);model.eval()
    def evaluate(x,y):
        with torch.no_grad():pred=model(x.to(device),shape).cpu().numpy()*osd+om
        return _metrics(pred,y)
    metrics=dict(train=evaluate(t_x,ty),val=evaluate(v_x,vy),test=evaluate(s_x,sy),
                 best_epoch=best_epoch,total_epochs=len(history),model=kind,
                 samples=dict(train=len(tx),val=len(vx),test=len(sx)))
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (outdir/"history.json").write_text(json.dumps(history,indent=2))
    return metrics

def predict(model_path,x,device="cpu"):
    """Load only trusted locally generated checkpoints (torch.load uses pickle)."""
    state=torch.load(model_path,map_location=device,weights_only=True)
    model=make_model(state["model"],state["in_features"],state["out_features"],state["options"])
    model.load_state_dict(state["state_dict"]);model.to(device).eval()
    arr=np.asarray(x)
    if not np.isfinite(arr).all():raise ValueError("Nonfinite input")
    xx=(torch.as_tensor(arr,dtype=torch.float32,device=device)-state["input_mean"].to(device)) / state["input_scale"].to(device)\n    xx=xx.float()
    if xx.ndim==2:xx=xx[None]
    with torch.no_grad():out=model(xx,state["grid_shape"]).cpu().numpy()*state["output_scale"].cpu().numpy()+state["output_mean"].cpu().numpy()
    return out
