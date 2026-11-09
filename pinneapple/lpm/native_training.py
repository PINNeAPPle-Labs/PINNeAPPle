"""Use the existing PINNeAPPLe Trainer for solver-surrogate learning."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

from pinneapple_neural.trainer.trainer import Trainer,TrainConfig
from .models import make_model
from .training import _load,_metrics


def fit_native(records,config,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    train_x,train_y=_load(records,"train")
    val_x,val_y=_load(records,"val")
    test_x,test_y=_load(records,"test")
    kind=config.training.get("model","mlp")
    if kind=="fno" and config.domain!="cfd":
        raise ValueError("2D FNO only supports CFD")
    shape=(int(config.numerical["ny"]),int(config.numerical["nx"])) if config.domain=="cfd" else (
        int(config.numerical["nz"])+1,int(config.numerical["ny"])+1,int(config.numerical["nx"])+1)
    if train_x.shape[1]!=int(np.prod(shape)):
        raise ValueError("Sample nodes do not match the campaign mesh topology")
    im=train_x.reshape(-1,train_x.shape[-1]).astype("float64").mean(0)
    isd=train_x.reshape(-1,train_x.shape[-1]).astype("float64").std(0)
    isd=np.where(isd<1e-12,1.,isd)
    om=train_y.reshape(-1,train_y.shape[-1]).astype("float64").mean(0)
    osd=train_y.reshape(-1,train_y.shape[-1]).astype("float64").std(0)
    osd=np.where(osd<1e-20,1.,osd)
    def loader(x,y,shuffle):
        xx=torch.from_numpy(((x-im)/isd).astype("float32"))
        yy=torch.from_numpy(((y-om)/osd).astype("float32"))
        return DataLoader(TensorDataset(xx,yy),batch_size=int(config.training.get("batch_size",4)),shuffle=shuffle)
    opts=config.training
    core=make_model(kind,train_x.shape[-1],train_y.shape[-1],opts)
    class LayoutAdapter(torch.nn.Module):
        def __init__(self,model,grid):
            super().__init__();self.model=model;self.grid=grid
        def forward(self,x):
            return self.model(x,self.grid if kind in ("fno","graph") else None)
    wrapper=LayoutAdapter(core,shape)
    trainer=Trainer(wrapper,loss_fn=lambda model,pred,batch: torch.nn.functional.mse_loss(pred,batch["y"]),metrics=[])
    result=trainer.fit(
        loader(train_x,train_y,True),
        loader(val_x,val_y,False),
        TrainConfig(epochs=int(opts.get("epochs",100)),lr=float(opts.get("learning_rate",.001)),
                    weight_decay=float(opts.get("weight_decay",1e-5)),
                    grad_clip=1.,seed=config.seed,device=str(opts.get("device","cpu")),
                    log_dir=str(outdir/"native_logs"),run_name="lpm",save_best=True),
    )
    best_path=result.get("best_path")
    if not best_path:
        raise RuntimeError("Native Trainer did not produce a best validation checkpoint")
    # Native trainer checkpoint format is owned by PINNeAPPLe; restore the
    # selected state through the native checkpoint reader before exporting.
    from pinneapple_neural.trainer.checkpoint import load_checkpoint
    restored=load_checkpoint(best_path,model=wrapper)
    wrapper.eval().to("cpu")
    state=dict(state_dict=wrapper.model.state_dict(),model=kind,
        in_features=train_x.shape[-1],out_features=train_y.shape[-1],options=opts,
        grid_shape=shape if kind in ("fno","graph") else None,
        input_mean=im,input_scale=isd,output_mean=om,output_scale=osd,
        source_config=config.to_dict())
    torch.save(state,outdir/"model.pt")
    def evaluate(x,y):
        with torch.no_grad():
            xx=torch.from_numpy(((x-im)/isd).astype("float32"))
            pred=wrapper(xx).numpy()*osd+om
        return _metrics(pred,y)
    metrics=dict(train=evaluate(train_x,train_y),val=evaluate(val_x,val_y),
                 test=evaluate(test_x,test_y),model=kind,
                 samples=dict(train=len(train_x),val=len(val_x),test=len(test_x)),
                 native_trainer=True,training_run=result)
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str))
    return metrics
