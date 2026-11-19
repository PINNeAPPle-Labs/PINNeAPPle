"""Fail-closed physical data, holdout and VeriPhysics publication gates."""
import importlib
import json
from pathlib import Path
import numpy as np

def data_audit(records):
    errors=[]
    for record in records:
        if record["status"]!="ok":
            errors.append(f"{record['id']}: {record.get('status')} {record.get('error','')}")
            continue
        try:
            with np.load(record["path"],allow_pickle=False) as a:
                x=a["x"];y=a["y"];coords=a["coords"]
                if x.ndim!=2 or y.ndim!=2 or coords.ndim!=2 or len(x)!=len(y) or len(x)!=len(coords):
                    errors.append(f"{record['id']}: inconsistent dataset shape")
                if not all(np.isfinite(v).all() for v in (x,y,coords)):
                    errors.append(f"{record['id']}: nonfinite dataset")
        except Exception as exc:errors.append(f"{record['id']}: {type(exc).__name__}: {exc}")
    return dict(passed=not errors,errors=errors,checks=["solver_exit","dataset_shape","dataset_finite"])

def model_audit(metrics,gates):
    limit=float(gates.get("max_relative_l2",1.0))
    val=metrics["val"]["relative_l2"];test=metrics["test"]["relative_l2"]
    passed=bool(np.isfinite([val,test]).all() and val<=limit and test<=limit)
    return dict(passed=passed,threshold=limit,val_relative_l2=val,test_relative_l2=test)

def veriphysics_audit(spec,root,records,metrics):
    if not spec:return dict(status="not_configured",passed=False,reason="No VeriPhysics adapter configured")
    module,sep,func=spec.partition(":")
    if not sep:raise ValueError("VeriPhysics adapter must be module:function")
    fn=getattr(importlib.import_module(module),func)
    result=fn(Path(root),records,metrics)
    if not isinstance(result,dict) or not isinstance(result.get("passed"),bool):
        raise TypeError("VeriPhysics adapter must return dict with boolean passed")
    return dict(status="executed",**result)

def report(root,records,metrics,config):
    data=data_audit(records);model=model_audit(metrics,config.gates)
    try:\n        adapter=config.gates.get("veriphysics_adapter")\n        if adapter:\n            veriphysics=veriphysics_audit(adapter,root,records,metrics)\n        else:\n            from .native_validation import evaluate\n            veriphysics=evaluate(root,records,metrics,config)
    except Exception as exc:
        veriphysics=dict(status="error",passed=False,reason=f"{type(exc).__name__}: {exc}")
    required=bool(config.gates.get("require_veriphysics",True))
    accepted=bool(data["passed"] and model["passed"] and (veriphysics["passed"] if required else True))
    result=dict(publishable=accepted,data=data,model=model,veriphysics=veriphysics,require_veriphysics=required)
    (Path(root)/"validation.json").write_text(json.dumps(result,indent=2,default=str))
    return result
