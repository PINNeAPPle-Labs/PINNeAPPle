"""Safe numeric-only export of the actual native scalar surrogate models.

No pickle is deserialized by this module. Only the trusted in-memory models
returned by the native training workflow can be exported. Unsupported families
fail closed; serialized data contains no executable Python objects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FORMAT = "pinneapple.native-scalar-json.v1"


def _array(value):
    a = np.asarray(value, dtype=np.float64)
    if a.size > 2_000_000 or not np.isfinite(a).all():
        raise ValueError("oversized or nonfinite native model array")
    return a.tolist()


def _scalar_model(model, family):
    if family == "quadratic":
        return {"family": family, "weights": _array(model.w), "sigma": float(model.sigma)}
    if family == "mlp_ensemble":
        return {"family": family, "mu": float(model.mu), "sd": float(model.sd),
                "members": [{"coefs": [_array(c) for c in m.coefs_],
                             "intercepts": [_array(b) for b in m.intercepts_],
                             "activation": m.activation,
                             "out_activation": m.out_activation_}
                            for m in model.members]}
    if family == "gp":
        gp = model.gp
        # Native kernel: Constant * Matern(nu=2.5, ARD) + WhiteKernel.
        k = gp.kernel_
        if not (k.k1.k2.nu == 2.5 and gp.normalize_y):
            raise ValueError("unsupported GP kernel")
        return {"family": family, "train_x": _array(gp.X_train_),
                "alpha": _array(gp.alpha_), "cholesky": _array(gp.L_),
                "constant": float(k.k1.k1.constant_value),
                "length_scales": _array(np.atleast_1d(k.k1.k2.length_scale)),
                "noise": float(k.k2.noise_level),
                "mean": float(np.asarray(gp._y_train_mean).item()),
                "scale": float(np.asarray(gp._y_train_std).item())}
    raise ValueError("unsupported native surrogate family")


def export_native_scalar(models: dict, families: dict, q: dict, inputs: list[str],
                         outputs: list[str], bounds: dict, path: str) -> str:
    if len(inputs)>32 or len(outputs)>32 or not inputs or not outputs:
        raise ValueError("invalid model dimensions")
    if set(models)!=set(outputs) or set(families)!=set(outputs) or set(q)!=set(outputs):
        raise ValueError("incomplete selected native models")
    payload={"format": FORMAT, "inputs": inputs, "outputs": outputs,
             "bounds": {k: _array(bounds[k]) for k in inputs},
             "models": {o: _scalar_model(models[o], families[o]) for o in outputs},
             "q": {o: float(q[o]) for o in outputs}}
    data=json.dumps(payload,allow_nan=False,separators=(",",":")).encode()
    if len(data)>32*1024*1024:
        raise ValueError("native model exceeds export limit")
    Path(path).write_bytes(data)
    return path


def predict_native_scalar(payload: dict, rows: list[dict]) -> dict:
    if payload.get("format")!=FORMAT:
        raise ValueError("unsupported native model format")
    inputs=payload["inputs"]
    outputs=payload["outputs"]
    if not 1<=len(rows)<=256 or not 1<=len(inputs)<=32 or not 1<=len(outputs)<=32:
        raise ValueError("invalid inference batch")
    if any(set(r)!=set(inputs) for r in rows):
        raise ValueError("inference input fields mismatch")
    x=np.asarray([[r[k] for k in inputs] for r in rows],dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite inference input")
    bounds=np.asarray([payload["bounds"][k] for k in inputs],dtype=float)
    if bounds.shape!=(len(inputs),2) or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError("invalid applicability envelope")
    if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):
        raise ValueError("prediction outside native model training domain")
    u=(x-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    result={}
    for name in outputs:
        m=payload["models"][name]
        family=m["family"]
        if family=="quadratic":
            basis=np.column_stack([np.ones(len(u)),*u.T,
                 *(u[:,i]*u[:,j] for i in range(u.shape[1]) for j in range(i,u.shape[1]))])
            mean=basis@np.asarray(m["weights"])
            std=np.full(len(u),m["sigma"])
        elif family=="mlp_ensemble":
            members=[]
            for member in m["members"]:
                if member["activation"]!="tanh" or member["out_activation"]!="identity":
                    raise ValueError("unsupported native MLP activation")
                z=u
                for index,(w,b) in enumerate(zip(member["coefs"],member["intercepts"])):
                    z=z@np.asarray(w)+np.asarray(b)
                    if index<len(member["coefs"])-1:
                        z=np.tanh(z)
                members.append(z.reshape(-1))
            predictions=np.stack(members)*m["sd"]+m["mu"]
            mean=predictions.mean(0)
            std=predictions.std(0)
        elif family=="gp":
            from scipy.spatial.distance import cdist
            train=np.asarray(m["train_x"])
            scale=np.asarray(m["length_scales"])
            if scale.size==1:
                scale=np.repeat(scale,u.shape[1])
            dist=cdist(u/scale,train/scale)
            sqrt5=np.sqrt(5)*dist
            kernel=m["constant"]*(1+sqrt5+5*dist*dist/3)*np.exp(-sqrt5)
            mean=(kernel@np.asarray(m["alpha"]))*m["scale"]+m["mean"]
            from scipy.linalg import solve_triangular
            v=solve_triangular(np.asarray(m["cholesky"]),kernel.T,lower=True)
            variance=np.maximum(m["constant"]+m["noise"]-np.sum(v*v,axis=0),0)
            std=np.sqrt(variance)*m["scale"]
        else:
            raise ValueError("unsupported native model family")
        half=np.asarray(std)*float(payload["q"][name])
        if not np.isfinite(mean).all() or not np.isfinite(half).all():
            raise ValueError("nonfinite native inference result")
        result[name]={"mean":mean.tolist(),"half_width90":half.tolist(),"family":family}
    return result
