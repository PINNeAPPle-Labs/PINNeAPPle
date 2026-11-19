"""Reference-blind, model-independent design validation using the existing evidence ledger."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .ledger import EvidenceLedger, sha256_json

@dataclass(frozen=True)
class DesignDecision:
    status: str
    reasons: tuple
    predicted_improvement: float
    lower_improvement: float
    distance: float
    reference_improvement: float | None = None
    def to_dict(self):
        return asdict(self)

class DesignVerificationHarness:
    """Provisional acceptance never represents independent physical verification."""
    def __init__(self,bounds,train_points,baseline_prediction,threshold=0.005,
                 ood_distance=0.42,uncertainty_multiplier=0.15,ledger=None):
        self.bounds=np.asarray(bounds,dtype=float)
        self.train=np.asarray(train_points,dtype=float)
        if self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or np.any(self.bounds[:,1]<=self.bounds[:,0]):
            raise ValueError("invalid bounds")
        if self.train.ndim!=2 or self.train.shape[1]!=len(self.bounds) or not len(self.train):
            raise ValueError("invalid training support")
        self.baseline=float(baseline_prediction)
        self.threshold=float(threshold)
        self.ood=float(ood_distance)
        self.multiplier=float(uncertainty_multiplier)
        self.ledger=ledger if ledger is not None else EvidenceLedger()

    def check(self,candidate,prediction,error_scale,*,physics_check=None,reference=None,
              reference_baseline=None,check_id=None):
        x=np.asarray(candidate,dtype=float)
        if x.shape!=(len(self.bounds),) or not np.all(np.isfinite(x)):
            raise ValueError("invalid candidate")
        if not math.isfinite(prediction) or not math.isfinite(error_scale) or error_scale<0:
            raise ValueError("invalid prediction or uncertainty")
        width=self.bounds[:,1]-self.bounds[:,0]
        distance=float(np.min(np.linalg.norm((self.train-x)/width,axis=1)))
        valid=bool(np.all(x>=self.bounds[:,0]) and np.all(x<=self.bounds[:,1]))
        physics_ok=physics_check(x) if physics_check is not None else None
        predicted=float(prediction-self.baseline)
        lower=predicted-self.multiplier*error_scale
        reasons=[]
        if not valid: reasons.append("hard_constraint_violation")
        if physics_ok is False: reasons.append("physics_check_failed")
        if distance>self.ood: reasons.append("out_of_distribution")
        if lower<self.threshold: reasons.append("insufficient_conservative_improvement")
        status="reject" if not valid or physics_ok is False else ("escalate" if reasons else "provisional")
        actual=None
        if reference is not None and status!="reject":
            if reference_baseline is None: raise ValueError("missing reference baseline")
            actual=float(reference(x)-reference_baseline)
            if not math.isfinite(actual): raise ValueError("nonfinite reference")
            status="verified" if actual>=self.threshold else "reject"
            if status=="reject": reasons.append("reference_failed")
        result=DesignDecision(status,tuple(reasons),predicted,lower,distance,actual)
        if check_id is not None:
            info={"x":x.tolist(),"prediction":float(prediction),"uncertainty":float(error_scale),
                  "bounds":self.bounds.tolist(),"threshold":self.threshold,
                  "ood":self.ood,"multiplier":self.multiplier}
            with self.ledger.run(check_id,inputs_sha256=sha256_json(info),
                                 tool="pinneapple_veriphysics.design_harness") as r:
                r.outputs=result.to_dict()
                r.passed=(status=="verified")
        return result
