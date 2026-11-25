"""Minimal surrogate baseline with explicit training and validity checks."""
from math import isfinite

class NearestNeighborSurrogate:
    def fit(self, records):
        if not records:
            raise ValueError("Training data required")
        self.rows = []
        for row in records:
            p = row["parameters"]
            y = row["outputs"]["bottomhole_circulating_pressure_pa"]
            x = (p["mud_density_kg_m3"], p["flow_m3_s"])
            if not all(isfinite(v) for v in (*x,y)):
                raise ValueError("Nonfinite data")
            self.rows.append((x,y))
        self.bounds = [(min(x[i] for x,_ in self.rows),max(x[i] for x,_ in self.rows)) for i in range(2)]
        return self

    def predict(self, mud_density_kg_m3, flow_m3_s):
        x = (mud_density_kg_m3,flow_m3_s)
        if not hasattr(self, "rows"):
            raise RuntimeError("Fit model first")
        if any(not isfinite(v) for v in x):
            raise ValueError("Nonfinite input")
        if any(not lo <= v <= hi for v,(lo,hi) in zip(x,self.bounds)):
            return {"accepted":False,"reason":"outside_training_bounds","prediction_pa":None}
        scales = [max(hi-lo,1e-12) for lo,hi in self.bounds]
        best = min(self.rows,key=lambda r:sum(((v-r[0][i])/scales[i])**2 for i,v in enumerate(x)))
        return {"accepted":True,"prediction_pa":best[1],
                "verified":False,"method":"nearest_neighbor_baseline"}
