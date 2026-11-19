"""KIT metadata declares decimal-hour time; synthetic fixtures are unit tests only."""
import numpy as np
import pandas as pd
from pinneapple_lab.experiments.real_world_thermal import load_real

def test_kit_decimal_hour_clock(tmp_path):
    n=400
    p=tmp_path/"kit.csv"
    pd.DataFrame({
        "time": 100.0 + np.arange(n)/60,
        "b3_r1_tzone___c_": 20+np.sin(np.arange(n)/50),
        "tamb___c_": np.full(n,5.0),
        "b3_r1_qth__w_": np.full(n,500.0),
    }).to_csv(p,index=False)
    mapping={"time":"time","temperature":"b3_r1_tzone___c_",
             "outdoor":"tamb___c_","heat_w":"b3_r1_qth__w_"}
    frame,dt,checksum=load_real(p,mapping)
    assert len(frame)==n
    assert abs(dt-60)<.01
    assert len(checksum)==64
