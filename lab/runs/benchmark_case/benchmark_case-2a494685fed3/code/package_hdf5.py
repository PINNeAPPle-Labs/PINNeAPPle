"""Consolidate the OpenRadioss DoE into one HDF5 file (same schema as the reproduced write-up):

training_data.hdf5
  inputs/Exp_i/{nodes [N,3], elements [E,4] (-1 padded), part_ids [E], node_ids [N],
                thickness_dv1, thickness_dv2}
  outputs/Exp_i/{displacement_field [N, T, 3], node_history [T] (u_x at node 1806), time [T]}
  attrs: reference node id/index, solver energy error per run (from engine.log)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))
from pinneapple_simulation.external_solvers.openradioss import read_case_displacements, read_vtk_mesh  # noqa

REF_NODE = 1806


def energy_error(case: Path):
    """Largest |ERR| (energy balance error, %) printed by the engine during the run."""
    errs = [float(x) for x in re.findall(r"ERR=\s*([-\d.]+)%", (case / "engine.log").read_text())]
    return max(abs(e) for e in errs) if errs else float("nan")


def main():
    design = {r["exp"]: r for r in csv.DictReader(open(ROOT / "data" / "design_table.csv"))}
    runs = sorted([p for p in (ROOT / "runs").glob("Exp_*") if (p / "done.flag").exists()],
                  key=lambda p: int(p.name.split("_")[1]))
    mesh = read_vtk_mesh(runs[0] / "frames" / "frame_001.vtk")
    ref_idx = int(np.where(mesh.node_ids == REF_NODE)[0][0])
    elems = -np.ones((len(mesh.cells), 4), np.int64)
    for k, c in enumerate(mesh.cells):
        elems[k, :len(c)] = c
    with h5py.File(ROOT / "data" / "training_data.hdf5", "w") as f:
        f.attrs["reference_node_id"] = REF_NODE
        f.attrs["reference_node_index"] = ref_idx
        f.attrs["reference_node_xyz"] = mesh.points[ref_idx]
        for case in runs:
            e = case.name
            m = read_vtk_mesh(case / "frames" / "frame_001.vtk")
            assert np.array_equal(m.node_ids, mesh.node_ids), f"{e}: node ordering differs"
            t, disp = read_case_displacements(case / "frames")
            gi = f.create_group(f"inputs/{e}")
            gi["nodes"] = m.points
            gi["elements"] = elems
            gi["part_ids"] = mesh.part_ids
            gi["node_ids"] = mesh.node_ids
            gi["thickness_dv1"] = float(design[e]["dv1_dp1000_mm"])
            gi["thickness_dv2"] = float(design[e]["dv2_dp600_mm"])
            go = f.create_group(f"outputs/{e}")
            go.create_dataset("displacement_field", data=disp.transpose(1, 0, 2), compression="gzip", compression_opts=4)
            go["node_history"] = disp[:, ref_idx, 0]
            go["time"] = t
            go.attrs["energy_error_pct_max"] = energy_error(case)
    print("packaged", len(runs), "runs")


if __name__ == "__main__":
    main()
