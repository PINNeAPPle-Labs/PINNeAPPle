# Runs under pvpython: pvpython paraview_export.py job.json
# Opens an OpenFOAM case at its last time, exports a cell field (cell centres + values) to CSV and renders the
# field on a slice to PNG. Writes the job's "result" JSON with what was exported.
import json
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import (
    CellCenters,
    ColorBy,
    GetActiveViewOrCreate,
    GetAnimationScene,
    GetDisplayProperties,
    GetScalarBar,
    MergeBlocks,
    OpenFOAMReader,
    Render,
    SaveData,
    SaveScreenshot,
    Show,
    Slice,
    UpdatePipeline,
)

job = json.load(open(sys.argv[1], encoding="utf-8"))
reader = OpenFOAMReader(FileName=job["foam_file"])
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = job.get("arrays", ["U", "p"])
reader.UpdatePipelineInformation()
times = list(reader.TimestepValues) if reader.TimestepValues else [0.0]
t = times[-1]
GetAnimationScene().AnimationTime = t
UpdatePipeline(time=t, proxy=reader)
merged = MergeBlocks(Input=reader)
UpdatePipeline(time=t, proxy=merged)

data = servermanager.Fetch(merged)
n_cells = data.GetNumberOfCells()
field = job.get("field", "U")
arr = data.GetCellData().GetArray(field)
rng = [arr.GetRange(-1)[0], arr.GetRange(-1)[1]] if arr is not None else None
bounds = list(data.GetBounds())

centres = CellCenters(Input=merged)
SaveData(job["csv"], proxy=centres, PointDataArrays=job.get("arrays", ["U", "p"]), Precision=8)

axis = job.get("slice_normal", [0, 0, 1])
origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
sl = Slice(Input=merged)
sl.SliceType = "Plane"
sl.SliceType.Origin = origin
sl.SliceType.Normal = axis
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = job.get("size", [1400, 520])
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
disp = Show(sl, view)
ColorBy(disp, ("CELLS", field, "Magnitude"))
disp.RescaleTransferFunctionToDataRange(True, False)
disp.SetScalarBarVisibility(view, True)
bar = GetScalarBar(GetDisplayProperties(sl, view).LookupTable, view)
bar.Title = job.get("bar_title", field)
bar.ComponentTitle = "magnitude"
bar.TitleColor = bar.LabelColor = [0.1, 0.1, 0.1]
Render(view)                                     # the first render resets the camera; set it after

up = np.asarray(job.get("view_up", [0, 1, 0]), float)
nrm = np.asarray(axis, float)
right = np.cross(up, nrm)
ext = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
h, w = float(np.abs(up) @ ext), float(np.abs(right) @ ext)
size = job.get("size", [1400, 520])
view.CameraParallelProjection = 1
view.CameraFocalPoint = origin
view.CameraPosition = [origin[i] - 10 * float(ext.max()) * axis[i] for i in range(3)]
view.CameraViewUp = list(up)
view.CameraParallelScale = 0.55 * max(h, w * size[1] / size[0] / 0.8)   # leave room for the colour bar
bar.WindowLocation = "Any Location"
bar.Orientation = "Vertical"
bar.Position = [0.89, 0.12]
bar.ScalarBarLength = 0.76
Render(view)
SaveScreenshot(job["png"], view, ImageResolution=job.get("size", [1400, 520]))

json.dump({"time": t, "times": times, "cells": n_cells, "field": field, "range": rng, "bounds": bounds,
           "slice_origin": origin, "slice_normal": axis}, open(job["result"], "w"), indent=1)
