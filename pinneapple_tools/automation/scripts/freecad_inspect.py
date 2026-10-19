# Runs inside FreeCAD (ARGS / RESULT are provided by freecad_gui_job.py): open a document and describe it.
# ruff: noqa: F821  (FreeCAD, ARGS and RESULT are provided by freecad_gui_job.py)
# ARGS: document (path), solid (name of the object to measure, optional)
doc = FreeCAD.openDocument(ARGS["document"])
doc.recompute()
params = {}
for sheet in [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]:
    for cell in sorted(sheet.getUsedCells()):
        alias = sheet.getAlias(cell)
        if alias:
            v = sheet.get(alias)
            params[alias] = {"value": float(getattr(v, "Value", v)), "unit": str(getattr(v, "Unit", "")),
                             "text": str(getattr(v, "UserString", v)), "sheet": sheet.Name, "cell": cell}
RESULT["document"] = doc.FileName
RESULT["label"] = doc.Label
RESULT["objects"] = [{"name": o.Name, "type": o.TypeId, "label": o.Label} for o in doc.Objects]
RESULT["parameters"] = params
invalid = [o.Name for o in doc.Objects if "Invalid" in o.State or "Error" in o.State]
RESULT["invalid_objects"] = invalid
solid = ARGS.get("solid")
if solid:
    shape = doc.getObject(solid).Shape
    RESULT["solid"] = {"name": solid, "volume_mm3": shape.Volume, "faces": len(shape.Faces),
                       "valid": shape.isValid(), "bound_box_mm": [shape.BoundBox.XMin, shape.BoundBox.YMin,
                                                                   shape.BoundBox.ZMin, shape.BoundBox.XMax,
                                                                   shape.BoundBox.YMax, shape.BoundBox.ZMax]}
