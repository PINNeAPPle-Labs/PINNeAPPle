# Builds channel_obstacle.FCStd, the parametric FreeCAD project of the channel_obstacle workflow.
#   freecadcmd build_project.py            (writes the file next to this script)
# A rectangular channel (400 mm long in x, 50 mm wide in y, 100 mm high in z) with a rectangular obstacle on its floor spanning the width; every
# dimension is an alias of the Params spreadsheet and the fluid domain is the channel minus the obstacle.
import os

import FreeCAD as App

here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
doc = App.newDocument("channel_obstacle")
sheet = doc.addObject("Spreadsheet::Sheet", "Params")
rows = [("length", 400, "channel length (flow direction x)"), ("height", 100, "channel height (z, vertical)"),
        ("width", 50, "channel width (y)"), ("obstacle_x", 120, "distance from the inlet to the obstacle"),
        ("obstacle_length", 40, "obstacle length along x"), ("obstacle_height", 40, "obstacle height (z), from the floor")]
for i, (alias, value, note) in enumerate(rows, 1):
    sheet.set(f"A{i}", alias)
    sheet.set(f"B{i}", f"{value} mm")
    sheet.setAlias(f"B{i}", alias)
    sheet.set(f"C{i}", note)
doc.recompute()
channel = doc.addObject("Part::Box", "Channel")
for prop, alias in (("Length", "length"), ("Width", "width"), ("Height", "height")):
    channel.setExpression(prop, f"Params.{alias}")
obstacle = doc.addObject("Part::Box", "Obstacle")
for prop, alias in (("Length", "obstacle_length"), ("Width", "width"), ("Height", "obstacle_height")):
    obstacle.setExpression(prop, f"Params.{alias}")
obstacle.setExpression(".Placement.Base.x", "Params.obstacle_x")
domain = doc.addObject("Part::Cut", "FluidDomain")
domain.Base, domain.Tool = channel, obstacle
doc.recompute()
assert domain.Shape.isValid() and abs(domain.Shape.Volume - (400 * 100 * 50 - 40 * 40 * 50)) < 1e-6
doc.saveAs(os.path.join(here, "channel_obstacle.FCStd"))
