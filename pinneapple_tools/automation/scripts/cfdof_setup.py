# Runs inside the FreeCAD GUI (CfdOF needs FreeCADGui), with ARGS / RESULT from freecad_gui_job.py.
# ruff: noqa: F821  (FreeCAD, ARGS and RESULT are provided by freecad_gui_job.py)
# Builds a CfdOF analysis on a solid through CfdOF's Python API and writes the gmsh mesh case and the OpenFOAM case.
# ARGS: document, solid, output_dir, foam_dir, gmsh, inlet_velocity_mm_s, mesh_size_mm, max_iterations,
#       convergence_tol, fluid {name, density, viscosity}, inlet ("x_min"...), outlet ("x_max"...)
import os

from CfdOF import CfdAnalysis, CfdTools
from CfdOF.Mesh import CfdMesh, CfdMeshTools
from CfdOF.Solve import (
    CfdCaseWriterFoam,
    CfdFluidBoundary,
    CfdFluidMaterial,
    CfdInitialiseFlowField,
    CfdPhysicsSelection,
    CfdSolverFoam,
)

prefs = CfdTools.getPreferencesLocation()
FreeCAD.ParamGet(prefs).SetBool("AppendDocNameToOutputPath", False)
FreeCAD.ParamGet(prefs).SetString("InstallationPath", ARGS["foam_dir"])
FreeCAD.ParamGet(prefs).SetString("GmshPath", ARGS["gmsh"])

doc = FreeCAD.openDocument(ARGS["document"])
doc.recompute()
part = doc.getObject(ARGS["solid"])
shape = part.Shape
bb = shape.BoundBox
planes = {"x_min": (0, bb.XMin), "x_max": (0, bb.XMax), "y_min": (1, bb.YMin), "y_max": (1, bb.YMax),
          "z_min": (2, bb.ZMin), "z_max": (2, bb.ZMax)}


def on_plane(face, plane):
    axis, value = planes[plane]
    c, n = face.CenterOfMass, face.normalAt(0, 0)
    return abs(c[axis] - value) < 1e-6 * max(bb.DiagonalLength, 1.0) and abs(n[axis]) > 0.999


groups = {"inlet": [], "outlet": [], "walls": []}
for i, f in enumerate(shape.Faces, 1):
    key = "inlet" if on_plane(f, ARGS["inlet"]) else "outlet" if on_plane(f, ARGS["outlet"]) else "walls"
    groups[key].append(f"Face{i}")
if not groups["inlet"] or not groups["outlet"]:
    raise RuntimeError(f"no face on the inlet plane {ARGS['inlet']} or the outlet plane {ARGS['outlet']}: {groups}")
RESULT["faces"] = groups
RESULT["inlet_area_mm2"] = sum(shape.Faces[int(n[4:]) - 1].Area for n in groups["inlet"])
RESULT["outlet_area_mm2"] = sum(shape.Faces[int(n[4:]) - 1].Area for n in groups["outlet"])

analysis = CfdAnalysis.makeCfdAnalysis("CfdAnalysis")
CfdTools.setActiveAnalysis(analysis)
analysis.OutputPath = ARGS["output_dir"]

phys = CfdPhysicsSelection.makeCfdPhysicsSelection()
analysis.addObject(phys)
phys.Time, phys.Flow, phys.Turbulence = "Steady", "Isothermal", "Laminar"

fluid = ARGS["fluid"]
mat = CfdFluidMaterial.makeCfdFluidMaterial("FluidProperties")
analysis.addObject(mat)
m = dict(mat.Material)
m.update({"Name": fluid["name"], "Density": f"{fluid['density']} kg/m^3",
          "DynamicViscosity": f"{fluid['viscosity']} kg/m/s"})
mat.Material = m

u = f"{ARGS['inlet_velocity_mm_s']} mm/s"
init = CfdInitialiseFlowField.makeCfdInitialFlowField()
analysis.addObject(init)
# no potential-flow initialisation: with OpenFOAM v1912, potentialFoam -initialiseUBCs wrote outlet velocities of
# hundreds of m/s on this case and simpleFoam stopped with a floating-point exception; a uniform start converges
init.PotentialFlow = False
init.PotentialFlowP = False
init.UseInletUValues = False
init.Ux, init.Uy, init.Uz = u, "0 mm/s", "0 mm/s"
init.UseOutletPValue = False
init.Pressure = "0 Pa"

solver = CfdSolverFoam.makeCfdSolverFoam()
analysis.addObject(solver)
solver.MaxIterations = int(ARGS["max_iterations"])
solver.ConvergenceTol = float(ARGS["convergence_tol"])
solver.Parallel = False

mesh = CfdMesh.makeCfdMesh("FluidMesh")
analysis.addObject(mesh)
mesh.Part = part
mesh.MeshUtility = "gmsh"
mesh.ElementDimension = "3D"
mesh.CharacteristicLengthMax = f"{ARGS['mesh_size_mm']} mm"

for name, kind, sub in (("inlet", "inlet", "uniformVelocityInlet"), ("outlet", "outlet", "staticPressureOutlet"),
                        ("walls", "wall", "fixedWall")):
    bc = CfdFluidBoundary.makeCfdFluidBoundary(name)
    analysis.addObject(bc)
    bc.BoundaryType, bc.BoundarySubType = kind, sub
    if name == "inlet":
        bc.VelocityIsCartesian = True
        bc.Ux, bc.Uy, bc.Uz = u, "0 mm/s", "0 mm/s"
    if name == "outlet":
        bc.Pressure = "0 Pa"
    bc.ShapeRefs = [(part, tuple(groups[name]))]
doc.recompute()

mesh_tools = CfdMeshTools.CfdMeshTools(mesh)
mesh_tools.writeMesh()
writer = CfdCaseWriterFoam.CfdCaseWriterFoam(analysis)
writer.writeCase()
RESULT["mesh_case"] = mesh_tools.mesh_case_dir
RESULT["case"] = writer.case_folder
out_doc = os.path.join(ARGS["output_dir"], "cfd_setup.FCStd")
doc.saveAs(out_doc)
RESULT["document"] = out_doc
