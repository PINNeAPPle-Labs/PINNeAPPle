"""3-D multiphysics study of an interior-PM motor: reference model, pretrained surrogate, optimisation without FEA.

The pieces, in the order a study uses them:

* :mod:`.design` -- the design vector (:data:`~.design.VARIABLES`, :class:`~.design.Motor3DDesign`).
* :mod:`.thermal3d` -- the axial-resolved lumped thermal network (the stack is not isothermal: it has two cooled
  ends and a hot middle, and the winding conducts along the copper).
* :mod:`.model` -- :class:`~.model.Motor3DModel`: electromagnetic FEA, skew by axial slices, iron, magnet and copper
  loss, cooling at a coolant flow, the magnet temperature back on torque and demagnetisation, and the axial profiles.
* :mod:`.dataset` -- design of experiments over :data:`~.design.VARIABLES`, evaluated by the reference model and
  stored with its provenance.
* :mod:`.surrogate` -- :class:`~.surrogate.MotorSurrogate3D`, an ensemble of networks trained on that set with its
  held-out errors, validity box and model card, and :func:`~.surrogate.load_pretrained` for the one shipped here.
* :mod:`.optimize` -- the design problem and the search that runs **entirely on the surrogate** (no FEA per
  candidate), the Pareto front, and the re-check of the front with the reference model.
* :mod:`.workflow` -- the whole study as one command, with figures and a report.

The headline use is three lines::

    from pinneapple_simulation.electromagnetics.machines.motor3d import load_pretrained, optimize
    surrogate = load_pretrained()
    front = optimize(budget=4000)["front"]        # 4000 designs, zero FEA calls

and the workflow that produced the surrogate is one more::

    python -m pinneapple_simulation.electromagnetics.machines.motor3d.workflow all --out motor3d_run
"""
from .dataset import MotorDataset, build_dataset, parameter_space
from .design import BOUNDS, NAMES, VARIABLES, DesignVariable, Motor3DDesign, from_matrix, to_matrix
from .model import OUTPUTS, PRIUS, PROFILES, SCALARS, Motor3DModel, MotorOutput
from .optimize import benchmark, design_problem, optimize, verify_front
from .surrogate import PRETRAINED_VERSION, MotorSurrogate3D, SurrogateConfig, load_pretrained
from .thermal3d import AxialThermalModel, axial_network

__all__ = ["BOUNDS", "NAMES", "VARIABLES", "DesignVariable", "Motor3DDesign", "from_matrix", "to_matrix",
           "OUTPUTS", "PROFILES", "SCALARS", "Motor3DModel", "MotorOutput", "PRIUS",
           "AxialThermalModel", "axial_network",
           "MotorDataset", "build_dataset", "parameter_space",
           "MotorSurrogate3D", "SurrogateConfig", "load_pretrained", "PRETRAINED_VERSION",
           "design_problem", "optimize", "verify_front", "benchmark"]
