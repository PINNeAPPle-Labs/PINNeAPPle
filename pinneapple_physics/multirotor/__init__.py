"""Multirotor flight with physics-verified dynamics (#335).

The chain, in the order a flight runs it:

* :mod:`.rotor` -- the propeller, from blade element momentum theory: thrust, torque and the figure of merit;
* :mod:`.model` -- the 6-DoF airframe: four rotors, the pack that sags, drag, ground effect, motor lag and the mixer;
* :mod:`.control` -- the cascade that flies it, which is never told about a fault;
* :mod:`.faults` -- the eight faults, injected where they physically act;
* :mod:`.residuals` -- the sensors, and Newton-Euler on the nominal vehicle against them;
* :mod:`.detect` -- three detectors tuned to the same false-alarm budget, and the diagnosis that reads the direction;
* :mod:`.sim` -- one flight, and the campaign over faults and seeds.
"""
from __future__ import annotations

from .control import Controller, Trajectory, quaternion_error
from .detect import (
                     Calibration,
                     CusumDetector,
                     Detection,
                     KalmanDetector,
                     ThresholdDetector,
                     calibrate,
                     diagnose,
                     tune,
)
from .faults import FAULTS, Fault, FaultState, apply_faults, chip_factors
from .model import QUAD_X, Airframe, Battery, Multirotor, RotorTable, State, Telemetry
from .residuals import CHANNELS, Residual, ResidualModel, SensorModel, SensorObservatory, Sensors
from .rotor import PROPS, Rotor, RotorState, figure_of_merit
from .sim import Flight, campaign, fly

__all__ = ["Rotor", "RotorState", "PROPS", "figure_of_merit", "Airframe", "Battery", "Multirotor", "RotorTable",
           "State", "Telemetry", "QUAD_X", "Controller", "Trajectory", "quaternion_error", "Fault", "FaultState",
           "FAULTS", "apply_faults", "chip_factors", "Sensors", "SensorModel", "SensorObservatory", "Residual",
           "ResidualModel", "CHANNELS", "Detection", "Calibration", "ThresholdDetector", "KalmanDetector",
           "CusumDetector", "calibrate", "tune", "diagnose", "Flight", "fly", "campaign"]
