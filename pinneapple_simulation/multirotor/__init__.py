"""Multirotor (drone) models: rotor aerodynamics by momentum and blade-element momentum theory (:mod:`.rotor`), the
brushless motor and the LiPo pack (:mod:`.power`), the 6-DOF quadrotor with its LQR controller and the mission flown
on the battery (:mod:`.quad`), and the standard design workflow (:mod:`.workflow`, #1201).

    from pinneapple_simulation.multirotor.workflow import drone_workflow
    r = drone_workflow(payload_kg=0.3, endurance_min=15.0)       # sized, controlled, flown until the pack is empty
    r["mission"]["endurance_s"], r["checks"]
"""
