"""Planner capabilities for the example flows of nodes that live elsewhere: the industrial presets
(:mod:`pinneapple_flow.nodes.presets`), the science experiments (:mod:`pinneapple_flow.nodes.science`) and the
autonomous-lab missions (:mod:`pinneapple_flow.nodes.autolab`, ``rover.learn_planet``)."""
from __future__ import annotations


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    caps = [
        ("presets.water_hammer", "Water hammer of a valve closure (industrial preset)", "engineering presets",
         "industrial_presets", {"water hammer": 4, "valve closure": 3, "surge": 1.5, "pipeline transient": 2},
         ("preset.run", "preset.validate", "examples.compare"), ("white2011fm",),
         "industrial_presets/water_hammer_valve_closure"),
        ("science.em_experiment", "An electromagnetism experiment against its cited reference (pp.science)", "science",
         "science", {"faraday": 2.5, "ampere": 2.5, "gauss law": 2.5, "biot-savart": 1.5, "pp.science": 3,
                     "electromagnetism experiment": 3, "induction": 1},
         ("science.run", "examples.compare"), (), "science/faraday_induction"),
        ("autolab.board_cooling", "Self-optimizing engineer: cool a board with the lightest heat sink", "design",
         "autolab", {"board cooling": 4, "heat sink": 2, "junction temperature": 2, "soc": 1, "self-optimizing": 3},
         ("autolab.board_cooling", "examples.flatten", "examples.compare"), ("incropera2007",),
         "autolab/self_optimizing_engineer"),
        ("autolab.pipe_twin", "A digital twin that knows when it is wrong (pipe loop)", "digital twin",
         "autolab_digital_twin", {"knows when it is wrong": 4, "pipe loop": 3, "fault diagnosis": 2.5,
                                  "sensor fault": 2, "digital twin": 1},
         ("autolab.pipe_twin", "examples.flatten", "examples.compare"), ("kalman1960filter",),
         "autolab_digital_twin/twin_knows_when_wrong"),
        ("autolab.rover", "A rover that learns the planet's soil and replans", "vehicles", "autolab_rover",
         {"learns the planet": 4, "rover": 1.5, "soil": 1.5, "replan": 2, "slip": 1},
         ("rover.learn_planet", "examples.compare"), (), "autolab_rover/rover_learns_planet"),
        ("autolab.aerodynamicist", "The AI Aerodynamicist: lower drag of a race car with OpenFOAM", "aerodynamics",
         "autolab_aero", {"aerodynamicist": 4, "race car": 3, "racecar": 3, "downforce": 2, "lower drag": 2},
         ("autolab.aerodynamicist", "examples.compare"), ("menter1994sst",), "autolab_aero/autolab_aero"),
    ]
    for cid, title, domain, example, kw, nodes, refs, template in caps:
        register_capability(Capability(
            id=cid, title=title, domain=domain, example=example, keywords=kw, min_score=3, nodes=nodes,
            references=refs, uncited="" if refs else "the study cites its references in its own result files",
            template=template, quick=True, description=f"examples/{example}"))


_register_capabilities()
