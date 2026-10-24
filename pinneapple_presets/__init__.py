"""Industrial presets: engineering problems already set up, solved and validated with PINNeAPPle.

    import pinneapple as pp
    pp.presets.names()                                   # catalogue
    p = pp.presets.load("water_hammer_valve_closure")
    p.card()                                             # data sheet: inputs (SI, defaults with sources), outputs
    r = p.run(t_close=4.0)                               # solver; fidelity="surrogate" for the trained surrogate
    print(r)                                             # outputs, decision, trust report
    p.validate()                                         # reference cases with the measured error

CLI: ``python -m pinneapple_presets list | card NAME | run NAME k=v ... | validate NAME | report NAME``.
Flow Studio node: ``preset.run``. The Hub reads ``pinneapple_presets/index.json`` (:func:`write_index`).
"""
from .core import Output, Param, Preset, Reference, Result, SurrogateSpec, check
from .registry import index, list_presets, load, names, register, write_index

__all__ = ["Preset", "Param", "Output", "Reference", "Result", "SurrogateSpec", "check", "load", "names",
           "list_presets", "register", "index", "write_index"]
