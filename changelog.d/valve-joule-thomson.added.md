`pinneapple_systems.process_components.joule_thomson`: Joule–Thomson cooling of natural gas across a globe valve. It computes:
- the outlet temperature from GERG-2008, Peng–Robinson, SRK and the ideal gas;
- the isenthalpic path and μ_JT, with the flash checked against the μ_JT integral;
- the valve flow per IEC 60534-2-1, with a choked-flow check;
- the coldest point inside the jet.

It can also check a CFD outlet temperature, with a diagnosis: no cooling points to an ideal-gas equation of state. A PT/EN text parser reads the case, and the flow node `process.valve_joule_thomson` comes with a capability (#1463). The figure texts follow the language of the request (Portuguese or English).
