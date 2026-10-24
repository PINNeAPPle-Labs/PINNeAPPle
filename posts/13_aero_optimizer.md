Day 13 of 30: Aircraft Design Optimizer

An airliner wing is a negotiation. A larger span cuts induced drag but has to fit a 36 m gate. More sweep allows a faster cruise but makes the approach faster too. Fuel has to fit in the wing, the aircraft has to stay stable, and the stall has to start at the root.

The Aircraft Design Optimizer searches 13 numbers for a single-aisle airliner: the wing section (from a CFD-trained graph network and MLP ensemble), area, aspect ratio, taper, sweep, twist, wing position and cruise Mach. NSGA-II looks for less CO2 per passenger-km, a faster cruise and a slower approach.

In the video, the default mission (180 passengers, 4800 km) flies 1,920 airliners in 96 s: 920 are certifiable and 151 sit on the Pareto front. The balanced pick burns 7.5 % less CO2 than the A320 class at Mach 0.798. Every rejected design keeps its reason.

The surrogate's answer does not get the last word. The picks were run in OpenFOAM 3-D (0.44 M cells, k-omega SST, 70 m/s, 4 deg). For the balanced pick, OpenFOAM gives a wing CL of 0.555 against 0.605 from the vortex lattice, and 0.640 for the whole aircraft, because the lattice leaves out the lift of the fuselage. Both methods rank the five aircraft the same way. All 8 requirements pass; you can inspect the skin pressure, the streamlines and the symmetry-plane slices in the browser.

GitHub: https://github.com/pinneapple-labs/pinneapple (apps/aero_optimizer)

#AircraftDesign #CFD #OpenFOAM #Optimization #SurrogateModels #GraphNeuralNetworks #Aerospace #PINNeAPPle
