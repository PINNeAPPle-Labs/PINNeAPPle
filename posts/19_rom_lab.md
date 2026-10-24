Day 19 of 30: Reduced Order Model Lab

A finite-element run takes under a second here, but design exploration, optimisation and control need thousands of them. A reduced-order model replays a simulation in milliseconds or less. The questions: how much error does it cost, and was it checked on cases it never saw?

The Reduced Order Model Lab takes field snapshots (OpenFOAM, VTU/VTK, CalculiX, NPZ/HDF5, CSV) and builds POD + RBF, POD + GPR, DMD, Operator Inference and a neural surrogate at equal time budget. Every model is scored on held-out snapshots, next to the POD projection floor and a baseline it must beat.

In the video: 100 solid-FEM cantilevers (height, width, load angle), 20 designs held out at random. POD + GPR at rank 8 reaches 4.40 % mean error on the unseen designs (max 26.2 %), at 4.0 ms per design, 181x faster than the 718.5 ms FEM. The nearest training design gives 38.7 %, POD + RBF 17.1 %.

The 3-D view shows |ROM - FEM| of the von Mises stress on the deformed body: 2.89 % error for that design, with 31 % of the squared error in the clamp zone, on 10 % of the nodes. Checks: 3 of 3 pass (9.2x better than the nearest design, within the 5 % tolerance, nothing diverged). The limits are stated: no physics check on the reduced fields, valid only inside the training box. Models download as .rom.npz, ONNX and a ModelRef.

Live: https://rom.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/rom_lab)

#ReducedOrderModeling #ROM #FEA #Simulation #MachineLearning #ComputationalMechanics #SurrogateModels
