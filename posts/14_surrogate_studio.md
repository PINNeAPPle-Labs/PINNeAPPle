Day 14 of 30: Surrogate Studio

Every new design point usually means another CFD or FEA run. A surrogate model can answer in microseconds, but only if someone picks the method fairly, tests it on data it never trained on, and states where it stops being valid.

Surrogate Studio takes a table of simulation runs and trains every applicable method on the same splits with the same compute budget. Methods that do not fit the data are listed with the reason. The validation error picks the model; the test and shift sets never do. The chosen model is then served at a versioned endpoint.

In the video I use an OpenFOAM airfoil campaign (six shape weights and the angle of attack, predicting Cl and Cd), with whole airfoils held out. The MLP was chosen. Its error was 2.58 % on held-out runs and 2.96 % on airfoils it never saw. Polynomial ridge scored 8.29 % and gradient-boosted trees 10.4 %. Inference took 81.6 µs per query.

What sets it apart is that the model says where it can be trusted. The trust report checks the held-out error, R², and the ratio of shift error to held-out error (1.15, limit 3), and returns APPROVED. The endpoint enforces the training range on every query: at 21° angle of attack, outside the trained range of -2.0° to 13.4°, the prediction comes back flagged as an extrapolation. The downloads include the report, the shared data contracts, and ONNX and TorchScript exports.

Live: https://surrogate.pinneapple.org
GitHub: https://github.com/pinneapple-labs/pinneapple (apps/surrogate_studio)

#SurrogateModeling #CFD #MachineLearning #ScientificML #SimulationEngineering #MLOps #OpenFOAM
