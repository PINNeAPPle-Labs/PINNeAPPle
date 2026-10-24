Day 16 of 30: Engineering Sensitivity Lab

Calibrations, optimisations and test campaigns often carry ten or more parameters, and most of them barely matter. Teams usually find out by changing one parameter at a time around one point, which misses interactions and says nothing about how sure the answer is.

Sensitivity Lab computes Sobol first-order and total indices with bootstrap 95 % intervals, plus second-order interactions, Morris screening, effect curves and local derivatives. It works on a zoo model, an uploaded ONNX graph, runs done in OpenFOAM or CalculiX, or a dataset. The evaluation count is shown before anything runs.

In the video I use a heat-sink surrogate with 10 parameters and 2 responses. The study took 2.33 s. Fin count controls the thermal resistance: total index 0.54 (95 % CI 0.50-0.59). For the pressure drop, fin count (0.40) and air velocity (0.37) dominate.

Ambient temperature and power can be frozen. The rule is that the upper bound of the interval must be below 0.01 for every response, and the largest bound was 1.5e-4. The app then checks this: it re-ran the model on 512 samples with both fixed, and the variance ratio was 1.000 with a largest change of 0.071 standard deviations. On the Ishigami benchmark, where the exact indices are known, the largest error was 0.0027, and all 6 exact values fell inside the intervals. Results download as a CSV of indices and a StudyResult JSON.

Live: https://sensitivity.pinneapple.org
GitHub: https://github.com/pinneapple-labs/pinneapple (apps/sensitivity_lab)

#SensitivityAnalysis #UncertaintyQuantification #SimulationEngineering #DesignOfExperiments #ScientificML #ThermalManagement #CFD
