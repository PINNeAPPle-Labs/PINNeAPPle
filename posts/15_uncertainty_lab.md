Day 15 of 30: Engineering Uncertainty Lab

A surrogate quoted to three digits looks accurate, yet it can drive a bad decision when the case lies outside its data, where data are sparse, or when its error exceeds the margin to a limit.

The Engineering Uncertainty Lab takes a model (a pp.zoo model or an uploaded ONNX graph), reference data and a list of cases. It builds split-conformal intervals, measures their coverage on held-out data, flags each case inside, sparse or outside, and routes each case to the surrogate or the high-fidelity solver, with the reason.

In the video, the heat-sink surrogate gets a 90 % interval of ±1.60 % of the prediction. Measured on 900 held-out designs, it covers 91.0 % (95 % CI 89.0–92.7 %). Of 11 designs screened against R ≤ 0.4 K/W, 5 go to the solver: 2 outside the range, 2 in sparse data, 1 whose interval contains the limit.

To check the rule, the physics engine also solved all 11 designs. The 6 kept on the surrogate fell 100 % inside their intervals. The only miss, the corner of the design space, had already been flagged sparse and routed. Propagating the uncertain inputs gives P(meets limit) = 66.6 % on the prediction alone and 54.5 % once the model's own error is included.

A scope card lists what was validated and the limits. Results download as a cases CSV and a StudyResult JSON.

Live: https://uncertainty.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/uncertainty_lab)

#UncertaintyQuantification #ConformalPrediction #SurrogateModels #Simulation #ThermalEngineering #MachineLearning #EngineeringDesign
