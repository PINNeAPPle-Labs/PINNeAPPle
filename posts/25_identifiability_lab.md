Day 25 of 30: Parameter Identifiability Lab

A calibration always returns numbers. It does not tell you when the data could not have determined them.

Take a stainless pin fin with five thermocouples (±0.5 °C) and two unknowns: the convection coefficient h and the conductivity k. Fit it and you get h = 73.8 W/m²K and k = 47.3 W/m·K. Both are wrong (truth 25 and 16), yet the fit is fine, because the temperatures depend only on h/k: 1.56 fitted, 1.56 true.

The Parameter Identifiability Lab asks the question before you calibrate. On this case it reports Fisher rank 1 of 2, h and k structurally non-identifiable, and names the combination the data do fix (h/k). The MCMC posterior agrees: h anywhere from 1.7 to 672 W/m²K, limited by the bounds rather than the data.

It then says what to measure. A greedy D-optimal design picks the heat flow at the fin base, which raises the rank to 2 of 2. With that one ±2 % heat-flux reading added, both parameters become identifiable: profile-likelihood 95 % intervals h 24.1–26.2 and k 15.4–16.8, Cramér-Rao bounds about ±2 %.

Four methods look at the same problem: Fisher information, local sensitivity, profile likelihood and an MCMC posterior. A scope card states the limits: the analysis is local and assumes the model form is right. The report, the CSV tables and a StudyResult that the calibration and inverse apps can read are all downloadable.

Live: https://identifiability.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/identifiability_lab)

#Identifiability #ParameterEstimation #DesignOfExperiments #InverseProblems #HeatTransfer #ScientificComputing #Engineering
