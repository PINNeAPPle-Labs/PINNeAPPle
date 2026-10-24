Day 21 of 30: Inverse Model Studio

Inverse problems are usually solved one way. Was a PINN needed, or would least squares do? Is the reconstructed field right where nobody measured? Each method lives in its own script, so they are rarely compared on the same data.

Inverse Model Studio takes one problem formulation (measurements, a physical model, the unknowns) and solves it with up to five methods on the same data: least squares, adjoint gradient, PyTorch autodiff, Bayesian regularisation and a physics-informed network. Held-out measurements, never fitted, rank them.

In the video, a fan cools a 6 mm aluminium fin unevenly. From 12 thermocouples (±0.15 °C) it reconstructs h(x) in four zones. Least squares, adjoint and autodiff agree: 91.1 / 44 / 23.6 / 27.3 W/m²K against a truth of 90 / 45 / 20 / 30, at a cost of 11, 38 and 47 model runs. The Bayesian method gives 91.2 / 43.8 / 23.9 / 27.2 with narrower intervals. The truth is inside the 95 % interval for 4 of 4 methods. The temperature field is within 0.15 °C of the truth everywhere, and the held-out RMSE is 0.173 °C.

The checks come with the answer: identifiability (h3 least determined, ±15 %), residuals, and a scope card stating the limit: if the model misses a physical effect, every method can agree and all be wrong. The PINN is greyed out here, with the reason.

Downloads: report JSON, StudyResult, CSV estimates and fields.

Live: https://inverse-studio.pinneapple.org
GitHub: https://github.com/pinneapple-labs/pinneapple (apps/inverse_studio)

#Engineering #HeatTransfer #InverseProblems #AdjointMethods #BayesianInference #PINN #Simulation #Python
