Day 24 of 30: Physics Validation Lab

A surrogate, a PINN or a new solver version produces numbers that look plausible. Before anyone uses them, someone must ask: are the equations satisfied, is energy conserved, do the boundary conditions hold, are the units right, does it match the reference? The answers live in ad-hoc scripts; the record that they were asked, nowhere.

The Validation Lab runs these checks on uploaded results, reference data or zoo models and sorts them into three groups: numerical verification, validation against data and physical consistency. Each check reports its value and tolerance and leaves an evidence record: SHA-256 of its inputs, outputs, start and finish time. A check that did not run is shown as "not run", never as a pass.

In the video, a temperature field described in a form passes 12 of 12 checks: PDE residual 2.2e-14, energy imbalance 4.5e-14, boundary deviation 0. Validation against data reads "not run" because no reference was given.

Then a case built to fail: a POD + GPR reduced model of a 3-D cantilever at its worst held-out design, against a fresh FEM solve. The clamp condition holds, but the reference check fails: 21.2 % relative L2 error on displacement and 27.7 % on von Mises stress, against a 5 % limit. The 3-D view puts the stress error near the clamp corners and the displacement error at the free tip.

Downloads: the report (JSON, HTML), the configuration that reproduces it and the evidence ledger.

Try it: https://validate.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/validation_lab)

#ScientificML #PhysicsInformedML #VerificationAndValidation #SurrogateModeling #FEM #ReducedOrderModeling #MachineLearning
