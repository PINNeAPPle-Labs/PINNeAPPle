- `pp.potential` (`pinneapple_physics.potential`, issue #928): derivative trees from one potential. `Potential(fn,
  variables, names=..., responses=...)` in torch or SymPy (`Potential.from_sympy`), `tree(potential, depth, at=...)`
  by autodiff (scalar or tensor variables, symbolic derivatives alongside for SymPy potentials), `legendre(...)`
  (symbolic when the conjugate equation has one root, else bracketed Newton with derivatives exact to seventh order)
  with a convexity check that reports where the transform fails (the van der Waals spinodal region),
  `maxwell_relations` and `consistency_check` (how far a property-by-property model, a table or grid data is from
  one potential). Adapters: `thermo` (Helmholtz / Gibbs trees, ideal gas, van der Waals, Peng-Robinson; identities
  in `(T, v)` and `(T, rho)`), `hyperelastic` (Neo-Hookean, Mooney-Rivlin, major symmetry), `mechanics` (Lagrangian
  trees, Euler-Lagrange, Hamiltonian, RK4), `machines` (local co-energy potential of a machine from FEA samples).
  Known-answer tests in `tests/test_potential.py`: closed forms, CoolProp's Peng-Robinson backend (agreement within
  2e-13 relative where installed), and the surface-PM machine torque from co-energy against Maxwell stress (16.854
  vs 16.837 N m). Flow nodes `potential.*`; extra `pinneapple[potential]` (SymPy, CoolProp).
