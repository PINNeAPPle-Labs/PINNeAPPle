- **The verification certificate, measured** (#341): `pinneapple_analysis.trust.certificate.verify(fields, problem)`
  returns a certificate for advection-diffusion, heat, incompressible flow, shallow water and elasticity — per check the
  number, the limit it was judged against and where it came from, the point where it is worst, the units, the declared
  boundaries, positivity and the domain gate. `pinneapple_analysis.trust.benchmark` measures the decision on the five
  families and on one real CFD surrogate (the POD `OperatorSurrogate` over the Taylor-Green vortex, four viscosities
  inside its fitted envelope and four outside): 19/19 in-domain predictions certified, 14/20 out-of-domain ones refused,
  in six seconds (`python scripts/certificate_benchmark.py`). The six misses are reported by name instead of averaged
  into the rate — the sign inversion the residual of a linear homogeneous law cannot see, the undamped direction of a
  viscosity error, and one small extrapolation of the surrogate. Five-minute quick start:
  [`docs/guides/verification_service.md`](docs/guides/verification_service.md).
