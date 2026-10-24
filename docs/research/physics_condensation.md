# Adaptive Physics Condensation: geometry-independent thermal models of chip packages

Research line, track 1 of 3 (condensation). Tracks 2 (a physics query engine) and 3 (verification of what a PINN
has actually learned) build on this one and are reported separately.

Code: [`pinneapple_condense`](../../pinneapple_condense). Benchmark: [`examples/physics_condensation`](../../examples/physics_condensation/run_benchmark.py).
Every number below was measured in this repository's CI container (4 cores, 15 GB RAM, no GPU). Each comes from
`results/*.json` of that example. Each study ran in a fresh Python process, and *peak memory* is that process's
peak resident set size. Nothing below is quoted from another source unless it carries a citation.

## 1. The question

A 2.5D/3D package has 10^4 to 10^6 microbumps, TSVs and C4 bumps, and a fab-resolution CAD model has up to 10^9
polygons. An engineer asks only a few questions of it:

- the peak junction temperature;
- the temperature of each junction;
- how the heat divides between the lid and the board, and through each bump;
- the thermal resistance.

Can the cost of answering them track the *physical* complexity of the package rather than its element count:

- how many distinct kinds of structure it has;
- how far heat spreads;
- how many independent power patterns it sees?

Three objects get confused here:

| | what it is | what drives its size |
|---|---|---|
| **geometry** | the solids and their materials | number of features (bumps, vias, layers) |
| **discretisation** | the unknowns a solver needs to resolve the geometry and the fields | features x cells per feature |
| **fields / observables** | temperatures and fluxes, and the few numbers asked of them | the physics: diffusion length, sources, contrasts |

A fine mesh couples all three. The cost of a direct solve grows with the discretisation even when 10^5 bumps are
identical and the answer is a handful of numbers. Homogenisation (Keff) cuts the discretisation by replacing the
geometry with averages. It also erases whatever happens below the averaging scale, and Section 4.3 measures what it
erases. This work keeps the exact fine operator and *condenses* it:

1. every repeated block is reduced once to its interface operator;
2. identical blocks, and identical arrangements of blocks, share that work;
3. interior fields are rebuilt only where they are asked for;
4. an error estimate decides where the reduction has to be undone.

## 2. Related work

*How the references were checked.* The network in this container blocked direct access to publisher pages, DOI
resolvers, arXiv, patent offices and the vendor's site; only a web-search tool worked. Each reference below was
checked against search-result records of publisher, repository or index pages. Where a field could not be confirmed
it is marked.

**Homogenisation and multiscale methods.** Periodic homogenisation gives effective coefficients from cell problems
(Bensoussan, Lions and Papanicolaou 1978). For two-phase composites, the Hashin–Shtrikman bounds bracket the
effective conductivity (Hashin and Shtrikman, *J. Appl. Phys.* 33 (1962) 3125, doi:10.1063/1.1728579). Our Keff
baseline uses the lower bound in plane and the parallel rule through the layer. Multiscale finite elements (Hou and
Wu, *J. Comput. Phys.* 134 (1997) 169, doi:10.1006/jcph.1997.5682) and the heterogeneous multiscale method (E and
Engquist, *Commun. Math. Sci.* 1 (2003) 87) keep fine-scale information in coarse basis functions or local solves.

**Substructuring and component mode synthesis.**
- Static condensation onto interfaces and component mode synthesis go back to Hurty (*AIAA J.* 3 (1965) 678; DOI
  from citing lists only) and Craig and Bampton (*AIAA J.* 6 (1968) 1313, doi:10.2514/3.4741).
- Characteristic constraint modes reduce the interface itself (Castanier, Tan and Pierre, *AIAA J.* 39 (2001) 1182;
  DOI not confirmed).
- The closest prior work to ours is the static-condensation reduced basis element method. Huynh, Knezevic and Patera
  (*ESAIM: M2AN* 47 (2013) 213, doi:10.1051/m2an/2012022) and Eftang and Patera (*Int. J. Numer. Methods Eng.* 96
  (2013) 269, doi:10.1002/nme.4543) introduce port reduction with a posteriori error estimation.
- Our reduced interfaces with a residual-based estimate follow the same idea. The differences are the port basis
  (load-informed POD, Section 3.3), the nested memoised hierarchy, and the learned-operator fallback.
- Load-dependent Ritz vectors (Wilson, Yuan and Dickens, *Earthquake Eng. Struct. Dyn.* 10 (1982) 813; details from
  a citing list) motivate building interface modes from loads rather than from an eigenproblem.

**Fast direct solvers.**
- Nested dissection orders a mesh so that elimination proceeds through separators (George, *SIAM J. Numer. Anal.* 10
  (1973) 345, doi:10.1137/0710032).
- Hierarchical matrices (Hackbusch, *Computing* 62 (1999) 89, doi:10.1007/s006070050015) and HSS multifrontal
  solvers (Xia, Chandrasekaran, Gu and Li, *SIAM J. Matrix Anal. Appl.* 31 (2009) 1382, doi:10.1137/09074543X)
  compress the dense separator Schur complements, whose off-diagonal blocks have low numerical rank.
- The fast multipole method exploits the same far-field smoothness for particle interactions (Greengard and Rokhlin,
  *J. Comput. Phys.* 73 (1987) 325, doi:10.1016/0021-9991(87)90140-9).
- Our hierarchy is nested dissection on reduced interfaces, with memoisation of identical subtrees. Section 4.6
  measures the off-diagonal ranks of its top operator, but the H-matrix arithmetic is not implemented.

**Model reduction.** Projection-based parametric model reduction is surveyed by Benner, Gugercin and Willcox (*SIAM
Rev.* 57 (2015) 483, doi:10.1137/130932715). Hyper-reduction of nonlinear terms uses the empirical interpolation
method (Barrault, Maday, Nguyen and Patera, *C. R. Math.* 339 (2004) 667, doi:10.1016/j.crma.2004.08.006) and DEIM
(Chaturantabut and Sorensen, *SIAM J. Sci. Comput.* 32 (2010) 2737, doi:10.1137/090766498). The learned block
operator of Section 3.5 is a parametric POD of the operator entries with a Gaussian-process regression, the existing
`pinneapple_neural.architectures.rom.parametric.ParametricPOD`.

**Chip thermal models and neural operators.**
- Compact thermal models such as HotSpot (Skadron et al., ISCA 2003, pp. 2–13, doi:10.1109/ISCA.2003.1206984) and
  3D-ICE (Sridhar et al., ICCAD 2010, pp. 463–470, doi:10.1109/ICCAD.2010.5653749) use coarse grids or
  layer-homogenised resistances.
- Neural operators learn the map from power maps to fields:
  - FNO (Li et al., ICLR 2021, arXiv:2010.08895);
  - DeepONet (Lu et al., *Nat. Mach. Intell.* 3 (2021) 218, doi:10.1038/s42256-021-00302-5);
  - for 3D ICs, DeepOHeat (Liu, Li, Hu, Yu, Shiau, Ai, Zeng and Zhang, DAC 2023, arXiv:2302.12949);
  - physics-enforced networks for full-chip thermal analysis (Chen, Lu, Jin and Tan, ICCAD 2023; DOI not confirmed).
- These learn the whole field map, so their error is not bounded by a solver. Here learning is confined to block
  operators, inside an exact algebraic assembly with a check against the exact operator.

**PINN failure modes** (they motivate track 3): spectral bias and stiffness of the loss landscape (Wang, Teng and
Perdikaris, *SIAM J. Sci. Comput.* 43 (2021) A3055, doi:10.1137/20M1318043; Wang, Yu and Perdikaris, *J. Comput.
Phys.* 449 (2022) 110768, doi:10.1016/j.jcp.2021.110768), and failures at higher coefficients (Krishnapriyan et al.,
NeurIPS 2021, arXiv:2109.01050). PINNs: Raissi, Perdikaris and Karniadakis, *J. Comput. Phys.* 378 (2019) 686,
doi:10.1016/j.jcp.2018.10.045.

**Reference company (ideas only).** Vinci (getvinci.ai; formerly Vinci4D) sells physics AI for semiconductor
packaging, thermal and thermo-elastic. What search-result text of their pages and press releases says (the pages
themselves could not be loaded from this container):

- "up to 1000x faster than conventional simulation tools" (Business Wire, 2 Dec 2025);
- a homepage table of 20 s against 2 h for a commercial FEA solver at 117,440,512 DOF;
- no manual meshing;
- research notes that report results both without homogenisation and with per-layer Keff tensors.

These are the company's own claims; they are not verified here and are not compared with our numbers. The brief for
this work names a patent application, WO2026064513A1, as theirs, describing:

- a latent encoding of geometry and a separate topology representation;
- boundary conditions, materials and sources as inputs;
- a field-prediction model;
- verification or correction by the governing equations or a solver.

We could not retrieve the document: patent offices were unreachable and searches returned no record. Title,
applicant and claims are therefore **not verified**. The design here was not taken from it: there is no geometry
latent and no learned field predictor. Learning is confined to block operators inside an exact assembly. Whether
this work is free to operate against that application, or anything else, needs an IP specialist; nothing here is a
freedom-to-operate opinion.

## 3. Method

### 3.1 Benchmark and reference solver

`pinneapple_condense.package` builds a synthetic multilayer package from bottom to top:

| layer | thickness | contents |
|---|---|---|
| organic substrate | 200 µm | — |
| C4 bumps in underfill | 60 µm | — |
| silicon interposer with copper TSVs | 100 µm | — |
| microbumps in underfill | 20 µm | — |
| die | 100 µm | power on its bottom cell row (flip chip) |
| TIM | 30 µm | contact resistance 5e-6 m²K/W on both faces |
| copper lid | 500 µm | Robin cooling h = 2e4 W/m²K on top |

- The board side has h = 1e3 W/m²K. Laterally the package is a square array of 40 µm unit cells (*blocks*), each
  with one microbump, one TSV and one C4 bump, all square in plan.
- Materials are representative: Si k = 148, Cu k = 401, solder 58, underfill 0.5, TIM 4, substrate 0.8 W/m/K. The
  copper values match Incropera et al.'s Table A.1 (as transcribed); the silicon value is a handbook value (Wikipedia
  gives ~149). The package is synthetic and is not a product.
- One block is 8 x 8 x 23 = 1,472 cells.
- Power maps are a uniform background (40 %) plus Gaussian hotspots, at a mean of 1e6 W/m² (100 W/cm²).
- The reference (`fv`) is a cell-centred finite-volume scheme: harmonic face conductances and contact resistances
  in series. It is solved by sparse LU up to 6e4 unknowns and by AMG-preconditioned CG (pyamg) to a relative
  residual of 1e-11 above that.
- With all heat generated on the active row and non-negative power, the discrete maximum principle for this M-matrix
  scheme puts Tmax on that row (Varga, *Matrix Iterative Analysis*, 2nd ed., Springer 2000; Ciarlet and Raviart,
  *Comput. Methods Appl. Mech. Eng.* 2 (1973) 17, doi:10.1016/0045-7825(73)90019-4). Every reference run checks it.

### 3.2 Block condensation (exact)

Each block is a voxel box with one *port* node per lateral boundary face. Each port is joined to its cell by the
half-cell conductance; eliminating a shared port gives exactly the fine scheme's harmonic face conductance. The fine
problem is therefore the assembly of blocks through their ports, and nothing is approximated. Eliminating the cells
gives

```
S = K_pp − K_pc K_cc⁻¹ K_cp,     g = W p,  W = −K_pc K_cc⁻¹ E,     T_c = Z p − Y t
```

Blocks with the same variant (materials, a voided bump, a missing bump, a degraded contact) share `S`, `W`, `Y` and
`Z`, which are cached. A test checks this path against the reference to 1e-7 relative.

### 3.3 Reduced interfaces, error estimate, refinement

A face has 8 x 23 = 184 port nodes, and its trace is restricted to `m` modes: the constant, then the leading POD
modes of the traces seen on the interior faces of a 4 x 4 patch under every unit active-cell load. That is the space
of interface fields any power map can produce locally. The global problem is then a Galerkin restriction, with two
consequences:

- the error is the energy-norm best approximation over the chosen interface space;
- with the constant in every basis, the global heat balance holds to round-off.

The textbook choice, characteristic constraint modes (the softest eigenvectors of the face Schur complement), fails
here (Section 4.2). Its soft modes localise in the low-k underfill and substrate and miss the vertical temperature
profile.

Interiors are exact, so the fine residual lives only on port nodes. The per-face estimate is `η_F² = r_Fᵀ S_FF⁻¹
r_F`. Refinement marks the faces holding half of `Σ η²` (Dörfler marking) and either doubles their modes (`p`) or
restores their full port field (`full`).

### 3.4 Hierarchical condensation with memoised subtrees

`hierarchy.Hierarchy` splits the array in halves recursively. Each merge eliminates the shared faces of two
children: dense Schur complements on reduced ports, that is, nested dissection. A node's operator depends only on
its shape and its children's operators, so nodes are memoised by that key:

- a uniform 2^L x 2^L array needs 2L merges;
- a few variants add a few per level;
- the build cost follows the number of *distinct subtrees*, plus the dense algebra of the top separators, which
  grows with the array side.

A power map goes up the tree as condensed loads; ports come down. The downward pass can be restricted to the path
to one block, which is what a point query needs (track 2).

### 3.5 Learned block operators with verification

When blocks differ continuously (a void fraction, TIM degradation, TSV quality), each block would need its own
factorisation. `learned.LearnedBlocks` learns the reduced operator `(S_r, W_r, O_t, O_p)` as a function of the
block parameters `θ` from 48 exact condensations (Latin hypercube). It uses POD of the entries, taken as a correction
to the operator of the centre block, plus a GP regression. `solve_adaptive` then runs four steps:

1. Blocks outside the training box, or with a large predictive deviation, are condensed exactly.
2. The global problem is solved.
3. The hottest blocks and a random audit sample are compared with their exact operator *at the solved state*. The
   check uses the local junction discrepancy and the global effect of the heat-flow residual they inject,
   `A⁻¹ r_b`, computed with the already factorised global matrix.
4. Failing blocks are replaced by their exact operator and the problem is solved again. The audit gives an additive
   estimate for the blocks never checked.

### 3.6 Transient

`transient.CBTransient` applies Craig–Bampton to the heat equation:

- per block, static constraint modes, the exact static response to the block's own power, and `k` slowest
  fixed-interface modes;
- interface modes from steady *and* Laplace-shifted (`K + sC`) traces, because steady traces miss early-time
  profiles (Section 4.7);
- implicit Euler at a fixed step, against the fine reference at the same step.

## 4. Results

RESULTS_PLACEHOLDER
