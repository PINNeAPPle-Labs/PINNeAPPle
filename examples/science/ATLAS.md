# Science atlas 1.1.0

| Area | Subarea | Status | Covered by |
|---|---|---|---|
| Classical mechanics | Oscillators and resonance | partial | Lab experiment `oscillator` (damped oscillator, integrators against the exact solution) |
| Classical mechanics | Central forces, Kepler and N-body | partial | module `pinneapple_physics.astro.nbody` (Sun-planet N-body with conservation diagnostics); preset `kepler_two_body_orbit` (two-body orbit problem); preset `cr3bp_planar_synodic` (circular restricted three-body problem); Lab experiment `kepler_law` (law discovery from orbits) |
| Classical mechanics | Rigid bodies and multibody systems | partial | module `pinneapple_simulation.multibody.tree` (floating-base multibody: kinematics, mass matrix, implicit step); module `pinneapple_simulation.particle_dynamics.rigid_body` (differentiable rigid-body dynamics); preset `spacecraft_attitude_euler_rotation` (Euler rotation equations) |
| Classical mechanics | Lagrangian and Hamiltonian mechanics, chaos | planned | - |
| Classical mechanics | Flight dynamics | partial | module `pinneapple_simulation.flight` (6-DOF aircraft on the vortex lattice) |
| Electromagnetism and optics | Electrostatics | available | pp.science `em.coulomb_superposition`; pp.science `em.gauss_law`; pp.science `em.parallel_plate_capacitor`; module `pinneapple_simulation.electromagnetics.electrostatics` (2-D FEM electrostatics); preset `em_two_dielectric_capacitor` (two-dielectric capacitor) |
| Electromagnetism and optics | Magnetostatics | available | pp.science `em.biot_savart`; pp.science `em.ampere_law`; module `pinneapple_simulation.electromagnetics.fem` (A_z magnetostatics FEM); preset `em_wire_az` (wire, vector potential) |
| Electromagnetism and optics | Induction and circuits | available | pp.science `em.faraday_induction`; pp.science `em.circuits_rc_rl_rlc`; module `pinneapple_simulation.electromagnetics.eddy` (time-harmonic eddy currents); preset `em_skin_effect` (skin effect) |
| Electromagnetism and optics | Charge transport in fields (Hall effect) | available | pp.science `em.hall_effect` |
| Electromagnetism and optics | Waves, radiation and media | available | pp.science `em.dipole_radiation`; pp.science `em.lossy_plane_wave`; module `pinneapple_simulation.electromagnetics.fdtd` (FDTD on the Yee grid, 1-D to 3-D); module `pinneapple_simulation.electromagnetics.media` (Drude, Debye, Lorentz and anisotropic media) |
| Electromagnetism and optics | Optics: interfaces and scattering | available | pp.science `em.fresnel_snell`; pp.science `em.mie_scattering`; module `pinneapple_simulation.electromagnetics.mie` (Mie series); preset `em_cylinder_scattering` (cylinder scattering) |
| Electromagnetism and optics | Waveguides and resonators | available | pp.science `em.waveguide_modes`; pp.science `em.cavity_resonance`; preset `em_tm_cavity` (TM cavity) |
| Electromagnetism and optics | Laboratory (virtual bench) | available | pp.science `em.bench_mu0_solenoid`; pp.science `em.bench_rc_discharge` |
| Electromagnetism and optics | Research: FDTD 3D+t, convergence, UQ, inverse problems, surrogates | available | pp.science `em.research_fdtd_3d`; pp.science `em.research_uq_waveguide`; pp.science `em.research_inverse_permittivity`; pp.science `em.research_surrogate_hall` |
| Electromagnetism and optics | Machines and antennas | partial | module `pinneapple_simulation.electromagnetics.machines` (interior-PM machine FEA and drive); module `pinneapple_simulation.electromagnetics.antenna` (end-fire antennas with EMerge); module `pinneapple_physics.pcb` (PCB physics) |
| Electromagnetism and optics | Physics checks of any field | available | pp.science `em.gauss_law`; pp.science `em.dipole_radiation`; module `pinneapple_physics.maxwell_checks` (Gauss, Faraday, Ampere-Maxwell, Poynting residuals) |
| Thermodynamics and statistical physics | Equations of state and processes | partial | module `pinneapple_systems.process_components.real_gas_eos` (multi-component real-gas Helmholtz EOS); module `pinneapple_systems.process_components.polytropic_path` (polytropic compression and expansion) |
| Thermodynamics and statistical physics | Heat conduction | partial | module `pinneapple_physics.closed_form.heat` (closed-form heat conduction); preset `transient_heat_3d` (transient heat 3-D); Lab experiment `heat_xtfc` (heat equation by X-TFC) |
| Thermodynamics and statistical physics | Statistical mechanics (ensembles, Ising) | partial | module `pinneapple_quantum.data.quantum_dataset` (1-D Ising spin-chain energies) |
| Thermodynamics and statistical physics | Kinetic theory and transport | partial | preset `crystal_phonon` (phonon Boltzmann transport) |
| Fluid mechanics | Incompressible flow | partial | module `pinneapple_simulation.numerical_solvers.lbm` (lattice Boltzmann); module `pinneapple_simulation.numerical_solvers.ns3d_spectral` (3-D spectral Navier-Stokes); preset `ns_incompressible_2d` (2-D Navier-Stokes); Lab experiment `cylinder_lbm` (cylinder wake) |
| Fluid mechanics | Compressible flow and shocks | partial | module `pinneapple_simulation.numerical_solvers.compressible_fv` (Euler / Navier-Stokes finite volumes); module `pinneapple_physics.closed_form.compressible` (isentropic and shock relations) |
| Fluid mechanics | Turbulence | partial | module `pinneapple_physics.multiscale.turbulence` (decaying 2-D turbulence) |
| Fluid mechanics | Pipe flow and transients | partial | module `pinneapple_physics.closed_form.pipe_flow` (pipe flow closed forms); module `pinneapple_simulation.numerical_solvers.water_hammer_moc` (water hammer, method of characteristics) |
| Fluid mechanics | Burgers and nonlinear waves | partial | preset `burgers_1d` (viscous Burgers) |
| Continuum and solid mechanics | Linear elasticity | partial | module `pinneapple_simulation.numerical_solvers.solid_fem` (solid FEM); module `pinneapple_simulation.numerical_solvers.elasticity3d_fdm` (3-D elasticity FD); module `pinneapple_physics.solid.axisymmetric` (axisymmetric elasticity); preset `thick_walled_cylinder_lame` (Lame cylinder) |
| Continuum and solid mechanics | Fracture and fatigue | partial | preset `material_fracture_2d` (phase-field fracture); module `pinneapple_physics.closed_form.cantilever_fatigue` (cantilever fatigue) |
| Continuum and solid mechanics | Contact, friction and wear | partial | module `pinneapple_physics.tribology.archard` (Archard wear) |
| Continuum and solid mechanics | Soft and granular matter | partial | module `pinneapple_simulation.particle_dynamics.mpm` (material point method); module `pinneapple_simulation.terramechanics.wong_reece` (wheel-soil terramechanics) |
| Acoustics and waves | Acoustic resonators | partial | module `pinneapple_physics.closed_form.helmholtz_resonator` (Helmholtz resonator); preset `helmholtz_acoustics_3d` (3-D Helmholtz acoustics) |
| Acoustics and waves | Ultrasound | partial | preset `wave_ultrasound_3d` (3-D ultrasound wave); module `pinneapple_analysis.inspection.phased_array_ultrasonic` (phased-array inspection) |
| Acoustics and waves | Pressure waves in pipes | partial | module `pinneapple_systems.process_components.water_hammer` (water hammer) |
| Quantum mechanics | Schrodinger equation and bound states | partial | module `pinneapple_quantum.loss.schrodinger` (harmonic, well, Coulomb and double-well potentials); module `pinneapple_quantum.pipeline.presets` (variational and hybrid PINNs) |
| Quantum mechanics | Non-Hermitian and PT-symmetric systems | partial | module `pinneapple_physics.non_hermitian.pt_dimer` (PT dimers and exceptional points) |
| Quantum mechanics | Scattering, spin and angular momentum | planned | - |
| Special and general relativity | Special relativity | planned | no Lorentz-transformation code yet |
| Special and general relativity | General relativity: geodesics and lensing | partial | module `pinneapple_physics.blackhole.raytrace` (Schwarzschild geodesic ray tracing); module `pinneapple_physics.astro.lensing` (gravitational lensing); preset `schwarzschild_light_bending_weak_field` (weak-field light bending) |
| Nuclear and particle physics | Radioactive decay and chains | planned | - |
| Nuclear and particle physics | Relativistic kinematics and scattering | planned | - |
| Nuclear and particle physics | Detector response | planned | - |
| Condensed matter | Phonons and thermal transport | partial | preset `crystal_phonon` (phonon Boltzmann transport) |
| Condensed matter | Spin models | partial | module `pinneapple_quantum.data.quantum_dataset` (Ising spin chains) |
| Condensed matter | Electronic transport (Drude, Hall) | available | pp.science `em.hall_effect` |
| Condensed matter | Band structure | planned | - |
| Plasma physics | Cold plasma and dispersion | partial | module `pinneapple_simulation.electromagnetics.media` (Drude model of a plasma) |
| Plasma physics | Vlasov / particle-in-cell | planned | - |
| Plasma physics | Magnetohydrodynamics | planned | - |
| Astrophysics and cosmology | Orbital mechanics and ephemerides | partial | module `pinneapple_physics.astro.orbits` (orbital mechanics); module `pinneapple_physics.astro.nbody` (N-body) |
| Astrophysics and cosmology | Stars | partial | module `pinneapple_physics.astro.stars` (star catalogue, luminosity, Planck spectra); preset `lane_emden_polytrope` (Lane-Emden polytrope) |
| Astrophysics and cosmology | Accretion and black holes | partial | module `pinneapple_physics.blackhole.hydro` (black-hole accretion hydrodynamics); preset `shakura_sunyaev_accretion_disk` (thin disc); Lab experiment `bondi_accretion` (Bondi accretion) |
| Astrophysics and cosmology | Astronomical data analysis | partial | module `pinneapple_physics.pde_environment.presets.astroml_bridge` (astroML periodograms, Bayesian blocks); module `pinneapple_physics.pde_environment.presets.astropy_bridge` (astropy constants and frames) |
| Astrophysics and cosmology | Cosmology | partial | preset `nfw_dark_matter_potential` (NFW dark-matter halo) |
| Geophysics | Subsurface flow | partial | module `pinneapple_simulation.geophysics.richards` (Richards equation) |
| Geophysics | Geophysical fluid dynamics | partial | module `pinneapple_simulation.geophysics.sphere_shallow_water` (shallow water on the sphere) |
| Geophysics | Seismic and elastic waves | planned | - |
| Chemistry | Reaction kinetics | partial | module `pinneapple_systems.process_components.reaction_kinetics` (mass-action networks, Arrhenius); module `pinneapple_simulation.numerical_solvers.stiff_kinetics_ode` (stiff kinetics) |
| Chemistry | Thermochemistry | planned | - |
| Chemistry | Molecular quantum chemistry | planned | - |
| Chemistry | Molecular dynamics | partial | module `pinneapple_simulation.particle_dynamics.molecular_dynamics` (Lennard-Jones MD, velocity Verlet) |
| Materials science | Phase fields | partial | module `pinneapple_simulation.numerical_solvers.reaction_diffusion` (Allen-Cahn and Fisher-KPP); preset `material_fracture_2d` (phase-field fracture) |
| Materials science | Magnetic and dielectric materials | partial | module `pinneapple_simulation.electromagnetics.hysteresis` (Jiles-Atherton hysteresis); module `pinneapple_simulation.electromagnetics.media` (dispersive media) |
| Biology and biophysics | Pharmacokinetics | partial | preset `pk_two_compartment` (two-compartment PK) |
| Biology and biophysics | Transport in tissue | partial | preset `drug_diffusion_tissue` (drug diffusion) |
| Biology and biophysics | Pattern formation | partial | preset `reaction_diffusion_2d` (reaction-diffusion 2-D) |
| Neuroscience | Excitable media | partial | module `pinneapple_arena.problems` (FitzHugh-Nagumo 1-D) |
| Neuroscience | Hodgkin-Huxley and integrate-and-fire | planned | - |
| Ecology and epidemiology | Epidemics | partial | preset `sir_epidemic` (SIR model) |
| Ecology and epidemiology | Population dynamics | partial | module `pinneapple_arena.dataset_bench` (Lotka-Volterra dataset) |
| Ecology and epidemiology | Social dynamics | partial | preset `opinion_dynamics_2d` (opinion dynamics) |
| Earth and climate sciences | Climate energy balance | partial | module `pinneapple_simulation.geophysics.energy_balance` (two-layer energy-balance model) |
| Earth and climate sciences | ENSO | partial | module `pinneapple_systems.time_series.enso.oscillators` (recharge and delayed oscillators) |
| Earth and climate sciences | Atmosphere and ocean | partial | preset `climate_atmosphere_2d` (atmosphere 2-D); preset `climate_ocean_gyre` (ocean gyre); module `pinneapple_physics.weather.model` (global forecast network) |
| Applied mathematics and numerical methods | Time integrators | partial | module `pinneapple_simulation.numerical_solvers.irk_gauss_legendre` (Gauss-Legendre IRK); module `pinneapple_simulation.numerical_solvers.etdrk4` (ETDRK4) |
| Applied mathematics and numerical methods | Discretisations | partial | module `pinneapple_simulation.numerical_solvers.fvm` (finite volumes); module `pinneapple_simulation.numerical_solvers.meshfree` (RBF and MLS); module `pinneapple_simulation.numerical_solvers.xtfc_pde` (X-TFC) |
| Applied mathematics and numerical methods | Verification and convergence | partial | module `pinneapple_analysis.verification.convergence` (Richardson, GCI); pp.science `em.research_fdtd_3d` |
| Applied mathematics and numerical methods | Uncertainty quantification | partial | module `pinneapple_analysis.uncertainty` (propagation, calibration, conformal); pp.science `em.research_uq_waveguide` |
| Applied mathematics and numerical methods | Inverse problems | partial | module `pinneapple_analysis.inverse_problems` (EKI, sensitivity, misfits); pp.science `em.research_inverse_permittivity` |
