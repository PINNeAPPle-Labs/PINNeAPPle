"""Physical-domain metadata: label, glyph and description per domain key.

Keep in sync with the `domains` tuples used in pinneapple_arena/external/models_data.py —
adding a new domain to a ModelSpec requires an entry here too.
"""

from __future__ import annotations

from pinneapple_arena.external.schema import DomainMeta

DOMAIN_META: dict[str, DomainMeta] = {
    'fluids': DomainMeta(key='fluids', label='Fluid Dynamics & CFD', glyph='∇·F=0', desc='Solvers and surrogates for incompressible/compressible flow, turbulence and aerodynamics.'),
    'structural': DomainMeta(key='structural', label='Structural & Solid Mechanics', glyph='σ=Eε', desc='Stress–strain, elasticity, deformation and mesh-based structural simulation.'),
    'general': DomainMeta(key='general', label='Heat Transfer & General Multiphysics PDEs', glyph='∂u/∂t', desc='General-purpose PINN frameworks applicable to any user-defined PDE — heat, diffusion, custom multiphysics.'),
    'particulate': DomainMeta(key='particulate', label='Particulate, Granular & Geotechnical', glyph='Σᵢmᵢ', desc='Particle- and mesh-free surrogates for granular flow, slope stability and multi-material systems.'),
    'rigidbody': DomainMeta(key='rigidbody', label='Rigid-Body Dynamics & Robotics', glyph='ẍ=F/m', desc='Fully differentiable rigid-body and contact physics for robotics and reinforcement learning.'),
    'plasma': DomainMeta(key='plasma', label='Plasma & Fusion Energy', glyph='∇×B=μJ', desc='Transport and confinement modeling for tokamak and fusion-reactor physics.'),
    'operator': DomainMeta(key='operator', label='General Operator Learning', glyph='𝒢(a)', desc='Function-to-function surrogates for design-space exploration, digital twins and any PDE family solved many times over.'),
    'industrial': DomainMeta(key='industrial', label='Industrial Multi-Domain Platforms', glyph='⊕', desc='Frameworks that bundle multiple physics domains into a single production-oriented stack.'),
    'materials': DomainMeta(key='materials', label='Materials Science & Molecular Dynamics', glyph='Σᵢⱼφ(rᵢⱼ)', desc='Machine-learned interatomic potentials and differentiable MD for materials and molecular simulation.'),
    'weather': DomainMeta(key='weather', label='Weather & Climate Forecasting', glyph='∂p/∂t', desc='Data-driven and hybrid physics-ML models for medium-range global weather and climate prediction.'),
    'em': DomainMeta(key='em', label='Electromagnetics & Photonics', glyph='∇×E=-∂B/∂t', desc='Differentiable Maxwell-equation solvers for photonic and electromagnetic inverse design.'),
    'geophysics': DomainMeta(key='geophysics', label='Geophysics & Seismic Imaging', glyph='∫∂²u/∂t²', desc='Wave-equation modeling and full-waveform inversion for subsurface imaging.'),
    'combustion': DomainMeta(key='combustion', label='Combustion & Reacting Flows', glyph='∂Yₖ/∂t', desc='Chemical kinetics and turbulent combustion accelerated by learned source-term models.'),
    'biomechanics': DomainMeta(key='biomechanics', label='Biomechanics & Physiological Modeling', glyph='∇·σ+f=0', desc='Patient-specific finite-element and multiphysics simulation of the cardiovascular system and soft tissue.'),
    'cosmology': DomainMeta(key='cosmology', label='Cosmology & Astrophysics', glyph='H²=8πGρ/3', desc='N-body and hydrodynamic simulation of cosmic structure formation, differentiable for field-level inference.'),
    'acoustics': DomainMeta(key='acoustics', label='Acoustics & Sound Propagation', glyph='∇²p=1/c²∂²p/∂t²', desc='Room acoustics and sound-field simulation underlying acoustic-ML and spatial audio research.'),
    'electrochemistry': DomainMeta(key='electrochemistry', label='Electrochemistry & Batteries', glyph='∂c/∂t=∇·(D∇c)', desc='Physics-based electrochemical models of battery charge, discharge and degradation.'),
    'quantum': DomainMeta(key='quantum', label='Quantum Many-Body Physics', glyph='Ĥψ=Eψ', desc='Neural-network variational wavefunctions for ground-state search in quantum lattice and electronic-structure problems.'),
    'epidemiology': DomainMeta(key='epidemiology', label='Epidemiology', glyph='dS/dt=-βSI', desc='Compartmental disease-transmission models fit to case data via physics-informed learning.'),
    'traffic': DomainMeta(key='traffic', label='Traffic Flow', glyph='∂ρ/∂t+∂q/∂x=0', desc='Macroscopic traffic-flow PDEs (LWR, CTM) reconstructed from sparse sensor data.'),
    'power': DomainMeta(key='power', label='Electrical Power Systems', glyph='ΣP=0', desc='Power-flow and grid-physics constraints embedded into learned optimization for the electrical grid.'),
    'gravwaves': DomainMeta(key='gravwaves', label='Gravitational Waves & Relativistic Astrophysics', glyph='□h_μν=16πGT_μν', desc='Detection, parameter estimation and waveform modeling for gravitational-wave and relativistic-astrophysics signals.'),
    'manufacturing': DomainMeta(key='manufacturing', label='Additive Manufacturing', glyph='ρc∂T/∂t=∇·(k∇T)', desc='Thermal and process-physics modeling for metal 3D printing and other advanced manufacturing processes.'),
    'hydrology': DomainMeta(key='hydrology', label='Hydrology & Shallow Water', glyph='∂h/∂t+∇·(hu)=0', desc='Shallow-water and subsurface-flow PDEs for flood prediction and groundwater modeling.'),
    'petroleum': DomainMeta(key='petroleum', label='Petroleum Engineering & Reservoir Simulation', glyph='∇·(k/μ ∇p)=φc∂p/∂t', desc='Subsurface fluid flow in oil, gas and geothermal reservoirs — Darcy flow, well behavior and wellbore management.'),
    'prognostics': DomainMeta(key='prognostics', label='Structural Health & Damage Prognosis', glyph='da/dN=C(ΔK)ᵐ', desc='Fatigue, corrosion and crack-growth models for remaining-useful-life prediction of mechanical components.'),
    'cryosphere': DomainMeta(key='cryosphere', label='Cryosphere & Ice Sheet Modeling', glyph='∇·(ηh∇u)=ρgh∇s', desc='Glaciology — ice flow, basal friction and sea-level-rise contribution modeling from sparse satellite data.'),
    'buildings': DomainMeta(key='buildings', label='Building Energy & HVAC', glyph='C dT/dt=UA(Tₐ-T)+Q', desc='Grey-box and physics-informed thermal models for building energy management and control.'),
    'optics': DomainMeta(key='optics', label='Optics & Laser Physics', glyph='i∂A/∂z=β∂²A/∂t²', desc='Nonlinear pulse propagation, photonic circuits and laser-plasma interaction, from PDE solvers to inverse photonic design.'),
    'heliophysics': DomainMeta(key='heliophysics', label='Heliophysics & Space Weather', glyph='∇×(v×B)=∂B/∂t', desc='Solar wind, coronal magnetic field and space-weather forecasting models.'),
    'crowd': DomainMeta(key='crowd', label='Crowd & Pedestrian Dynamics', glyph='m dv/dt=Σf_social', desc='Social-force and physics-informed models of pedestrian flow, evacuation and crowd simulation.'),
    'aerospace': DomainMeta(key='aerospace', label='Aerospace & Aerodynamics', glyph='L=½ρv²SC_L', desc='Airfoil, aircraft and rocket aerodynamics surrogates for rapid, gradient-friendly design optimization.'),
    'finance': DomainMeta(key='finance', label='Financial Markets', glyph='∂V/∂t+½σ²S²∂²V/∂S²=rV', desc='Option pricing and derivative-valuation PDEs (Black-Scholes and extensions) solved via physics-informed learning.'),
    'dynamics': DomainMeta(key='dynamics', label='Nonlinear Dynamics & System Identification', glyph='ẋ=Θ(x)ξ', desc='Data-driven discovery of governing ODEs/PDEs directly from time-series measurements via sparse regression or differentiable simulation.'),
    'astrofluids': DomainMeta(key='astrofluids', label='Astrophysical Fluids & MHD', glyph='∂B/∂t=∇×(v×B)', desc='Differentiable magnetohydrodynamics and compressible-fluid solvers for astrophysical structure formation, distinct from tokamak fusion-plasma transport.'),
    'astrodynamics': DomainMeta(key='astrodynamics', label='Astrodynamics & Orbital Mechanics', glyph='r̈=-μr/|r|³', desc='Differentiable orbit propagation, spacecraft trajectory and satellite-conjunction modeling.'),
    'ocean': DomainMeta(key='ocean', label='Ocean Modeling', glyph='∂u/∂t+u·∇u=-∇p/ρ+f', desc='General ocean circulation models spanning turbulence, mixing and large-scale currents, increasingly built on differentiable array backends.'),
    'softbody': DomainMeta(key='softbody', label='Soft-Body & Deformable Simulation', glyph='ρü=∇·P+f', desc='Differentiable FEM/projective-dynamics simulation of elastic and deformable solids, for inverse design and soft-robotics control.'),
    'biomolecular': DomainMeta(key='biomolecular', label='Biomolecular Structure & Folding', glyph='ΔG=ΔH-TΔS', desc='Deep-learning models predicting 3D protein/RNA structure from sequence, grounded in geometric and physical constraints.'),
    'wind': DomainMeta(key='wind', label='Wind Energy & Turbine Aerodynamics', glyph='P=½ρAv³Cₚ', desc='Learned surrogates for wind-turbine wake aerodynamics and wind-farm layout/yaw optimization.'),
    'wildfire': DomainMeta(key='wildfire', label='Wildfire Spread Modeling', glyph='∂φ/∂t+F|∇φ|=0', desc='Physics-informed fire-front propagation models coupling the level-set equation with satellite/sensor data assimilation.'),
    'patterns': DomainMeta(key='patterns', label='Pattern Formation & Active Matter', glyph='∂u/∂t=D∇²u+f(u,v)', desc='Reaction-diffusion Turing patterns and self-propelled active-particle suspensions.'),
}

