"""
Single source of truth for the external dataset catalog — the dataset
counterpart of models_data.py. See DatasetSpec in schema.py for why datasets
live in their own catalog instead of being forced into ModelSpec.
"""

from __future__ import annotations

from pinneapple_arena.external.schema import DatasetSpec, SubDataset


def _sub(name: str, label: str, domains: list[str], dims: str, desc: str) -> SubDataset:
    return SubDataset(name=name, label=label, domains=tuple(domains), dims=dims, desc=desc)


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        slug='the-well', name='The Well', org='Polymathic AI',
        desc='15 TB collection of 16 large-scale physics-simulation datasets — fluids, MHD, supernovae, stellar convection, '
             'acoustic scattering, active matter and more — produced by domain experts with established numerical solvers and '
             'shipped in a uniform HDF5 layout behind one PyTorch WellDataset interface. Several physical datasets come in '
             'more than one resolution or boundary-condition variant, so the loader exposes more identifiers than 16.',
        tags=('PyTorch', 'HDF5', 'Hugging Face', 'Benchmark', 'NeurIPS 2024'),
        repo='https://github.com/PolymathicAI/the_well',
        ref='Ohana et al. — The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning, NeurIPS 2024 Datasets & Benchmarks',
        size='15 TB',
        pip_package='the_well', import_name='the_well',
        stream_base_path='hf://datasets/polymathic-ai/',
        stream_requires=('huggingface_hub',),
        download_command='the-well-download --base-path ./datasets --dataset {subset} --split {split}',
        license='BSD-3-Clause',
        homepage='https://polymathic-ai.org/the_well/',
        subsets=(
            _sub('acoustic_scattering_discontinuous', 'Acoustic scattering — discontinuous', ['acoustics'], '2D',
                 'Acoustic wave propagation through a medium with discontinuous material properties.'),
            _sub('acoustic_scattering_inclusions', 'Acoustic scattering — inclusions', ['acoustics'], '2D',
                 'Acoustic wave propagation scattering off embedded inclusions.'),
            _sub('acoustic_scattering_maze', 'Acoustic scattering — maze', ['acoustics'], '2D',
                 'Acoustic wave propagation through a maze-like domain.'),
            _sub('active_matter', 'Active matter', ['patterns'], '2D',
                 'Suspension of self-propelled rod-like active particles in a Stokes fluid.'),
            _sub('convective_envelope_rsg', 'Red supergiant convective envelope', ['astrofluids'], '3D',
                 'Radiation-hydrodynamics of the convective envelope of a red supergiant star.'),
            _sub('euler_multi_quadrants_openBC', 'Euler multi-quadrant Riemann — open BC', ['fluids'], '2D',
                 'Compressible Euler equations from multi-quadrant Riemann initial conditions, open boundaries.'),
            _sub('euler_multi_quadrants_periodicBC', 'Euler multi-quadrant Riemann — periodic BC', ['fluids'], '2D',
                 'Compressible Euler equations from multi-quadrant Riemann initial conditions, periodic boundaries.'),
            _sub('gray_scott_reaction_diffusion', 'Gray-Scott reaction-diffusion', ['patterns'], '2D',
                 'Two-species reaction-diffusion system producing Turing-type patterns.'),
            _sub('helmholtz_staircase', 'Helmholtz staircase', ['acoustics'], '2D',
                 'Time-harmonic (Helmholtz) acoustic scattering off a periodic staircase boundary.'),
            _sub('MHD_64', 'MHD turbulence (64³)', ['astrofluids'], '3D',
                 'Compressible magnetohydrodynamic turbulence, 64³ resolution.'),
            _sub('MHD_256', 'MHD turbulence (256³)', ['astrofluids'], '3D',
                 'Compressible magnetohydrodynamic turbulence, 256³ resolution.'),
            _sub('planetswe', 'Planetary shallow water', ['weather'], '2D',
                 'Shallow-water equations on a rotating sphere — a simplified atmospheric-dynamics testbed.'),
            _sub('post_neutron_star_merger', 'Post neutron-star merger', ['astrofluids', 'gravwaves'], '3D',
                 'General-relativistic MHD of the accretion disk left after a binary neutron-star merger.'),
            _sub('rayleigh_benard', 'Rayleigh-Bénard convection', ['fluids'], '2D',
                 'Buoyancy-driven thermal convection between a heated bottom and cooled top plate.'),
            _sub('rayleigh_benard_uniform', 'Rayleigh-Bénard convection — uniform grid', ['fluids'], '2D',
                 'Rayleigh-Bénard convection variant on a uniform grid.'),
            _sub('rayleigh_taylor_instability', 'Rayleigh-Taylor instability', ['fluids'], '3D',
                 'Mixing of a heavy fluid over a light one under gravity.'),
            _sub('shear_flow', 'Shear flow', ['fluids'], '2D',
                 'Incompressible shear-layer flow developing Kelvin-Helmholtz-type instabilities.'),
            _sub('supernova_explosion_64', 'Supernova explosion (64³)', ['astrofluids'], '3D',
                 'Supernova blast wave expanding into a turbulent interstellar medium, 64³ resolution.'),
            _sub('supernova_explosion_128', 'Supernova explosion (128³)', ['astrofluids'], '3D',
                 'Supernova blast wave expanding into a turbulent interstellar medium, 128³ resolution.'),
            _sub('turbulence_gravity_cooling', 'Turbulence with gravity and cooling', ['astrofluids'], '3D',
                 'Turbulent interstellar medium with self-gravity and radiative cooling.'),
            _sub('turbulent_radiative_layer_2D', 'Turbulent radiative layer (2D)', ['astrofluids'], '2D',
                 'Radiatively cooling turbulent mixing layer between hot and cold gas.'),
            _sub('turbulent_radiative_layer_3D', 'Turbulent radiative layer (3D)', ['astrofluids'], '3D',
                 'Radiatively cooling turbulent mixing layer between hot and cold gas, 3D.'),
            _sub('viscoelastic_instability', 'Viscoelastic instability', ['fluids'], '2D',
                 'Instabilities in viscoelastic (polymeric) channel flow.'),
            _sub('viscoelastic_instability_v2', 'Viscoelastic instability — v2', ['fluids'], '2D',
                 'Revised release of the viscoelastic-instability dataset.'),
        ),
    ),
]

DATASETS_BY_SLUG: dict[str, DatasetSpec] = {d.slug: d for d in DATASETS}
