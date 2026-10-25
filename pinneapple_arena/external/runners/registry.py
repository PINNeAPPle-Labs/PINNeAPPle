"""
Runner registry: maps every model slug to its (dataset_fn, infer_fn) pair.

This is the single dispatch table the API layer uses for both the
"run inference" and the benchmark/compare endpoints. `call_infer()` wraps
the raw infer_* function call so callers never need to know whether a
given model's function accepts `include_heavy` or not.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from pinneapple_arena.external import demo_datasets as ds
from pinneapple_arena.external.local_command import local_command_for
from pinneapple_arena.external.models_data import MODELS_BY_SLUG
from pinneapple_arena.external.result import STATUS_OK, ModelResult
from pinneapple_arena.external.runners.applied_domains import (
    infer_crnn,
    infer_epipinn,
    infer_ml4gw,
    infer_neuralfoil,
    infer_neuralsolvers,
    infer_physics_informed_am,
    infer_physnet_thermal,
    infer_piml_converter,
    infer_pinn_ac_opf,
    infer_pinn_corrosion_fatigue,
    infer_pinn_multivp,
    infer_pinn_option_pricing,
    infer_pinn_swe_open,
    infer_pinn_wind_bearing,
    infer_pinnicle,
    infer_pise,
    infer_pyroomacoustics,
    infer_rirpinn,
    infer_simvascular,
    infer_socialforce,
    infer_spdiff,
    infer_stiff_pinn,
)
from pinneapple_arena.external.runners.dynamics_robotics_astro import (
    infer_astronomix,
    infer_diffpd,
    infer_diffrax,
    infer_diffsph,
    infer_difftaichi,
    infer_dojo_jl,
    infer_dsgp4,
    infer_genesis,
    infer_nimble_physics,
    infer_pysindy,
    infer_veros,
)
from pinneapple_arena.external.runners.em_photonics import (
    infer_ceviche,
    infer_hpinn,
    infer_photontorch,
)
from pinneapple_arena.external.runners.fluids_structural import (
    infer_deepflame,
    infer_dl4to,
    infer_fluidx3d,
    infer_gns,
    infer_graingnn,
    infer_jax_cfd,
    infer_jax_fluids,
    infer_lesnets,
    infer_meshgraphnets_original,
    infer_meshgraphnets_pytorch,
    infer_phiflow,
    infer_taichi,
    infer_twophasepinn,
    infer_warp,
)
from pinneapple_arena.external.runners.geophysics_energy import (
    infer_deepwave,
    infer_devito,
    infer_pybamm,
    infer_reservoirflow,
    infer_torax,
)
from pinneapple_arena.external.runners.materials2_and_misc import (
    infer_bpinn_wildfire,
    infer_chgnet,
    infer_fairchem,
    infer_grainlearning,
    infer_mattersim,
    infer_openfold,
    infer_orb_models,
    infer_tensorcircuit_ng,
    infer_torchmd,
    infer_transolver,
    infer_u_no,
    infer_wakenet,
)
from pinneapple_arena.external.runners.materials_md import (
    infer_allegro,
    infer_deepmd_kit,
    infer_jax_md,
    infer_mace,
    infer_nequip,
    infer_schnetpack,
)
from pinneapple_arena.external.runners.pinn_general import (
    infer_deeponet_fno_benchmark,
    infer_deeponet_original,
    infer_deepxde,
    infer_neuraloperator,
    infer_neuralpde_jl,
    infer_physicsnemo,
    infer_pina,
    infer_pinn_tfi_hss,
    infer_pinns_original,
    infer_pinns_torch,
    infer_pino,
    infer_sciann,
)
from pinneapple_arena.external.runners.quantum_cosmology import (
    infer_camels,
    infer_ferminet,
    infer_flowpm_jaxpm,
    infer_netket,
)
from pinneapple_arena.external.runners.rigidbody_robotics import infer_brax, infer_mujoco_mjx
from pinneapple_arena.external.runners.weather_climate import (
    infer_fourcastnet,
    infer_graphcast,
    infer_neuralgcm,
    infer_pangu_weather,
)

# slug -> (dataset_fn, infer_fn)
RUNNERS: dict[str, tuple[Callable[[], dict], Callable[..., ModelResult]]] = {
    'simvascular-svfsi': (ds.ds_grid_2d, infer_simvascular),
    'pyroomacoustics': (ds.ds_room, infer_pyroomacoustics),
    'rirpinn': (ds.ds_room, infer_rirpinn),
    'epipinn': (ds.ds_epidemic, infer_epipinn),
    'pise': (ds.ds_traffic, infer_pise),
    'pinn-ac-opf': (ds.ds_power_system, infer_pinn_ac_opf),
    'piml-converter': (ds.ds_power_system, infer_piml_converter),
    'ml4gw': (ds.ds_timeseries, infer_ml4gw),
    'physics-informed-am': (ds.ds_grid_2d, infer_physics_informed_am),
    'pinn-swe-open': (ds.ds_grid_2d, infer_pinn_swe_open),
    'pinn-corrosion-fatigue': (ds.ds_timeseries, infer_pinn_corrosion_fatigue),
    'pinn-wind-bearing': (ds.ds_timeseries, infer_pinn_wind_bearing),
    'pinnicle': (ds.ds_pde_1d, infer_pinnicle),
    'physnet-thermal': (ds.ds_grid_2d, infer_physnet_thermal),
    'neuralsolvers': (ds.ds_grid_2d, infer_neuralsolvers),
    'pinn-multivp': (ds.ds_timeseries, infer_pinn_multivp),
    'socialforce': (ds.ds_pedestrians, infer_socialforce),
    'spdiff': (ds.ds_pedestrians, infer_spdiff),
    'stiff-pinn': (ds.ds_combustion, infer_stiff_pinn),
    'crnn': (ds.ds_combustion, infer_crnn),
    'neuralfoil': (ds.ds_airfoil, infer_neuralfoil),
    'pinn-option-pricing': (ds.ds_option_pricing, infer_pinn_option_pricing),
    'ceviche': (ds.ds_em_grid, infer_ceviche),
    'photontorch': (ds.ds_em_grid, infer_photontorch),
    'hpinn': (ds.ds_em_grid, infer_hpinn),
    'phiflow': (ds.ds_grid_2d, infer_phiflow),
    'jax-cfd': (ds.ds_grid_2d, infer_jax_cfd),
    'jax-fluids': (ds.ds_grid_2d, infer_jax_fluids),
    'deepflame': (ds.ds_grid_2d, infer_deepflame),
    'fluidx3d': (ds.ds_grid_2d, infer_fluidx3d),
    'lesnets': (ds.ds_grid_2d, infer_lesnets),
    'twophasepinn': (ds.ds_grid_2d, infer_twophasepinn),
    'meshgraphnets-pytorch': (ds.ds_graph_mesh, infer_meshgraphnets_pytorch),
    'meshgraphnets-original': (ds.ds_graph_mesh, infer_meshgraphnets_original),
    'gns': (ds.ds_graph_mesh, infer_gns),
    'graingnn': (ds.ds_graph_mesh, infer_graingnn),
    'taichi': (ds.ds_grid_2d, infer_taichi),
    'warp': (ds.ds_rigidbody, infer_warp),
    'dl4to': (ds.ds_grid_2d, infer_dl4to),
    'devito': (ds.ds_seismic, infer_devito),
    'deepwave': (ds.ds_seismic, infer_deepwave),
    'torax': (ds.ds_weather_grid, infer_torax),
    'pybamm': (ds.ds_timeseries, infer_pybamm),
    'reservoirflow': (ds.ds_grid_2d, infer_reservoirflow),
    'jax-md': (ds.ds_molecular, infer_jax_md),
    'mace': (ds.ds_molecular, infer_mace),
    'nequip': (ds.ds_molecular, infer_nequip),
    'allegro': (ds.ds_molecular, infer_allegro),
    'deepmd-kit': (ds.ds_molecular, infer_deepmd_kit),
    'schnetpack': (ds.ds_molecular, infer_schnetpack),
    'deepxde': (ds.ds_pde_1d, infer_deepxde),
    'pina': (ds.ds_pde_1d, infer_pina),
    'physicsnemo': (ds.ds_grid_2d, infer_physicsnemo),
    'sciann': (ds.ds_pde_1d, infer_sciann),
    'neuralpde-jl': (ds.ds_pde_1d, infer_neuralpde_jl),
    'pinns-torch': (ds.ds_pde_1d, infer_pinns_torch),
    'pinns-original': (ds.ds_pde_1d, infer_pinns_original),
    'pinn-tfi-hss': (ds.ds_grid_2d, infer_pinn_tfi_hss),
    'neuraloperator': (ds.ds_burgers, infer_neuraloperator),
    'deeponet-original': (ds.ds_burgers, infer_deeponet_original),
    'deeponet-fno-benchmark': (ds.ds_burgers, infer_deeponet_fno_benchmark),
    'pino': (ds.ds_burgers, infer_pino),
    'netket': (ds.ds_spin_lattice, infer_netket),
    'ferminet': (ds.ds_spin_lattice, infer_ferminet),
    'flowpm-jaxpm': (ds.ds_cosmology, infer_flowpm_jaxpm),
    'camels': (ds.ds_cosmology, infer_camels),
    'brax': (ds.ds_rigidbody, infer_brax),
    'mujoco-mjx': (ds.ds_rigidbody, infer_mujoco_mjx),
    'graphcast': (ds.ds_weather_grid, infer_graphcast),
    'pangu-weather': (ds.ds_weather_grid, infer_pangu_weather),
    'fourcastnet': (ds.ds_weather_grid, infer_fourcastnet),
    'neuralgcm': (ds.ds_weather_grid, infer_neuralgcm),
    'pysindy': (ds.ds_lorenz, infer_pysindy),
    'diffrax': (ds.ds_lorenz, infer_diffrax),
    'genesis': (ds.ds_rigidbody, infer_genesis),
    'nimble-physics': (ds.ds_rigidbody, infer_nimble_physics),
    'dojo-jl': (ds.ds_rigidbody, infer_dojo_jl),
    'astronomix': (ds.ds_grid_2d, infer_astronomix),
    'dsgp4': (ds.ds_tle, infer_dsgp4),
    'veros': (ds.ds_grid_2d, infer_veros),
    'diffpd': (ds.ds_grid_2d, infer_diffpd),
    'openfold': (ds.ds_molecular, infer_openfold),
    'tensorcircuit-ng': (ds.ds_quantum_circuit, infer_tensorcircuit_ng),
    'transolver': (ds.ds_grid_2d, infer_transolver),
    'u-no': (ds.ds_burgers, infer_u_no),
    'difftaichi': (ds.ds_grid_2d, infer_difftaichi),
    'diffsph': (ds.ds_molecular, infer_diffsph),
    'torchmd': (ds.ds_molecular, infer_torchmd),
    'fairchem': (ds.ds_crystal, infer_fairchem),
    'mattersim': (ds.ds_crystal, infer_mattersim),
    'chgnet': (ds.ds_crystal, infer_chgnet),
    'orb-models': (ds.ds_crystal, infer_orb_models),
    'grainlearning': (ds.ds_molecular, infer_grainlearning),
    'wakenet': (ds.ds_grid_2d, infer_wakenet),
    'bpinn-wildfire': (ds.ds_grid_2d, infer_bpinn_wildfire),
}

def call_infer(slug: str, include_heavy: bool = False) -> ModelResult:
    """Build the demo dataset and run the real infer_* function for `slug`.
    Any non-"ok" result gets a `local_command` telling the user exactly how
    to get a real result on their own machine instead."""
    dataset_fn, infer_fn = RUNNERS[slug]
    dataset = dataset_fn()
    sig = inspect.signature(infer_fn)
    if "include_heavy" in sig.parameters:
        result = infer_fn(dataset, include_heavy=include_heavy)
    else:
        result = infer_fn(dataset)
    if result.status != STATUS_OK and not result.local_command:
        result.local_command = local_command_for(MODELS_BY_SLUG[slug], action="infer")
    return result


def get_dataset_preview(slug: str) -> dict:
    """Return the demo dataset's metadata + a JSON-safe preview of its arrays,
    without running inference — used by the "sample input" panel in the UI."""
    dataset_fn, _ = RUNNERS[slug]
    dataset = dataset_fn()
    preview = {}
    for key, value in dataset.items():
        if key in ("name", "description"):
            continue
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                preview[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                 "sample": value.flatten()[:8].tolist()}
            else:
                preview[key] = value
        except Exception:
            preview[key] = str(value)
    return {"name": dataset["name"], "description": dataset["description"], "fields": preview}
