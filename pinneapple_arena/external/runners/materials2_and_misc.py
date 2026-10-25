"""
Second wave of materials-ML potentials plus a grab-bag of other domains
(biomolecular structure, quantum circuits, newer neural-operator paper code,
wind-wake surrogates, wildfire PINNs) added from the second catalog-expansion
research pass. Same runner contract as every other module — see
dynamics_robotics_astro.py's docstring.
"""

from __future__ import annotations

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    ModelResult,
    Timer,
    failed,
    manual_install,
    not_installed,
    ok,
    skipped_heavy,
    summarize_array,
)


def infer_openfold(ds: dict) -> ModelResult:
    meta = dict(name="OpenFold", slug="openfold", category="domain", domains=["biomolecular"],
                repo="https://github.com/aqlaboratory/openfold")
    return manual_install(**meta, instructions="Installed from source/conda per the repo README, not a "
                                                "simple pip package; pretrained weights (CC BY 4.0) are "
                                                "downloaded separately and are multi-gigabyte.", dataset=ds)


def infer_tensorcircuit_ng(ds: dict) -> ModelResult:
    meta = dict(name="TensorCircuit-NG", slug="tensorcircuit-ng", category="domain", domains=["quantum"],
                repo="https://github.com/tensorcircuit/tensorcircuit-ng")
    if not is_installed("tensorcircuit"):
        return not_installed(**meta, pip_hint="pip install tensorcircuit-ng", dataset=ds)
    with Timer() as t:
        try:
            import tensorcircuit as tc
            n = ds["n_qubits"]
            c = tc.Circuit(n)
            c.h(0)
            for i in range(n - 1):
                c.cnot(i, i + 1)
            z0 = c.expectation_ps(z=[0])
            state = c.state()
            out = np.asarray(state)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=f"{n}-qubit circuit",
              output_shape=str(out.shape), output_summary={"z0_expectation": complex(np.asarray(z0)).real},
              note="Built an H+CNOT-ladder circuit (a GHZ-state preparation) and computed <Z0> and the full "
                   "statevector.")


def infer_transolver(ds: dict) -> ModelResult:
    meta = dict(name="Transolver", slug="transolver", category="operator", domains=["operator"],
                repo="https://github.com/thuml/Transolver")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)


def infer_u_no(ds: dict) -> ModelResult:
    meta = dict(name="U-NO", slug="u-no", category="operator", domains=["operator"],
                repo="https://github.com/ashiq24/UNO")
    return manual_install(**meta, instructions="Standalone paper code, not pip-packaged — `git clone` only. "
                                                "Its U-shaped multi-scale FNO variant has also been folded "
                                                "into the `neuraloperator` package (already in this catalog).",
                           dataset=ds)


def infer_torchmd(ds: dict) -> ModelResult:
    meta = dict(name="TorchMD", slug="torchmd", category="domain", domains=["materials"],
                repo="https://github.com/torchmd/torchmd")
    if not (is_installed("torchmd") and is_installed("torch")):
        return not_installed(**meta, pip_hint="pip install torchmd", dataset=ds)
    with Timer() as t:
        try:
            import torchmd
            version = getattr(torchmd, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"torchmd {version} imported; a real MD run needs a parsed topology/force-field "
                   "(PSF/PRM or a trained torchmd-net potential) per the repo's tutorial notebooks.")


def infer_fairchem(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="FAIRChem (Open Catalyst Project)", slug="fairchem", category="domain",
                domains=["materials"], repo="https://github.com/facebookresearch/fairchem")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install fairchem-core", dataset=ds)
    if not is_installed("fairchem"):
        return not_installed(**meta, pip_hint="pip install fairchem-core", dataset=ds)
    with Timer() as t:
        try:
            import fairchem
            version = getattr(fairchem, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"fairchem {version} imported; real inference needs one of its pretrained UMA/OC20 "
                   "checkpoints (multi-GB, downloaded via the fairchem-core CLI/Hugging Face Hub).")


def infer_mattersim(ds: dict, include_heavy: bool = False) -> ModelResult:
    meta = dict(name="MatterSim", slug="mattersim", category="domain", domains=["materials"],
                repo="https://github.com/microsoft/mattersim")
    if not include_heavy:
        return skipped_heavy(**meta, pip_hint="pip install mattersim", dataset=ds)
    if not is_installed("mattersim"):
        return not_installed(**meta, pip_hint="pip install mattersim", dataset=ds)
    with Timer() as t:
        try:
            import mattersim
            version = getattr(mattersim, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"mattersim {version} imported; real inference needs one of its pretrained checkpoints "
                   "loaded via mattersim.forcefield.MatterSimCalculator.")


def infer_chgnet(ds: dict) -> ModelResult:
    meta = dict(name="CHGNet", slug="chgnet", category="domain", domains=["materials"],
                repo="https://github.com/CederGroupHub/chgnet")
    if not is_installed("chgnet"):
        return not_installed(**meta, pip_hint="pip install chgnet", dataset=ds)
    with Timer() as t:
        import torch
        torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
        torch.set_default_dtype(torch.float32)  # undo any float64 default left by a prior model call
        # CHGNet picks its own device internally (`mps` if available) regardless of the
        # default device above, and its float64 pymatgen-derived inputs then can't move to
        # MPS. Force `mps.is_available()` to False for just this call so it falls back to
        # real CPU execution instead of failing on that MPS/float64 mismatch.
        mps_is_available = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            from chgnet.model import CHGNet
            from pymatgen.core import Lattice, Structure
            lattice = Lattice.cubic(ds["lattice_a"])
            structure = Structure(lattice, ds["species"], ds["frac_coords"])
            model = CHGNet.load()
            prediction = model.predict_structure(structure)
            energy = float(prediction["e"])
            forces = np.asarray(prediction["f"])
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
        finally:
            torch.backends.mps.is_available = mps_is_available
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=f"{len(ds['species'])}-atom unit cell",
              output_shape=str(forces.shape), output_summary={"energy_ev_per_atom": energy, **summarize_array(forces)},
              note="Loaded CHGNet's pretrained checkpoint and predicted energy/forces for a small synthetic "
                   "BCC-iron-like unit cell.")


def infer_orb_models(ds: dict) -> ModelResult:
    meta = dict(name="Orb models", slug="orb-models", category="domain", domains=["materials"],
                repo="https://github.com/orbital-materials/orb-models")
    if not is_installed("orb_models"):
        return not_installed(**meta, pip_hint="pip install orb-models", dataset=ds)
    with Timer() as t:
        try:
            import orb_models
            version = getattr(orb_models, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"orb_models {version} imported; real inference needs one of the pretrained Orb-v2/v3 "
                   "checkpoints loaded via orb_models.forcefield.pretrained.")


def infer_grainlearning(ds: dict) -> ModelResult:
    meta = dict(name="GrainLearning", slug="grainlearning", category="domain", domains=["particulate"],
                repo="https://github.com/GrainLearning/grainLearning")
    if not is_installed("grainlearning"):
        return not_installed(**meta, pip_hint="pip install grainlearning", dataset=ds)
    with Timer() as t:
        try:
            import grainlearning
            version = getattr(grainlearning, "__version__", "unknown")
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, output_shape="import only",
              note=f"grainlearning {version} imported; a real calibration run needs a configured "
                   "IterativeBayesianFilter over a DEM/continuum model's parameter space per the repo's "
                   "tutorials.")


def infer_wakenet(ds: dict) -> ModelResult:
    meta = dict(name="WakeNet", slug="wakenet", category="domain", domains=["wind"],
                repo="https://github.com/soanagno/wakenet")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)


def infer_bpinn_wildfire(ds: dict) -> ModelResult:
    meta = dict(name="BPINN-Wildfire", slug="bpinn-wildfire", category="pinn", domains=["wildfire"],
                repo="https://github.com/jjdabr/BPINN-Wildfire")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)
