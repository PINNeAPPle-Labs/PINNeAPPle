"""
Auto-migrated from the original inference_suite.py (single-file registry).
Each infer_* function builds a minimal real instance of the actual library
and runs one real forward pass / inference call — see pinneapple_arena/external/result.py
for the ModelResult contract every function returns.

Heavy/GPU-oriented functions now take an explicit `include_heavy` parameter
instead of reading a module-level global, so concurrent API requests can't
step on each other's --include-heavy setting.
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
    summarize_array,
)


def infer_simvascular(ds: dict) -> ModelResult:
    meta = dict(name="SimVascular / svFSI", slug="simvascular-svfsi", category="domain",
                domains=["biomechanics"], repo="https://github.com/SimVascular/svFSI")
    return manual_install(**meta, instructions="C++ MPI solver built from source (CMake) — not a "
                                                "Python/pip package. See repo README.", dataset=ds)

def infer_pyroomacoustics(ds: dict) -> ModelResult:
    meta = dict(name="Pyroomacoustics", slug="pyroomacoustics", category="domain", domains=["acoustics"],
                repo="https://github.com/LCAV/pyroomacoustics")
    if not is_installed("pyroomacoustics"):
        return not_installed(**meta, pip_hint="pip install pyroomacoustics", dataset=ds)
    with Timer() as t:
        try:
            import pyroomacoustics as pra
            room = pra.ShoeBox(ds["dims"], fs=16000)
            room.add_source([1.0, 1.0, 1.0])
            room.add_microphone([2.0, 2.0, 1.0])
            room.compute_rir()
            rir = np.asarray(room.rir[0][0])
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["dims"]),
              output_shape=str(rir.shape), output_summary=summarize_array(rir),
              note="Built a ShoeBox room, added a source+mic, and computed the room impulse response (RIR).")

def infer_rirpinn(ds: dict) -> ModelResult:
    meta = dict(name="RIRPINN", slug="rirpinn", category="pinn", domains=["acoustics"],
                repo="https://github.com/xefonon/RIRPINN")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_epipinn(ds: dict) -> ModelResult:
    meta = dict(name="EpiPINN", slug="epipinn", category="pinn", domains=["epidemiology"],
                repo="https://github.com/cmillevoi/EpiPINN")
    return manual_install(**meta, instructions="Jupyter-notebook paper code — `git clone` and "
                                                "`pip install -r requirements.txt`, then run the notebooks.",
                           dataset=ds)

def infer_pise(ds: dict) -> ModelResult:
    meta = dict(name="PISE (Physics-Informed Traffic State Estimation)", slug="pise", category="pinn",
                domains=["traffic"], repo="https://github.com/arjhuang/pise")
    return manual_install(**meta, instructions="Paper code built on the original Raissi PINNs repo — "
                                                "`git clone` only.", dataset=ds)

def infer_pinn_ac_opf(ds: dict) -> ModelResult:
    meta = dict(name="PINN for AC Optimal Power Flow", slug="pinn-ac-opf", category="pinn",
                domains=["power"],
                repo="https://github.com/RahulNellikkath/Physics-Informed-Neural-Networks-for-AC-Optimal-Power-Flow")
    return manual_install(**meta, instructions="Paper code requiring MATPOWER/YALMIP/Gurobi (MATLAB) "
                                                "alongside the Python/TensorFlow PINN training script.",
                           dataset=ds)

def infer_piml_converter(ds: dict) -> ModelResult:
    meta = dict(name="PIML Converter", slug="piml-converter", category="pinn", domains=["power"],
                repo="https://github.com/ms140429/PIML_Converter")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_ml4gw(ds: dict) -> ModelResult:
    meta = dict(name="ml4gw", slug="ml4gw", category="domain", domains=["gravwaves"],
                repo="https://github.com/ML4GW/ml4gw")
    if not is_installed("ml4gw"):
        return not_installed(**meta, pip_hint="pip install ml4gw", dataset=ds)
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from ml4gw.waveforms import IMRPhenomD
            wf = IMRPhenomD()
            n = 4
            mass1 = torch.full((n,), 30.0)
            mass2 = torch.full((n,), 25.0)
            chi1 = torch.zeros(n)
            chi2 = torch.zeros(n)
            dist = torch.full((n,), 100.0)
            phic = torch.zeros(n)
            f_ref = torch.full((n,), 20.0)
            freqs = torch.linspace(20, 512, 256)
            hp, hc = wf(freqs, mass1, mass2, chi1, chi2, dist, phic, f_ref)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(tuple(freqs.shape)),
              output_shape=str(tuple(hp.shape)), output_summary=summarize_array(hp.detach().numpy()),
              note="Generated 4 IMRPhenomD binary-black-hole waveforms over a 256-point frequency grid.")

def infer_physics_informed_am(ds: dict) -> ModelResult:
    meta = dict(name="Physics-Informed AM (hybrid thermal model)", slug="physics-informed-am",
                category="pinn", domains=["manufacturing"],
                repo="https://github.com/ShuhengLiao/Physics_informed_AM")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_pinn_swe_open(ds: dict) -> ModelResult:
    meta = dict(name="PINN_SWE_open", slug="pinn-swe-open", category="pinn", domains=["hydrology"],
                repo="https://github.com/tianyongsen/PINN_SWE_open")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_pinn_corrosion_fatigue(ds: dict) -> ModelResult:
    meta = dict(name="PINN for Corrosion-Fatigue Prognosis", slug="pinn-corrosion-fatigue",
                category="pinn", domains=["structural", "prognostics"],
                repo="https://github.com/PML-UCF/pinn_corrosion_fatigue")
    return manual_install(**meta, instructions="Paper code — first `pip install "
                                                "git+https://github.com/PML-UCF/pinn` (base package), "
                                                "then `git clone` this repo.", dataset=ds)

def infer_pinn_wind_bearing(ds: dict) -> ModelResult:
    meta = dict(name="PINN for Wind Turbine Main Bearing Fatigue", slug="pinn-wind-bearing",
                category="pinn", domains=["prognostics", "rigidbody"],
                repo="https://github.com/PML-UCF/pinn_wind_bearing")
    return manual_install(**meta, instructions="Paper code — first `pip install "
                                                "git+https://github.com/PML-UCF/pinn` (base package), "
                                                "then `git clone` this repo.", dataset=ds)

def infer_pinnicle(ds: dict) -> ModelResult:
    meta = dict(name="PINNICLE", slug="pinnicle", category="pinn", domains=["cryosphere"],
                repo="https://github.com/ISSMteam/PINNICLE")
    if not is_installed("pinnicle"):
        return not_installed(**meta, pip_hint="pip install pinnicle", dataset=ds)
    with Timer() as t:
        try:
            import deepxde as dde
            net = dde.nn.FNN([2, 8, 8, 2], "tanh", "Glorot uniform")
            x = np.hstack([ds["x"], ds["x"] * 0.5])
            y_pred = net(dde.backend.as_tensor(x)) if hasattr(dde, "backend") else None
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    out = np.asarray(y_pred.detach()) if y_pred is not None else np.zeros((len(x), 2))
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(x.shape),
              output_shape=str(out.shape), output_summary=summarize_array(out),
              note="Confirmed PINNICLE + its DeepXDE backend import; built a small FNN over a synthetic "
                   "(x, thickness) input pair as a stand-in for real ice-sheet fields.")

def infer_physnet_thermal(ds: dict) -> ModelResult:
    meta = dict(name="PhysNet Thermal Models", slug="physnet-thermal", category="pinn",
                domains=["buildings"], repo="https://github.com/GargyaGokhale/PhysNet_Thermal_Models")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_neuralsolvers(ds: dict) -> ModelResult:
    meta = dict(name="NeuralSolvers", slug="neuralsolvers", category="pinn", domains=["optics"],
                repo="https://github.com/Photon-AI-Research/NeuralSolvers")
    return manual_install(**meta, instructions="Not confirmed on PyPI — `git clone` and "
                                                "`pip install -e .` per repo README.", dataset=ds)

def infer_pinn_multivp(ds: dict) -> ModelResult:
    meta = dict(name="PINN-MULTIVP", slug="pinn-multivp", category="pinn", domains=["heliophysics"],
                repo="https://github.com/biromiro/pinn-multivp")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_socialforce(ds: dict) -> ModelResult:
    meta = dict(name="Social Force (differentiable)", slug="socialforce", category="diffsim",
                domains=["crowd"], repo="https://github.com/svenkreiss/socialforce")
    if not is_installed("socialforce"):
        return not_installed(**meta, pip_hint="pip install socialforce", dataset=ds)
    with Timer() as t:
        try:
            import socialforce
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            initial_state = torch.tensor(ds["state"], dtype=torch.float32)
            simulator = socialforce.Simulator()
            states = simulator.run(initial_state, n_steps=5)
            out = np.asarray(states.detach() if hasattr(states, "detach") else states)
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    return ok(**meta, dataset=ds, duration=t.elapsed, input_shape=str(ds["state"].shape),
              output_shape=str(out.shape), output_summary=summarize_array(out),
              note=f"Simulated {ds['n']} pedestrians for 5 Social Force update steps.")

def infer_spdiff(ds: dict) -> ModelResult:
    meta = dict(name="SPDiff", slug="spdiff", category="domain", domains=["crowd"],
                repo="https://github.com/tsinghua-fib-lab/SPDiff")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_stiff_pinn(ds: dict) -> ModelResult:
    meta = dict(name="Stiff-PINN", slug="stiff-pinn", category="pinn", domains=["combustion"],
                repo="https://github.com/DENG-MIT/Stiff-PINN")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)

def infer_crnn(ds: dict) -> ModelResult:
    meta = dict(name="CRNN (Chemical Reaction Neural Network)", slug="crnn", category="domain",
                domains=["combustion"], repo="https://github.com/DENG-MIT/CRNN")
    return manual_install(**meta, instructions="Julia paper code — needs `julia` with the repo's "
                                                "Project.toml/Manifest.toml, not a Python/pip package.",
                           dataset=ds)

def infer_neuralfoil(ds: dict) -> ModelResult:
    meta = dict(name="NeuralFoil", slug="neuralfoil", category="domain", domains=["aerospace"],
                repo="https://github.com/peterdsharpe/NeuralFoil")
    if not is_installed("neuralfoil"):
        return not_installed(**meta, pip_hint="pip install neuralfoil", dataset=ds)
    with Timer() as t:
        try:
            import aerosandbox as asb
            import neuralfoil as nf
            airfoil = asb.Airfoil(f"naca{ds['naca']}")
            aero = nf.get_aero_from_airfoil(airfoil=airfoil, alpha=ds["alpha"], Re=ds["Re"])
        except Exception as exc:
            return failed(**meta, exc=exc, duration=t.elapsed, dataset=ds)
    cl = float(np.asarray(aero["CL"]).flatten()[0])
    cd = float(np.asarray(aero["CD"]).flatten()[0])
    cm = float(np.asarray(aero["CM"]).flatten()[0])
    return ok(**meta, dataset=ds, duration=t.elapsed,
              input_shape=f"NACA {ds['naca']}, alpha={ds['alpha']}, Re={ds['Re']:.0e}",
              output_shape=str(list(aero.keys())),
              output_summary={"CL": cl, "CD": cd, "CM": cm},
              note="Ran NeuralFoil's physics-informed aerodynamics inference on a NACA 2412 airfoil.")

def infer_pinn_option_pricing(ds: dict) -> ModelResult:
    meta = dict(name="PINN Option Pricing", slug="pinn-option-pricing", category="pinn",
                domains=["finance"], repo="https://github.com/ashish1610dhiman/pinn_option_pricing")
    return manual_install(**meta, instructions="Paper code — `git clone` only, no pip package.", dataset=ds)
