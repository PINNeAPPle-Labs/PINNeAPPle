"""The chosen preset is checked against the description before training (PINNeAPPle-apps#3): the guardrail re-checks
the physics of the preset it is given, never whether it is the right one, so "heat conduction" solved with Burgers
passed as trustworthy."""
import pytest

import pinneapple as pp
from pinneapple_llm import check_preset_coherence

WRONG = [                                                    # description, preset an LLM could pick, what is wrong
    ("heat conduction in a steel rod with fixed end temperatures", "burgers_1d", "heat"),
    ("a vibrating string fixed at both ends", "burgers_1d", "waves"),
    ("stress in a loaded cantilever plate", "poisson_2d", "structures"),
    ("airflow around a car", "plane_stress_2d", "fluid"),
    ("electrostatic potential in a two-dielectric capacitor", "sir_epidemic", "electrostatics"),
    ("condução de calor em uma barra de aço", "burgers_1d", "heat"),
]
RIGHT = [
    ("heat conduction in a steel rod with fixed end temperatures", "transient_heat_3d"),
    ("steady heat conduction in a 3D block", "steady_heat_conduction_3d"),
    ("ultrasound waves in a 3D solid", "wave_ultrasound_3d"),
    ("stress in a loaded cantilever plate", "plane_stress_2d"),
    ("airflow around a car", "car_external_aero"),
    ("electrostatic potential in a two-dielectric capacitor", "em_two_dielectric_capacitor"),
    ("an epidemic outbreak in a city", "sir_epidemic"),
    ("orbit of a satellite around the Earth with J2", "satellite_j2_perturbation"),
    ("pricing a European option", "black_scholes_1d"),
    ("viscous shock in 1D", "burgers_1d"),
    ("escoamento de ar em um duto 3D", "channel_flow_3d"),
]


@pytest.mark.parametrize("text,preset,cls", WRONG)
def test_a_preset_of_other_physics_is_incoherent(text, preset, cls):
    v = check_preset_coherence(text, pp.get_preset(preset))
    assert v.status == "incoherent" and not v.coherent and cls in v.description_classes
    assert v.reasons and preset.split("_")[0] in v.reasons[0]


@pytest.mark.parametrize("text,preset", RIGHT)
def test_the_right_preset_is_coherent(text, preset):
    v = check_preset_coherence(text, pp.get_preset(preset))
    assert v.status == "coherent", v.reasons


def test_regime_and_dimension_are_checked():
    v = check_preset_coherence("the temperature in a 2D plate at steady state", pp.get_preset("transient_heat_3d"))
    assert v.status == "incoherent" and len(v.reasons) == 2                     # steady vs t, and 2-D vs 3 axes
    assert check_preset_coherence("heat in a rod that cools down over time",
                                  pp.get_preset("steady_heat_conduction_3d")).status == "incoherent"


def test_no_recognised_physics_is_not_a_confirmation():
    v = check_preset_coherence("I want to start a standard thing", pp.get_preset("laplace_2d"))
    assert v.status == "unknown" and not v.coherent and v.reasons
    assert v.to_dict()["status"] == "unknown"
