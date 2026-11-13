"""Regression tests for solver-neutral contract and canal sampling methodology."""
from pathlib import Path
import pytest

from pinneapple_analysis.spatial_sampling import (
    SpatialReference, SamplingProtocol, IncompatibleComparison,
    absolute_distance, compare_protocols, require_compatible, liquid_peak,
)
from pinneapple_simulation.external_solvers.generic_contract import (
    SimulationSpec, SimulationError, UnsupportedSimulation, get_adapter, simulate,
)


def protocol(**changes):
    ref = SpatialReference("inlet", (-7.429,), ("y",))
    base = dict(reference=ref, field="k", field_units="m2/s2", statistic="peak",
                window_s=(20, 30), phase="liquid", phase_alpha_min=0.9)
    base.update(changes)
    return SamplingProtocol(**base)


@pytest.mark.parametrize("offset", [1.364, 1.819, 3.637])
def test_absolute_origin_is_not_first_cell(offset):
    inlet = -7.429
    assert absolute_distance(inlet + offset, inlet) == pytest.approx(offset)
    assert absolute_distance(inlet, inlet) == 0


def test_compatible_comparison():
    assert compare_protocols(protocol(), protocol()).allowed
    require_compatible(protocol(), protocol())


@pytest.mark.parametrize("change", [
    {"statistic": "mean"}, {"window_s": (0, 30)},
    {"phase": "air"}, {"phase_alpha_min": None},
    {"field_units": "Pa"}, {"spatial_interval_m": (2, 8)},
    {"reference": SpatialReference("first_sample", (0,), ("y",))},
    {"reference": SpatialReference("inlet", (-7.429,), ("y",), location="node")},
])
def test_incompatible_sampling_is_blocked(change):
    decision = compare_protocols(protocol(), protocol(**change))
    assert not decision.allowed and decision.reasons
    with pytest.raises(IncompatibleComparison):
        require_compatible(protocol(), protocol(**change))


def test_peak_ignores_air_and_nonfinite():
    assert liquid_peak([2, 100, 4, float("nan")], [1, 0.1, 0.95, 1]) == (2, 4.0)
    with pytest.raises(IncompatibleComparison):
        liquid_peak([100], [0.1])


def test_fenicsx_requires_trusted_callable(tmp_path):
    spec = SimulationSpec("fenicsx", tmp_path)
    with pytest.raises(UnsupportedSimulation):
        simulate(spec)
    result = simulate(spec, fenicsx_solve=lambda s: [1.0])
    assert result.status == "completed" and result.backend == "fenicsx"
    with pytest.raises(SimulationError):
        simulate(spec, fenicsx_solve=lambda s: None)


def test_openfoam_fails_closed_without_case(tmp_path):
    with pytest.raises(SimulationError, match="controlDict"):
        simulate(SimulationSpec("openfoam", tmp_path))


def test_calculix_fails_closed_without_deck(tmp_path):
    with pytest.raises(SimulationError, match="input deck"):
        simulate(SimulationSpec("calculix", tmp_path, case="beam"))


def test_unknown_backend_and_unsafe_job(tmp_path):
    with pytest.raises(UnsupportedSimulation):
        get_adapter("unknown")
    with pytest.raises(ValueError):
        simulate(SimulationSpec("calculix", tmp_path, case="../evil"))
