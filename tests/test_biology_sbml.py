"""SBML import and export: every shipped BioModels entry against libRoadRunner (live when installed, else its stored
reference), MathML coverage, the refusals, and the round trip ODESystem -> SBML -> ODESystem."""
import json
import math

import numpy as np
import pytest
import sympy as sp

from pinneapple_physics.biology import (
    ODESystem,
    SBMLUnsupported,
    biomodel,
    list_biomodels,
    read_sbml,
    write_sbml,
)
from pinneapple_physics.biology.sbml import BIOMODELS_DIR

REF = json.loads((BIOMODELS_DIR / "roadrunner_reference.json").read_text())


@pytest.mark.parametrize("mid", sorted(REF["models"]))
def test_biomodels_match_roadrunner_reference(mid):
    s = biomodel(mid)
    t = np.asarray(REF["t"])
    tr = s.simulate(t, rtol=1e-11, atol=1e-14)
    assert tr.success
    for name, ref in REF["models"][mid].items():
        ref = np.asarray(ref)
        err = np.linalg.norm(tr[name] - ref) / max(np.linalg.norm(ref), 1e-30)
        assert err < 1e-6, (mid, name, err)


def test_live_roadrunner_when_installed():
    roadrunner = pytest.importorskip("roadrunner")
    s = biomodel(12)
    r = roadrunner.RoadRunner(str(BIOMODELS_DIR / "BIOMD0000000012.xml"))
    r.integrator.relative_tolerance, r.integrator.absolute_tolerance = 1e-10, 1e-12
    res = r.simulate(0, 200, 101, selections=["time"] + [f"[{x}]" for x in s.states])
    tr = s.simulate(res[:, 0], rtol=1e-11, atol=1e-14)
    for i, x in enumerate(s.states):
        assert np.linalg.norm(tr[x] - res[:, i + 1]) / np.linalg.norm(res[:, i + 1]) < 1e-6


def test_shipped_models_and_structure():
    assert len(list_biomodels()) == 6
    s = biomodel(12)                                  # repressilator: 9 assignment rules feed the rates
    assert s.states == ["PX", "PY", "PZ", "X", "Y", "Z"]
    assert list(s.assignments)[:2] == ["t_ave", "beta"]
    assert set(s.assignments) >= {"k_tl", "alpha", "kd_mRNA"}
    k = biomodel(5)                                   # local kinetic-law parameters are renamed per reaction
    assert "Reaction4_k3" in k.parameters


@pytest.mark.parametrize("mid", sorted(REF["models"]))
def test_export_round_trip(mid):
    s = biomodel(mid)
    s2 = read_sbml(s.to_sbml())
    t = np.linspace(0, 50, 26)
    a, b = s.simulate(t), s2.simulate(t)
    for x in s.states:
        np.testing.assert_allclose(a[x], b[x], rtol=1e-10, atol=1e-14)


def test_export_validates_with_libsbml_when_installed():
    pytest.importorskip("libsbml")
    read_sbml(biomodel(21).to_sbml(), validate=True)


SMALL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfFunctionDefinitions>
   <functionDefinition id="mm"><math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
     <bvar><ci>s</ci></bvar><bvar><ci>v</ci></bvar><bvar><ci>k</ci></bvar>
     <apply><divide/><apply><times/><ci>v</ci><ci>s</ci></apply><apply><plus/><ci>k</ci><ci>s</ci></apply></apply>
   </lambda></math></functionDefinition>
  </listOfFunctionDefinitions>
  <listOfCompartments><compartment id="V" size="2" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="V" initialAmount="4" hasOnlySubstanceUnits="false" boundaryCondition="false"
            constant="false"/>
   <species id="P" compartment="V" initialConcentration="0" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="Vmax" value="1.5" constant="true"/>
   <parameter id="pulse" value="0" constant="false"/>
  </listOfParameters>
  <listOfRules>
   <assignmentRule variable="pulse"><math xmlns="http://www.w3.org/1998/Math/MathML"><piecewise>
     <piece><cn>1</cn><apply><lt/><csymbol encoding="text"
       definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>1</cn></apply></piece>
     <otherwise><cn>0</cn></otherwise></piecewise></math></assignmentRule>
  </listOfRules>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="P" stoichiometry="2" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>V</ci><apply><ci>mm</ci><ci>S</ci><ci>Vmax</ci><ci>Km</ci></apply></apply></math>
     <listOfLocalParameters><localParameter id="Km" value="0.5"/></listOfLocalParameters>
    </kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


def test_small_model_semantics():
    s = read_sbml(SMALL)
    assert s.states == ["S", "P"]
    np.testing.assert_allclose(s.initial_values(), [2.0, 0.0])          # amount 4 in V = 2 -> concentration 2
    x = {sp.Symbol("S", real=True): 1.0, sp.Symbol("P", real=True): 0.0}
    rate = float(s.expanded(s.rhs["S"]).subs(x).subs({sp.Symbol(k, real=True): v for k, v in s.parameters.items()}))
    assert rate == pytest.approx(-1.5 * 1.0 / (0.5 + 1.0))           # V * v / V
    tr = s.simulate(np.linspace(0, 5, 51))
    np.testing.assert_allclose(tr["P"], 2 * (2.0 - tr["S"]), atol=1e-8)  # stoichiometry 2 conserves 2S + P
    assert "R_Km" in s.parameters
    np.testing.assert_allclose(tr["pulse"][[0, -1]], [1.0, 0.0])


def test_refusals():
    events = SMALL.replace("</listOfReactions>", """</listOfReactions><listOfEvents><event id="e" useValuesFromTriggerTime="true">
      <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><gt/><ci>S</ci><cn>1</cn></apply></math></trigger></event></listOfEvents>""")
    with pytest.raises(SBMLUnsupported, match="events"):
        read_sbml(events)
    with pytest.raises(SBMLUnsupported, match="algebraic"):
        read_sbml(SMALL.replace("<listOfRules>", """<listOfRules><algebraicRule><math
            xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>S</ci><cn>1</cn></apply></math>
            </algebraicRule>"""))


def test_ode_system_from_strings_and_sensitivities():
    s = ODESystem.from_strings("decay", {"x": "-k*x"}, {"x": "x0"}, {"k": 0.7, "x0": 2.0})
    t = np.linspace(0, 3, 31)
    traj, S = s.sensitivities(t, ["k", "x_0"])
    np.testing.assert_allclose(traj["x"], 2 * np.exp(-0.7 * t), rtol=1e-6)
    np.testing.assert_allclose(S[:, 0, 0], -t * 2 * np.exp(-0.7 * t), rtol=1e-5, atol=1e-9)   # dx/dk
    np.testing.assert_allclose(S[:, 0, 1], np.exp(-0.7 * t), rtol=1e-6)                       # dx/dx0
    assert "<rateRule" in write_sbml(s)


def test_hill_jacobian_is_finite_at_zero():
    s = ODESystem.from_strings("hill", {"x": "a/(1+y**n) - x", "y": "x - y"}, {"x": 0.0, "y": 0.0},
                               {"a": 2.0, "n": 2.5})
    tr = s.simulate(np.linspace(0, 10, 11))
    assert tr.success and all(math.isfinite(v) for v in tr["x"])
