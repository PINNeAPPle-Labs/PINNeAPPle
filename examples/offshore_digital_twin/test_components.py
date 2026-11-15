import unittest
from solvers import DrillingState, drillstring_power, riser_static_extension
from surrogate import NearestNeighborSurrogate
from scenarios import generate_scenarios
from scenario_library import run_scenario
from benchmark import run_benchmarks

class ComponentTests(unittest.TestCase):
    def test_drilling_power(self):
        self.assertGreater(drillstring_power(DrillingState())["rotational_power_w"],0)
    def test_riser_extension(self):
        self.assertAlmostEqual(riser_static_extension(10,100,1000,1)["extension_m"],1)
    def test_surrogate_rejects_outside(self):
        model=NearestNeighborSurrogate().fit(generate_scenarios(n=10))
        self.assertFalse(model.predict(10000,.025)["accepted"])
    def test_scenario(self):
        self.assertNotEqual(run_scenario("nominal")["results"]["bottomhole_circulating_pressure_pa"],
                            run_scenario("reduced_density")["results"]["bottomhole_circulating_pressure_pa"])
    def test_benchmarks(self):
        self.assertTrue(run_benchmarks()["passed"])
if __name__ == "__main__":
    unittest.main()
