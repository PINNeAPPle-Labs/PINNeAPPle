import unittest
from offshore_twin import WellConfig, evaluate

class OffshoreTwinTests(unittest.TestCase):
    def test_zero_flow_hydrostatic(self):
        c = WellConfig(flow_m3_s=0)
        x = evaluate(c)
        self.assertAlmostEqual(x["bottomhole_circulating_pressure_pa"], x["hydrostatic_pa"])
        self.assertEqual(x["pump_differential_pa"], 0)
    def test_flow_increases_ecd(self):
        low = evaluate(WellConfig(flow_m3_s=0.01))
        high = evaluate(WellConfig(flow_m3_s=0.03))
        self.assertGreater(high["equivalent_circulating_density_kg_m3"], low["equivalent_circulating_density_kg_m3"])
    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            evaluate(WellConfig(hole_diameter_m=0.10))
    def test_invalid_pressure(self):
        with self.assertRaises(ValueError):
            evaluate(WellConfig(pore_pressure_pa=50e6))
if __name__ == "__main__":
    unittest.main()
