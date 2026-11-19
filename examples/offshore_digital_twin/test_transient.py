import math
import unittest
from offshore_twin import WellConfig, evaluate
from transient import TransientConfig, simulate

class TransientTests(unittest.TestCase):
    def test_exact_relaxation(self):
        w = WellConfig()
        target = evaluate(w)["bottomhole_circulating_pressure_pa"]
        initial = target - 1e6
        t = TransientConfig(dt_s=2, steps=20, relaxation_s=10,
                            initial_pressure_pa=initial)
        rows = simulate(w, t)
        self.assertEqual(len(rows), 21)
        for row in rows:
            exact = target + (initial-target)*math.exp(-row["time_s"]/10)
            self.assertAlmostEqual(row["bottomhole_pressure_pa"], exact, places=5)
    def test_equilibrium_is_fixed(self):
        w = WellConfig()
        rows = simulate(w)
        self.assertTrue(all(r["bottomhole_pressure_pa"] == rows[0]["bottomhole_pressure_pa"] for r in rows))
    def test_bad_dt(self):
        with self.assertRaises(ValueError):
            simulate(WellConfig(), TransientConfig(dt_s=0))
    def test_bad_nan(self):
        with self.assertRaises(ValueError):
            simulate(WellConfig(), TransientConfig(initial_pressure_pa=float("nan")))
if __name__ == "__main__":
    unittest.main()
