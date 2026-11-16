import math
import unittest
from benchmark import exact, forcing, evaluate


class AnalyticalReferenceTests(unittest.TestCase):
    def test_boundary_conditions(self):
        self.assertAlmostEqual(exact(0), 0, places=12)
        self.assertAlmostEqual(exact(1), 0, places=12)

    def test_source(self):
        self.assertAlmostEqual(forcing(0.5), math.pi ** 2, places=12)

    def test_residual_convergence(self):
        coarse = evaluate(n=101, h=1e-2)["rms_residual"]
        fine = evaluate(n=101, h=1e-3)["rms_residual"]
        self.assertLess(fine, coarse / 50)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            evaluate(n=2)


if __name__ == "__main__":
    unittest.main()
