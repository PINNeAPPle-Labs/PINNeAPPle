import math
import unittest

from benchmark import evaluate, exact, forcing


class AnalyticalReferenceTests(unittest.TestCase):
    def test_boundary_conditions(self):
        self.assertAlmostEqual(exact(0), 0, places=12)
        self.assertAlmostEqual(exact(1), 0, places=12)

    def test_source(self):
        self.assertAlmostEqual(forcing(0.5), math.pi ** 2, places=12)

    def test_residual_convergence(self):
        """Second-order convergence, measured rather than assumed.

        The residual is the truncation error of the central second difference applied to the exact
        solution, so it must fall as h^2: a step five times smaller leaves a residual 25 times smaller.
        The previous version of this test asked for a factor of 50, which the operator cannot deliver
        and which therefore failed on a correct implementation.
        """
        coarse = evaluate(n=101, h=5e-3)["rms_residual"]
        fine = evaluate(n=101, h=1e-3)["rms_residual"]
        ratio = coarse / fine
        self.assertGreater(ratio, 20)
        self.assertLess(ratio, 32)

    def test_residual_is_the_truncation_error_of_the_operator(self):
        """Against the analytic truncation error (h^2/12) * u'''' , which is pi^4 sin(pi x) here."""
        h = 1e-3
        got = evaluate(n=101, h=h)["rms_residual"]
        want = (h ** 2 / 12) * math.pi ** 4 / math.sqrt(2)      # RMS of sin over (0, 1) is 1/sqrt(2)
        self.assertAlmostEqual(got / want, 1.0, delta=0.02)     # the O(h^4) term and the discrete RMS

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            evaluate(n=2)


if __name__ == "__main__":
    unittest.main()
