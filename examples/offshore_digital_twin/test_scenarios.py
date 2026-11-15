import unittest
from offshore_twin import WellConfig, evaluate
from scenarios import generate_scenarios, manifest
from verification import audit

class ScenarioTests(unittest.TestCase):
    def test_reproducible(self):
        a = generate_scenarios(n=10, seed=7)
        b = generate_scenarios(n=10, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(manifest(a, seed=7), manifest(b, seed=7))
    def test_splits(self):
        rows = generate_scenarios(n=10)
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})
    def test_consistency(self):
        self.assertTrue(audit(WellConfig(), evaluate(WellConfig()))["checks"]["hydrostatic_identity"])
    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            generate_scenarios(n=0)
if __name__ == "__main__":
    unittest.main()
