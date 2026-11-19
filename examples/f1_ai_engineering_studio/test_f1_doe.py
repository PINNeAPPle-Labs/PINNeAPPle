import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from f1_doe import generate_variants, export_variants, PARAMETERS

class F1DOE(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(generate_variants(10, 42), generate_variants(10, 42))
        self.assertNotEqual(generate_variants(10, 42), generate_variants(10, 43))
    def test_bounds(self):
        variants = generate_variants()
        self.assertEqual(len(variants), 10)
        for row in variants:
            for key, (low, high) in PARAMETERS.items():
                self.assertTrue(low <= row[key] <= high)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            generate_variants(0)
    def test_export(self):
        with TemporaryDirectory() as folder:
            export_variants(folder)
            self.assertTrue((Path(folder) / "variants.csv").exists())
            self.assertTrue((Path(folder) / "manifest.json").exists())

if __name__ == "__main__":
    unittest.main()
