import json
import tempfile
import unittest
from pathlib import Path
from offshore_twin import WellConfig
from transient import TransientConfig, simulate
from export import export_transient

class ExportTests(unittest.TestCase):
    def test_json_csv(self):
        rows = simulate(WellConfig(), TransientConfig(steps=2))
        with tempfile.TemporaryDirectory() as tmp:
            jp = export_transient(rows, Path(tmp) / "run.json")
            cp = export_transient(rows, Path(tmp) / "run.csv")
            self.assertEqual(len(json.loads(jp.read_text())["records"]), 3)
            self.assertEqual(len(cp.read_text().splitlines()), 4)
    def test_reject_unknown_format(self):
        with self.assertRaises(ValueError):
            export_transient([], "run.vtk")
if __name__ == "__main__":
    unittest.main()
