"""End-to-end reference workflow tests using the real SQLite runner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pinneapple_core.workflows.engine import WorkflowRunner
from pinneapple_core.workflows.evidence_report import export_evidence_report
from pinneapple_core.workflows.experiment_blueprint import ExperimentRequest, build_experiment_workflow
from pinneapple_core.workflows.experiment_execution import STAGES, run_experiment
from pinneapple_core.workflows.reference_conduction import reference_handlers


class ExperimentEndToEndTests(unittest.TestCase):
    def blueprint(self, length=2.0):
        return build_experiment_workflow(ExperimentRequest(
            objective="Verify a Fourier conduction baseline",
            domain="research",
            constraints={"length_m": length, "conductivity_W_mK": 15.0,
                         "left_temperature_K": 400.0, "right_temperature_K": 300.0,
                         "cells": 10},
        ))

    def test_reference_run_and_evidence_export(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(self.blueprint(), reference_handlers(), root)
            self.assertEqual(result["status"], "done")
            runner = WorkflowRunner(root)
            path = export_evidence_report(runner, result["id"], Path(root) / "report.json")
            import json
            report = json.loads(path.read_text())
            self.assertTrue(report["verification"]["approved"])
            self.assertEqual(len(report["steps"]), len(STAGES))
            self.assertEqual(report["run_status"], "done")
            self.assertEqual(len(report["sha256"]), 64)

    def test_invalid_physics_never_approves(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(self.blueprint(length=-1), reference_handlers(), root)
            self.assertNotEqual(result["status"], "done")
            runner = WorkflowRunner(root)
            path = export_evidence_report(runner, result["id"], Path(root) / "rejected.json")
            import json
            report = json.loads(path.read_text())
            self.assertFalse(report["verification"]["approved"])

    def test_verification_rejection_blocks_downstream(self):
        handlers = reference_handlers()
        handlers["verify"] = lambda params, ctx: {"approved": False, "reason": "outside validated domain"}
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(self.blueprint(), handlers, root)
            steps = {s["step_id"]: s for s in WorkflowRunner(root).steps(result["id"])}
            self.assertEqual(steps["verify"]["status"], "failed")
            self.assertNotEqual(steps["optimize"]["status"], "done")
            self.assertNotEqual(steps["report"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
