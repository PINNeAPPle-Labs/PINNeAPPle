import unittest

from pinneapple_core.workflows.evidence_report import build_evidence_report


class EvidenceReportTests(unittest.TestCase):
    def test_unverified_run_is_not_approved(self):
        run = {"id": "run-1", "name": "research_experiment", "status": "failed",
               "workflow": {"metadata": {"objective": "Test heat transfer"}}}
        report = build_evidence_report(run, [
            {"step_id": "verify", "status": "failed",
             "outputs": {"approved": True}, "hash": "abc", "seconds": 0.1}
        ])
        self.assertFalse(report["verification"]["approved"])
        self.assertEqual(len(report["sha256"]), 64)

    def test_successful_evidence_is_explicit(self):
        run = {"id": "run-2", "name": "industrial_experiment", "status": "done",
               "workflow": {"metadata": {"objective": "Reduce energy"}}}
        report = build_evidence_report(run, [
            {"step_id": "verify", "status": "done",
             "outputs": {"approved": True, "decision": "APPROVED"}}
        ])
        self.assertTrue(report["verification"]["approved"])


if __name__ == "__main__":
    unittest.main()
