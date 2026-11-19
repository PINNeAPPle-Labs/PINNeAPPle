import unittest

from pinneapple_core.workflows.experiment_blueprint import (
    ExperimentRequest, build_experiment_workflow,
)


class ExperimentBlueprintTests(unittest.TestCase):
    def test_both_domains_build_valid_dags(self):
        for domain in ("industrial", "research"):
            workflow = build_experiment_workflow(
                ExperimentRequest(objective="Evaluate candidate designs", domain=domain)
            )
            self.assertEqual(len(workflow.steps), 8)
            self.assertEqual(workflow.order()[0], "scope")
            self.assertEqual(workflow.order()[-1], "report")
            self.assertEqual(workflow.metadata["execution_mode"], "plan_only")

    def test_empty_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_workflow(ExperimentRequest(objective=" ", domain="research"))

    def test_approval_and_evidence_gates(self):
        workflow = build_experiment_workflow(ExperimentRequest(objective="Test", domain="industrial"))
        self.assertTrue(workflow.step("optimize").params["requires_approval"])
        self.assertTrue(workflow.step("report").params["requires_verified_evidence"])


if __name__ == "__main__":
    unittest.main()
