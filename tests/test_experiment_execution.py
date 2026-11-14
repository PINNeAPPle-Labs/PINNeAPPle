import tempfile
import unittest

from pinneapple_core.workflows.experiment_blueprint import (
    ExperimentRequest, build_experiment_workflow,
)
from pinneapple_core.workflows.experiment_execution import (
    STAGES, executable_workflow, run_experiment,
)


class ExecutionTests(unittest.TestCase):
    def blueprint(self):
        return build_experiment_workflow(
            ExperimentRequest(objective="Compare two candidate designs", domain="industrial")
        )

    def test_conversion_enforces_evidence_dependency(self):
        workflow = executable_workflow(self.blueprint())
        self.assertEqual(workflow.step("report").needs, ["optimize", "verify"])
        self.assertTrue(all(s.kind == "experiment_stage" for s in workflow.steps))

    def test_trusted_handlers_run_and_persist(self):
        def make_handler(stage):
            return lambda params, ctx: {"stage": stage, **({"approved": True} if stage == "verify" else {})}

        handlers = {stage: make_handler(stage) for stage in STAGES}
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(self.blueprint(), handlers, root)
            self.assertEqual(result["status"], "done")

    def test_rejected_verification_stops_downstream(self):
        called = []
        def make_handler(stage):
            def handle(params, ctx):
                called.append(stage)
                return {"approved": False} if stage == "verify" else {"stage": stage}
            return handle
        handlers = {stage: make_handler(stage) for stage in STAGES}
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(self.blueprint(), handlers, root)
            self.assertNotEqual(result["status"], "done")
            self.assertNotIn("optimize", called)
            self.assertNotIn("report", called)


if __name__ == "__main__":
    unittest.main()
