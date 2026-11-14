import tempfile
import unittest

from pinneapple_core.workflows.experiment_blueprint import ExperimentRequest, build_experiment_workflow
from pinneapple_core.workflows.experiment_execution import run_experiment
from pinneapple_core.workflows.reference_conduction import reference_handlers


class ConductionSmokeTest(unittest.TestCase):
    def test_end_to_end_reference_simulation(self):
        blueprint = build_experiment_workflow(ExperimentRequest(
            objective="Verify steady heat conduction",
            domain="research",
            constraints={"length_m": 2, "conductivity_W_mK": 15,
                         "left_temperature_K": 400, "right_temperature_K": 300,
                         "cells": 10},
        ))
        with tempfile.TemporaryDirectory() as root:
            result = run_experiment(blueprint, reference_handlers(), root)
            self.assertEqual(result["status"], "done")


if __name__ == "__main__":
    unittest.main()
