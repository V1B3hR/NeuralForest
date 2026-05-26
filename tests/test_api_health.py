import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import NeuralForestAPI
from NeuralForest import ForestEcosystem


class TestAPIHealth(unittest.TestCase):
    def _make_api(self):
        forest = ForestEcosystem(input_dim=4, max_trees=4).to(torch.device("cpu"))
        return NeuralForestAPI(forest=forest, device=torch.device("cpu"))

    def test_empty_forest_status_schema_is_stable(self):
        api = self._make_api()
        api.forest.trees = nn.ModuleList()
        api.forest.graph.clear()

        status = api.get_forest_status()

        self.assertEqual(status["status"], "empty")
        self.assertEqual(status["num_trees"], 0)
        self.assertIn("memory_usage", status)
        self.assertIn("tree_health", status)
        self.assertEqual(status["tree_health"]["average_fitness"], 0.0)
        self.assertEqual(status["tree_details"], [])

        health = api.health_check()
        self.assertIn("no_trees", health["issues"])
        self.assertEqual(health["status"]["num_trees"], 0)

    def test_predict_clamps_top_k_and_rejects_malformed_input(self):
        api = self._make_api()
        captured = {}

        def fake_forward_forest(x, top_k=3):
            captured["top_k"] = top_k
            return (
                torch.zeros(x.shape[0], 1),
                torch.tensor([[1.0]], dtype=torch.float32),
                [torch.zeros(x.shape[0], 1)],
            )

        api.forest.forward_forest = fake_forward_forest

        result = api.predict({"input": [1.0, 2.0, 3.0, 4.0]}, top_k=99)
        self.assertIn("prediction", result)
        self.assertEqual(captured["top_k"], api.forest.num_trees())

        bad = api.predict({"input": "not-a-tensor"})
        self.assertFalse(bad["success"])
        self.assertIn("Unable to parse input tensor", bad["error"])


if __name__ == "__main__":
    unittest.main()
