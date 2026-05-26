import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canopy import ForestCanopy


class DummyGrove(nn.Module):
    def __init__(self, value: float, fail: bool = False):
        super().__init__()
        self.value = value
        self.fail = fail

    def forward(self, x, top_k=3):
        if self.fail:
            raise RuntimeError("grove failure")

        output = torch.full((x.shape[0], 1), self.value, device=x.device)
        weights = torch.ones(x.shape[0], 1, device=x.device)
        return output, weights


class TestCanopyBatchRouting(unittest.TestCase):
    def test_per_sample_routing_selects_individual_groves(self):
        canopy = ForestCanopy(
            grove_dict={
                "image": DummyGrove(1.0),
                "audio": DummyGrove(2.0),
            },
            embedding_dim=4,
            num_heads=1,
        )
        canopy.grove_router.forward = lambda x: torch.tensor(
            [[0.1, 0.9], [0.9, 0.1]], device=x.device
        )

        x = torch.randn(2, 4)
        output, routing_info = canopy(x, modality_hint="image", top_k_groves=1)

        self.assertTrue(torch.allclose(output.squeeze(-1), torch.tensor([2.0, 1.0])))
        self.assertEqual(routing_info["failure_count"], 0)
        self.assertEqual(routing_info["groves_used"], ["audio", "image"])

    def test_failed_grove_is_logged_and_reported(self):
        canopy = ForestCanopy(
            grove_dict={
                "image": DummyGrove(1.0),
                "audio": DummyGrove(2.0, fail=True),
            },
            embedding_dim=4,
            num_heads=1,
        )
        canopy.grove_router.forward = lambda x: torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]], device=x.device
        )

        x = torch.randn(2, 4)
        with self.assertLogs("canopy.hierarchical_router", level="WARNING") as logs:
            output, routing_info = canopy(x, modality_hint="image", top_k_groves=2)

        self.assertTrue(any("failed for sample" in line for line in logs.output))
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(routing_info["failure_count"], 2)
        self.assertTrue(torch.allclose(output, torch.ones(2, 1)))


if __name__ == "__main__":
    unittest.main()
