import unittest

import torch

from roots.unified_backbone import SimpleRootNetwork


class TestSimpleRootNetwork(unittest.TestCase):
    def setUp(self):
        self.modality_dims = {"image": 4, "audio": 4}
        self.model = SimpleRootNetwork(embedding_dim=8, modality_dims=self.modality_dims)
        self.model.eval()

    def test_fusion_is_order_invariant(self):
        inputs_a = {
            "image": torch.randn(2, 4),
            "audio": torch.randn(2, 4),
        }
        inputs_b = {
            "audio": inputs_a["audio"],
            "image": inputs_a["image"],
        }

        out_a = self.model(inputs_a)
        out_b = self.model(inputs_b)

        self.assertTrue(torch.allclose(out_a, out_b))

    def test_unknown_modality_warns_and_is_ignored(self):
        inputs = {
            "image": torch.randn(2, 4),
            "unknown": torch.randn(2, 4),
        }

        with self.assertWarnsRegex(
            UserWarning, "Unknown modality 'unknown' will be ignored."
        ):
            output = self.model(inputs)

        self.assertEqual(output.shape, (2, 8))

    def test_all_unknown_modalities_raise_error(self):
        with self.assertWarnsRegex(
            UserWarning, "Unknown modality 'unknown' will be ignored."
        ):
            with self.assertRaisesRegex(ValueError, "No valid modality inputs found"):
                self.model({"unknown": torch.randn(2, 4)})

    def test_single_modality_uses_fusion_mlp(self):
        fusion_calls = {"count": 0}

        def _count_calls(module, module_inputs, module_output):
            fusion_calls["count"] += 1

        handle = self.model.fusion_mlp.register_forward_hook(_count_calls)
        try:
            _ = self.model({"image": torch.randn(2, 4)})
        finally:
            handle.remove()

        self.assertEqual(fusion_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
