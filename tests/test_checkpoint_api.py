import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import ForestCheckpoint
from NeuralForest import ForestEcosystem


class TestCheckpointAPI(unittest.TestCase):
    def test_save_preserves_metadata_without_readback(self):
        forest = ForestEcosystem(input_dim=4, max_trees=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "forest.pt")

            with patch("api.forest_api.torch.load") as mocked_load:
                ForestCheckpoint.save(forest, path, metadata={"source": "unit-test"})
                mocked_load.assert_not_called()

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["metadata"]["source"], "unit-test")

    def test_validate_and_get_info_handle_malformed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "malformed.pt")
            torch.save({"input_dim": 4}, path)

            self.assertFalse(ForestCheckpoint.validate(path))

            info = ForestCheckpoint.get_info(path)
            self.assertFalse(info["valid"])
            self.assertEqual(info["num_trees"], 0)

    def test_round_trip_checkpoint_load(self):
        forest = ForestEcosystem(input_dim=4, max_trees=4)
        forest._plant_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.pt")
            ForestCheckpoint.save(forest, path, metadata={"env": "test"})

            self.assertTrue(ForestCheckpoint.validate(path))

            info = ForestCheckpoint.get_info(path)
            self.assertTrue(info["valid"])
            self.assertEqual(info["metadata"]["env"], "test")

            loaded = ForestCheckpoint.load(path, device=torch.device("cpu"))
            self.assertEqual(loaded.num_trees(), forest.num_trees())


if __name__ == "__main__":
    unittest.main()
