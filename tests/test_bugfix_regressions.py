import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import NeuralForest
from NeuralForest import ForestEcosystem, ForestTeacher, TreeExpert, TreeArch
from ecosystem_simulation import EcosystemSimulator
from evolution.tree_graveyard import TreeGraveyard
from training_demos.cifar10_full_training import (
    PrioritizedMulch,
    rebuild_task_head,
    TreeNet,
)


class TestBugfixRegressions(unittest.TestCase):
    def test_module_import_has_no_global_demo_tensors(self):
        self.assertFalse(hasattr(NeuralForest, "X"))
        self.assertFalse(hasattr(NeuralForest, "X_plot"))

    def test_cifar_treenet_loop_indentation_builds_all_layers(self):
        arch = TreeArch(
            num_layers=3,
            hidden_dim=16,
            activation="relu",
            dropout=0.0,
            normalization="none",
            residual=False,
        )
        tree = TreeNet(input_dim=8, arch=arch)
        linear_layers = [m for m in tree.net if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 3)
        self.assertEqual(tree.out.in_features, arch.hidden_dim)

    def test_cifar_mulch_uses_fifo_eviction(self):
        mulch = PrioritizedMulch(capacity=2)
        x = torch.randn(4)
        y = torch.randn(1)
        mulch.add(x, y, 1.0)
        mulch.add(x, y, 2.0)
        mulch.add(x, y, 3.0)
        self.assertEqual(len(mulch), 2)
        priorities = [item[2] for item in mulch.buffer]
        self.assertEqual(priorities, [2.0, 3.0])

    def test_rebuild_task_head_tracks_tree_count(self):
        forest = ForestEcosystem(input_dim=4, max_trees=5)
        forest._plant_tree()
        forest._plant_tree()
        args = SimpleNamespace(output_dim_per_tree=3)
        head = rebuild_task_head(forest, args, torch.device("cpu"))
        self.assertEqual(head.fc1.in_features, 9)

        forest._plant_tree()
        head = rebuild_task_head(forest, args, torch.device("cpu"))
        self.assertEqual(head.fc1.in_features, 12)

    def test_save_checkpoint_path_without_directory(self):
        forest = ForestEcosystem(input_dim=4, max_trees=4)
        forest._plant_tree()
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                forest.save_checkpoint("forest.pt")
                self.assertTrue(os.path.exists("forest.pt"))
            finally:
                os.chdir(old_cwd)

    def test_load_checkpoint_sets_weights_only_flag(self):
        with patch("NeuralForest.torch.load", side_effect=RuntimeError("stop")) as mocked:
            with self.assertRaises(RuntimeError):
                ForestEcosystem.load_checkpoint("dummy.pt", device=torch.device("cpu"))
        self.assertIn("weights_only", mocked.call_args.kwargs)
        self.assertFalse(mocked.call_args.kwargs["weights_only"])

    def test_forest_teacher_uses_forest_device(self):
        forest = ForestEcosystem(input_dim=4, max_trees=4).to(torch.device("cpu"))
        forest._plant_tree()
        teacher = ForestTeacher(forest)
        self.assertEqual(
            next(teacher.parameters()).device,
            next(forest.parameters()).device,
        )

    def test_tree_fitness_update_stays_meaningful_at_start(self):
        tree = TreeExpert(input_dim=4, tree_id=1, arch=TreeArch(hidden_dim=8))
        before = tree.fitness
        tree.update_fitness(loss_value=1.0)
        self.assertGreaterEqual(tree.fitness, before)

    def test_pruning_rebuilds_ecosystem_optimizer(self):
        forest = ForestEcosystem(input_dim=4, max_trees=6)
        for _ in range(4):
            forest._plant_tree()
        simulator = EcosystemSimulator(forest=forest, selection_threshold=0.5)
        old_optimizer = simulator.optimizer
        pruned = simulator.prune_weak_trees(min_keep=2)
        self.assertGreater(pruned, 0)
        self.assertIsNot(simulator.optimizer, old_optimizer)

    def test_tree_graveyard_defaults_to_platform_tempdir(self):
        graveyard = TreeGraveyard()
        tmpdir = tempfile.gettempdir()
        self.assertEqual(str(graveyard.weights_dir.parent), tmpdir)
        self.assertEqual(str(graveyard.save_path.parent), tmpdir)


if __name__ == "__main__":
    unittest.main()
