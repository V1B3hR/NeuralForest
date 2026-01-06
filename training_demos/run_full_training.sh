#!/bin/bash

echo "=========================================="
echo "CIFAR-10 Full Training (100 epochs)"
echo "=========================================="
echo ""
echo "Starting training with default configuration:"
echo "  - Epochs: 100"
echo "  - Checkpoint every: 20 epochs"
echo "  - Max trees: 15"
echo "  - Learning rate: 0.001"
echo ""
echo "Estimated time: 40-50 minutes (GPU) / 4-5 hours (CPU)"
echo ""

python training_demos/cifar10_full_training.py \
    --epochs 100 \
    --checkpoint_every 20 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --max_trees 15 \
    --competition_fairness 0.3 \
    --output_dir training_demos/results/cifar10_full_100ep

echo ""
echo "=========================================="
echo "Training complete!"
echo "Results saved to: training_demos/results/cifar10_full_100ep"
echo "=========================================="
