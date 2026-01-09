#!/bin/bash
# CPU Training Helper Script

set -e

EPOCHS=${1:-100}
BATCH_SIZE=${2:-16}
OUTPUT_DIR=${3:-"training_demos/results/cpu_training"}

echo "========================================="
echo "NeuralForest CPU Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Estimated time: $((EPOCHS * 2 + EPOCHS / 2)) minutes"
echo ""

# Set CPU optimizations
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

echo "CPU Threads: $OMP_NUM_THREADS"
echo ""
echo "Starting training..."
echo ""

python training_demos/cifar10_hybrid_training.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device cpu \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
