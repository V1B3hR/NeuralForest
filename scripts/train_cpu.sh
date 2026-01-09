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
# Estimate: ~2.5 minutes per epoch on standard CPU
MINUTES=$((EPOCHS * 5 / 2))
HOURS=$((MINUTES / 60))
REMAINING_MINS=$((MINUTES % 60))
if [ $HOURS -gt 0 ]; then
    echo "Estimated time: $MINUTES minutes (~${HOURS}h ${REMAINING_MINS}m)"
else
    echo "Estimated time: $MINUTES minutes"
fi
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
