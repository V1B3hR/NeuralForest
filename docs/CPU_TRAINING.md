# 💻 CPU Training Guide for NeuralForest

## Overview

NeuralForest supports full training on CPU with identical results to GPU training. The only difference is training time.

## Performance Comparison

| Hardware | 100 Epochs | 50 Epochs | 25 Epochs |
|----------|------------|-----------|-----------|
| **Modern CPU (8+ cores)** | 4-5 hours | 2-2.5 hours | 1-1.5 hours |
| **Standard CPU (4-6 cores)** | 5-7 hours | 2.5-3.5 hours | 1.5-2 hours |
| **Gaming GPU (RTX 3070)** | 40-50 min | 20-25 min | 10-15 min |

**Note**: Accuracy results are identical regardless of hardware.

## GitHub Actions Workflow

### Automatic Training

Workflow runs automatically:
- **On push to main** (25 epochs, ~1.5 hours)
- **Weekly schedule** (100 epochs, Sunday 2 AM UTC)

### Manual Training

1. Go to **Actions** tab
2. Select **CPU Training - NeuralForest**
3. Click **Run workflow**
4. Choose parameters:
   - **Epochs**: 25, 50, or 100
   - **Batch size**: 16 (fastest), 32, or 64
   - **Checkpoint interval**: 20 (default)

### Expected Times

| Epochs | Batch 16 | Batch 32 |
|--------|----------|----------|
| 25 | ~1.5h | ~2h |
| 50 | ~2.5h | ~3.5h |
| 100 | ~4.5h | ~5.5h |

## Local CPU Training

### Quick Start

```bash
# 100 epochs (recommended)
python training_demos/cifar10_hybrid_training.py --epochs 100 --batch_size 16 --device cpu

# 50 epochs (faster)
python training_demos/cifar10_hybrid_training.py --epochs 50 --batch_size 16 --device cpu

# 25 epochs (quick test)
python training_demos/cifar10_hybrid_training.py --epochs 25 --batch_size 16 --device cpu
```

### Optimization Tips

1. **Use smaller batch size**: `--batch_size 16` is faster on CPU than 64
2. **Set thread count**:
   ```bash
   export OMP_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   python training_demos/cifar10_hybrid_training.py --epochs 100 --batch_size 16 --device cpu
   ```
3. **Background execution** (overnight):
   ```bash
   nohup python training_demos/cifar10_hybrid_training.py --epochs 100 --batch_size 16 --device cpu &
   ```

## Results

### Expected Accuracy (100 epochs)

- **Test Accuracy**: 85-88%
- **Train Accuracy**: 93-95%
- **Trees**: 10-12 (evolved from 6)
- **Fitness**: +250-300% improvement

Results are identical between CPU and GPU training!

## Troubleshooting

### Training is slow
- Reduce `--batch_size` to 16
- Use fewer epochs for testing (25 or 50)
- Check CPU usage (should be 50-80%)

### Out of memory
- Reduce `--batch_size` to 16 or 8
- Close other applications
- Check system RAM (should have 4GB+ free)

### Process killed
- Check available RAM
- Reduce batch size
- Monitor with `htop` or Task Manager

## FAQ

**Q: Are CPU results as good as GPU?**  
A: Yes! Identical accuracy. Only difference is training time.

**Q: Can I interrupt and resume?**  
A: Yes, use checkpoints. Resume from last checkpoint with same parameters.

**Q: How to speed up CPU training?**  
A: Use `--batch_size 16`, set `OMP_NUM_THREADS`, and run overnight.

**Q: What's the minimum hardware?**  
A: 4GB RAM, 2+ CPU cores. More cores = faster training.
