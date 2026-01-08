# 🚀 GPU Training Guide for NeuralForest

## NauraLas Workflow

NauraLas to automatyczny workflow do treningu NeuralForest z pełną obsługą GPU.

### Tryby treningu:

#### 1. CPU Training (automatyczny)
Uruchamia się przy każdym push/PR:
```bash
python ecosystem_simulation.py --epochs 10 --batch-size 16
```

**Parametry:**
- Runner: `ubuntu-latest`
- Timeout: 30 minut
- Artifacts: `training-results-cpu`

#### 2. Single GPU Training (automatyczny)
Uruchamia się przy każdym push/PR na GitHub GPU runner (NVIDIA T4):
```bash
python ecosystem_simulation.py --epochs 100 --batch-size 128 --use-amp --device cuda
```

**Parametry:**
- Runner: `ubuntu-latest-4-cores-16gb-gpu-t4`
- CUDA: 12.1
- PyTorch: 2.1.0 z CUDA support
- Mixed precision: Enabled
- Timeout: 120 minut
- Artifacts: `training-results-gpu`, `tensorboard-logs`

**Dodatkowo uruchamia:**
- `phase6_demo.py --device cuda`
- GPU Benchmark (matmul 4096x4096, 100 iterations)
- GPU Memory Report

#### 3. Multi-GPU Training (manual)
Dostępny tylko przez manual trigger w Actions:
- Używa NVIDIA A100
- Distributed Data Parallel
- 200 epochs, batch size 256

```bash
torchrun --nproc_per_node=auto --nnodes=1 ecosystem_simulation.py --epochs 200 --batch-size 256 --distributed
```

**Parametry:**
- Runner: `ubuntu-latest-8-cores-32gb-gpu-a100`
- Timeout: 180 minut
- Artifacts: `training-results-multi-gpu`

#### 4. Self-hosted GPU (manual)
Dla własnych serwerów z GPU:
- Długi trening (500 epochs)
- Gradient checkpointing
- Mixed precision

```bash
python ecosystem_simulation.py --epochs 500 --batch-size 256 --use-amp --gradient-checkpointing --device cuda
```

**Parametry:**
- Runner: `self-hosted-gpu`
- Timeout: 180 minut
- Artifacts: `training-results-self-hosted` (90 dni retencji)

### Lokalne użycie:

#### CPU Training
```bash
python ecosystem_simulation.py --epochs 50 --batch-size 32
```

#### Single GPU Training
```bash
python ecosystem_simulation.py --epochs 100 --batch-size 128 --use-amp --device cuda
```

#### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 ecosystem_simulation.py --epochs 200 --distributed
```

### Dostępne argumenty CLI:

| Argument | Typ | Domyślna wartość | Opis |
|----------|-----|------------------|------|
| `--epochs` | int | 100 | Liczba epok treningu |
| `--batch-size` | int | 32 | Rozmiar batcha |
| `--use-amp` | flag | False | Użyj automatic mixed precision |
| `--device` | str | auto | Urządzenie (cuda/cpu) |
| `--distributed` | flag | False | Użyj distributed training |
| `--gradient-checkpointing` | flag | False | Włącz gradient checkpointing |

### Wymagania:

#### Minimalne wymagania:
- Python 3.10+
- PyTorch 2.1.0+
- CUDA 12.1+ (dla GPU training)

#### Wymagania sprzętowe:

**CPU Training:**
- 2+ cores
- 4GB+ RAM

**Single GPU Training:**
- NVIDIA GPU z minimum 4GB VRAM (T4 lub lepszy)
- 16GB+ RAM
- CUDA 12.1+

**Multi-GPU Training:**
- 2+ NVIDIA GPUs
- 16GB+ VRAM per GPU (A100 lub lepszy)
- 32GB+ RAM
- CUDA 12.1+

**Batch size recommendations:**
- CPU: 16-32
- Single GPU (4GB VRAM): 64-128
- Single GPU (16GB+ VRAM): 128-256
- Multi-GPU: 256+

### Monitoring:

Workflow automatycznie generuje:
- **TensorBoard logs** (`runs/`) - dostępne w artifacts
- **Checkpoints** (`.pt` files) - zapisywane podczas treningu
- **Training plots** (`results/*.png`) - wizualizacje metryk
- **GPU utilization reports** - raporty użycia GPU

Wszystkie artifacts są dostępne w GitHub Actions po zakończeniu jobów.

### Uruchomienie workflow:

#### Automatyczne uruchomienie:
Workflow uruchamia się automatycznie przy:
- Push do branch `main` lub `develop`
- Pull Request do `main`

Jobs CPU i Single GPU uruchamiają się automatycznie.

#### Manualne uruchomienie:
1. Przejdź do zakładki **Actions** w repozytorium
2. Wybierz workflow **NauraLas**
3. Kliknij **Run workflow**
4. Wybierz branch
5. Kliknij **Run workflow**

Jobs Multi-GPU i Self-hosted uruchamiają się tylko przy manualnym triggerze.

### Troubleshooting:

#### "CUDA out of memory"
- Zmniejsz `--batch-size`
- Włącz `--gradient-checkpointing`
- Użyj `--use-amp` dla mixed precision

#### "GPU not available"
- Sprawdź czy CUDA jest zainstalowana: `nvidia-smi`
- Sprawdź wersję PyTorch: `python -c "import torch; print(torch.__version__)"`
- Upewnij się że PyTorch został zainstalowany z CUDA support

#### "Self-hosted runner offline"
- Self-hosted runner wymaga własnej konfiguracji
- Zobacz [GitHub Self-hosted Runners Documentation](https://docs.github.com/en/actions/hosting-your-own-runners)

#### Workflow fails gracefully
- Workflow jest zaprojektowany aby działać nawet jeśli niektóre joby failują
- Artifacts są uploadowane z `if: always()` aby zachować wyniki nawet przy błędach

### Performance Tips:

1. **Use mixed precision (`--use-amp`)** - może przyspieszyć trening 2-3x na nowoczesnych GPUs
2. **Increase batch size** - większy batch size zazwyczaj daje lepsze wykorzystanie GPU
3. **Use gradient checkpointing** - zmniejsza zużycie pamięci kosztem niewielkiego spowolnienia
4. **Monitor GPU utilization** - sprawdź czy GPU jest w pełni wykorzystane
5. **Use TensorBoard** - śledź metryki w czasie rzeczywistym

### Example Workflows:

#### Quick test (CPU):
```bash
python ecosystem_simulation.py --epochs 5 --batch-size 16
```

#### Standard training (Single GPU):
```bash
python ecosystem_simulation.py --epochs 100 --batch-size 128 --use-amp --device cuda
```

#### Production training (Multi-GPU):
```bash
torchrun --nproc_per_node=4 ecosystem_simulation.py --epochs 500 --batch-size 256 --distributed --use-amp
```

### FAQ:

**Q: Czy mogę używać własnego GPU?**  
A: Tak! Skonfiguruj self-hosted runner z labelką `self-hosted-gpu`.

**Q: Jak długo są przechowywane artifacts?**  
A: CPU (7 dni), Single GPU (14 dni), Multi-GPU (30 dni), Self-hosted (90 dni).

**Q: Czy workflow działa na forku?**  
A: Tak, ale GitHub GPU runners mogą nie być dostępne. CPU job zawsze działa.

**Q: Jak zmienić liczbę epok w workflow?**  
A: Edytuj `.github/workflows/NauraLas.yml` i zmień parametr `--epochs`.

**Q: Czy mogę dodać własny job?**  
A: Tak! Dodaj nowy job w `.github/workflows/NauraLas.yml` według wzoru istniejących.
