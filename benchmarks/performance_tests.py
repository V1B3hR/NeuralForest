"""
Performance Testing and Benchmarking for NeuralForest

Comprehensive benchmarks for:
- Inference latency
- Throughput
- Memory usage
- Scalability
- Accuracy metrics
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class BenchmarkResults:
    """
    Container for benchmark results with statistics.
    """

    name: str
    num_samples: int
    duration: float
    latencies: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        return np.mean(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def p50_latency_ms(self) -> float:
        """Median latency in milliseconds."""
        return np.percentile(self.latencies, 50) * 1000 if self.latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency in milliseconds."""
        return np.percentile(self.latencies, 95) * 1000 if self.latencies else 0.0

    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency in milliseconds."""
        return np.percentile(self.latencies, 99) * 1000 if self.latencies else 0.0

    @property
    def avg_throughput(self) -> float:
        """Average throughput in samples/second."""
        return np.mean(self.throughputs) if self.throughputs else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "name": self.name,
            "num_samples": self.num_samples,
            "duration": self.duration,
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "min_ms": min(self.latencies) * 1000 if self.latencies else 0,
                "max_ms": max(self.latencies) * 1000 if self.latencies else 0,
            },
            "throughput": {
                "avg_samples_per_sec": self.avg_throughput,
                "total_samples": self.num_samples,
                "duration_sec": self.duration,
            },
            "memory": self.memory_usage,
            "accuracy": self.accuracy_metrics,
        }

    def print_summary(self):
        """Print formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        print(f"Samples: {self.num_samples}")
        print(f"Duration: {self.duration:.2f}s")
        print("\nLatency:")
        print(f"  Average: {self.avg_latency_ms:.2f}ms")
        print(f"  Median (P50): {self.p50_latency_ms:.2f}ms")
        print(f"  P95: {self.p95_latency_ms:.2f}ms")
        print(f"  P99: {self.p99_latency_ms:.2f}ms")
        print("\nThroughput:")
        print(f"  Average: {self.avg_throughput:.2f} samples/sec")
        if self.memory_usage:
            print("\nMemory:")
            for key, val in self.memory_usage.items():
                print(f"  {key}: {val:.2f} MB")
        if self.accuracy_metrics:
            print("\nAccuracy:")
            for key, val in self.accuracy_metrics.items():
                print(f"  {key}: {val:.4f}")
        print(f"{'='*60}\n")

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Results saved to {path}")


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for NeuralForest.

    Tests:
    - Single-sample latency
    - Batch inference throughput
    - Memory efficiency
    - Scaling characteristics
    - Accuracy on test data
    """

    def __init__(self, forest_or_api, device=None):
        """
        Initialize benchmark suite.

        Args:
            forest_or_api: ForestEcosystem or NeuralForestAPI instance
            device: Device to run benchmarks on
        """
        # Check if it's an API or forest
        if hasattr(forest_or_api, "forest"):
            self.api = forest_or_api
            self.forest = forest_or_api.forest
        else:
            self.api = None
            self.forest = forest_or_api

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.results_history = []

    def benchmark_latency(
        self, input_dim: int, num_samples: int = 1000, batch_size: int = 1
    ) -> BenchmarkResults:
        """
        Benchmark inference latency.

        Args:
            input_dim: Dimension of input
            num_samples: Number of samples to test
            batch_size: Batch size for inference

        Returns:
            BenchmarkResults with latency statistics
        """
        print("\n🔬 Running latency benchmark...")
        print(f"   Samples: {num_samples}, Batch size: {batch_size}")

        latencies = []
        start_time = time.time()

        self.forest.eval()

        for i in range(num_samples // batch_size):
            # Generate random input
            x = torch.randn(batch_size, input_dim).to(self.device)

            # Measure single inference time
            iter_start = time.time()
            with torch.no_grad():
                if self.api:
                    self.api.predict({"input": x})
                else:
                    self.forest.forward_forest(x)

            latency = time.time() - iter_start
            latencies.append(latency)

        duration = time.time() - start_time

        results = BenchmarkResults(
            name="Latency Benchmark",
            num_samples=num_samples,
            duration=duration,
            latencies=latencies,
        )

        self.results_history.append(results)
        return results

    def benchmark_throughput(
        self,
        input_dim: int,
        duration_seconds: float = 10.0,
        batch_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResults]:
        """
        Benchmark throughput with different batch sizes.

        Args:
            input_dim: Dimension of input
            duration_seconds: How long to run each test
            batch_sizes: List of batch sizes to test

        Returns:
            List of BenchmarkResults for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32, 64]

        print("\n🔬 Running throughput benchmark...")
        print(f"   Duration: {duration_seconds}s per batch size")

        all_results = []
        self.forest.eval()

        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")

            samples_processed = 0
            throughputs = []
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                x = torch.randn(batch_size, input_dim).to(self.device)

                iter_start = time.time()
                with torch.no_grad():
                    if self.api:
                        self.api.predict({"input": x})
                    else:
                        self.forest.forward_forest(x)
                iter_duration = time.time() - iter_start

                throughput = batch_size / iter_duration
                throughputs.append(throughput)
                samples_processed += batch_size

            duration = time.time() - start_time

            results = BenchmarkResults(
                name=f"Throughput Benchmark (batch_size={batch_size})",
                num_samples=samples_processed,
                duration=duration,
                throughputs=throughputs,
            )

            all_results.append(results)
            self.results_history.append(results)

        return all_results

    def benchmark_memory(
        self, input_dim: int, batch_size: int = 32
    ) -> BenchmarkResults:
        """
        Benchmark memory usage.

        Args:
            input_dim: Dimension of input
            batch_size: Batch size for testing

        Returns:
            BenchmarkResults with memory statistics
        """
        print("\n🔬 Running memory benchmark...")

        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Measure model memory
        model_memory = sum(
            p.numel() * p.element_size() for p in self.forest.parameters()
        ) / (
            1024**2
        )  # Convert to MB

        # Measure mulch memory
        mulch_memory = (
            len(self.forest.mulch) * input_dim * 4 / (1024**2)
        )  # Rough estimate

        # Measure inference memory
        x = torch.randn(batch_size, input_dim).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                self.forest.forward_forest(x)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_memory = 0.0

        memory_usage = {
            "model_mb": model_memory,
            "mulch_mb": mulch_memory,
            "peak_inference_mb": peak_memory,
            "total_mb": model_memory + mulch_memory,
        }

        results = BenchmarkResults(
            name="Memory Benchmark",
            num_samples=batch_size,
            duration=0.0,
            memory_usage=memory_usage,
        )

        self.results_history.append(results)
        return results

    def benchmark_accuracy(
        self, test_function, num_samples: int = 100, input_dim: int = 2
    ) -> BenchmarkResults:
        """
        Benchmark accuracy on a test function.

        Args:
            test_function: Function that takes x and returns y
            num_samples: Number of test samples
            input_dim: Input dimension

        Returns:
            BenchmarkResults with accuracy metrics
        """
        print("\n🔬 Running accuracy benchmark...")
        print(f"   Test samples: {num_samples}")

        # Generate test data
        x_test = torch.randn(num_samples, input_dim).to(self.device)
        y_test = test_function(x_test)

        # Make predictions
        start_time = time.time()
        with torch.no_grad():
            y_pred, _, _ = self.forest.forward_forest(x_test)
        duration = time.time() - start_time

        # Compute metrics
        errors = (y_pred - y_test).abs()
        mse = (errors**2).mean().item()
        mae = errors.mean().item()
        max_error = errors.max().item()

        # R² score
        ss_res = ((y_test - y_pred) ** 2).sum().item()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        accuracy_metrics = {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "r2_score": r2,
        }

        results = BenchmarkResults(
            name="Accuracy Benchmark",
            num_samples=num_samples,
            duration=duration,
            accuracy_metrics=accuracy_metrics,
        )

        self.results_history.append(results)
        return results

    def run_full_benchmark(
        self, input_dim: int = 2, test_function=None
    ) -> Dict[str, BenchmarkResults]:
        """
        Run complete benchmark suite.

        Args:
            input_dim: Input dimension
            test_function: Optional test function for accuracy

        Returns:
            Dictionary of all benchmark results
        """
        print("\n" + "=" * 60)
        print("  RUNNING FULL BENCHMARK SUITE")
        print("=" * 60)

        results = {}

        # Latency
        results["latency"] = self.benchmark_latency(input_dim, num_samples=1000)
        results["latency"].print_summary()

        # Throughput
        throughput_results = self.benchmark_throughput(input_dim, duration_seconds=5.0)
        results["throughput"] = throughput_results
        for res in throughput_results:
            res.print_summary()

        # Memory
        results["memory"] = self.benchmark_memory(input_dim)
        results["memory"].print_summary()

        # Accuracy (if test function provided)
        if test_function:
            results["accuracy"] = self.benchmark_accuracy(
                test_function, num_samples=100, input_dim=input_dim
            )
            results["accuracy"].print_summary()

        print("=" * 60)
        print("  BENCHMARK SUITE COMPLETE")
        print("=" * 60 + "\n")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        return {
            "total_benchmarks": len(self.results_history),
            "results": [r.to_dict() for r in self.results_history],
        }

    def save_all_results(self, path: str):
        """Save all benchmark results to JSON file."""
        summary = self.get_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ All benchmark results saved to {path}")
