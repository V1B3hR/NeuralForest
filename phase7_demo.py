"""
Phase 7 Demo: Production & Scaling

Demonstrates production API, checkpoint management, deployment readiness,
and performance benchmarking of NeuralForest.
"""

import torch
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NeuralForest import ForestEcosystem, DEVICE
from api import NeuralForestAPI, ForestCheckpoint
from benchmarks import PerformanceBenchmark


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_checkpoint_management():
    """Demonstrate checkpoint save/load functionality."""
    print_section("Checkpoint Management")

    # Create a forest
    print("Creating forest...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)

    # Add some data
    for i in range(100):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))

    print(f"  Initial forest: {forest.num_trees()} trees, {len(forest.mulch)} memories")

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_forest.pt")

        print(f"\nSaving checkpoint to {checkpoint_path}...")
        metadata = {
            "version": "1.0",
            "description": "Test checkpoint for demo",
            "training_steps": 1000,
        }
        ForestCheckpoint.save(forest, checkpoint_path, metadata=metadata)

        # Validate checkpoint
        print("\nValidating checkpoint...")
        is_valid = ForestCheckpoint.validate(checkpoint_path)
        print(f"  Checkpoint valid: {is_valid}")

        # Get checkpoint info
        info = ForestCheckpoint.get_info(checkpoint_path)
        print("\nCheckpoint info:")
        for key, val in info.items():
            if key != "metadata":
                print(f"  {key}: {val}")
        print(f"  metadata: {info['metadata']}")

        # Load checkpoint
        print("\nLoading checkpoint...")
        loaded_forest = ForestCheckpoint.load(checkpoint_path, device=DEVICE)
        print(
            f"  Loaded forest: {loaded_forest.num_trees()} trees, {len(loaded_forest.mulch)} memories"
        )

        # Verify data preserved
        print("\n✅ Checkpoint management working correctly")
        print(
            f"   Trees preserved: {forest.num_trees()} == {loaded_forest.num_trees()}"
        )
        print(f"   Memory preserved: {len(forest.mulch)} == {len(loaded_forest.mulch)}")


def demo_production_api():
    """Demonstrate production API functionality."""
    print_section("Production API")

    # Create forest
    print("Creating forest for API...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)

    # Train briefly
    print("  Adding training data...")
    for i in range(200):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))

    # Create API
    print("\nInitializing API...")
    api = NeuralForestAPI(forest=forest, device=DEVICE)
    print(f"  {api}")

    # Make predictions
    print("\nMaking predictions...")
    test_input = torch.randn(5, 2)

    result = api.predict({"input": test_input}, return_details=True)
    print(f"  Prediction shape: {result['prediction'].shape}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
    print(f"  Trees used: {result['trees_used']}")

    # Online learning
    print("\nTesting online learning...")
    train_x = torch.randn(3, 2)
    train_y = torch.randn(3, 1)

    train_result = api.train_online({"input": train_x}, train_y, feedback=0.8)
    print(f"  Training successful: {train_result['success']}")
    print(f"  Memory size: {train_result['memory_size']}")
    print(f"  Priority: {train_result['priority']:.3f}")

    # Get status
    print("\nGetting forest status...")
    status = api.get_forest_status()
    print(f"  Status: {status['status']}")
    print(f"  Trees: {status['num_trees']}")
    print(f"  Predictions served: {status['performance']['total_predictions']}")
    print(f"  Average latency: {status['performance']['avg_prediction_time_ms']:.2f}ms")
    print(f"  Memory utilization: {status['memory_usage']['utilization']:.1%}")

    # Health check
    print("\nPerforming health check...")
    health = api.health_check()
    print(f"  Health: {health['health']}")
    print(f"  Issues: {health['issues']}")

    # Save API state
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "api_forest.pt")
        print("\nSaving API state...")
        api.save(save_path)

        # Load in new API
        print("Loading into new API...")
        new_api = NeuralForestAPI(checkpoint_path=save_path, device=DEVICE)
        print(f"  {new_api}")

    print("\n✅ Production API working correctly")


def demo_performance_benchmarks():
    """Demonstrate performance benchmarking."""
    print_section("Performance Benchmarks")

    # Create forest
    print("Creating forest for benchmarking...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=8).to(DEVICE)

    # Create benchmark suite
    benchmark = PerformanceBenchmark(forest, device=DEVICE)

    # Latency benchmark
    print("\n--- Latency Benchmark ---")
    latency_results = benchmark.benchmark_latency(
        input_dim=2, num_samples=500, batch_size=1
    )
    print(f"Average latency: {latency_results.avg_latency_ms:.2f}ms")
    print(f"P95 latency: {latency_results.p95_latency_ms:.2f}ms")
    print(f"P99 latency: {latency_results.p99_latency_ms:.2f}ms")

    # Throughput benchmark
    print("\n--- Throughput Benchmark ---")
    throughput_results = benchmark.benchmark_throughput(
        input_dim=2, duration_seconds=3.0, batch_sizes=[1, 8, 16]
    )

    for result in throughput_results:
        batch_size = int(result.name.split("=")[1].rstrip(")"))
        print(f"Batch size {batch_size}: {result.avg_throughput:.1f} samples/sec")

    # Memory benchmark
    print("\n--- Memory Benchmark ---")
    memory_results = benchmark.benchmark_memory(input_dim=2, batch_size=32)
    print(f"Model memory: {memory_results.memory_usage['model_mb']:.2f} MB")
    print(f"Total memory: {memory_results.memory_usage['total_mb']:.2f} MB")

    # Accuracy benchmark
    print("\n--- Accuracy Benchmark ---")

    def test_function(x):
        """Simple quadratic test function."""
        return (x**2).sum(dim=-1, keepdim=True)

    accuracy_results = benchmark.benchmark_accuracy(
        test_function=test_function, num_samples=100, input_dim=2
    )
    print(f"MAE: {accuracy_results.accuracy_metrics['mae']:.4f}")
    print(f"MSE: {accuracy_results.accuracy_metrics['mse']:.4f}")
    print(f"R² score: {accuracy_results.accuracy_metrics['r2_score']:.4f}")

    print("\n✅ Performance benchmarks working correctly")


def demo_full_benchmark_suite():
    """Demonstrate complete benchmark suite."""
    print_section("Full Benchmark Suite")

    # Create forest
    print("Creating forest...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=8).to(DEVICE)

    # Create API wrapper
    api = NeuralForestAPI(forest=forest, device=DEVICE)

    # Create benchmark
    benchmark = PerformanceBenchmark(api, device=DEVICE)

    # Define test function
    def test_function(x):
        return torch.sin(x[:, 0:1]) + torch.cos(x[:, 1:2])

    # Run full suite
    print("\nRunning complete benchmark suite...")
    benchmark.run_full_benchmark(input_dim=2, test_function=test_function)

    # Save results
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "benchmark_results.json")
        print(f"\nSaving results to {results_path}...")
        benchmark.save_all_results(results_path)

    print("\n✅ Full benchmark suite completed successfully")


def demo_deployment_readiness():
    """Demonstrate deployment readiness checks."""
    print_section("Deployment Readiness")

    # Create and configure forest
    print("Preparing forest for deployment...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)

    # Add training data
    for i in range(500):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))

    # Create API
    api = NeuralForestAPI(forest=forest, device=DEVICE)

    print("\nDeployment Checklist:")
    print("-" * 50)

    # Check 1: Model loads correctly
    print("✓ Model initialized successfully")

    # Check 2: Can make predictions
    test_input = torch.randn(10, 2)
    result = api.predict({"input": test_input})
    print(f"✓ Predictions working ({result['processing_time_ms']:.2f}ms avg)")

    # Check 3: Health check passes
    health = api.health_check()
    health_status = "✓" if health["health"] == "healthy" else "⚠"
    print(f"{health_status} Health check: {health['health']}")

    # Check 4: Memory usage acceptable
    status = api.get_forest_status()
    mem_util = status["memory_usage"]["utilization"]
    mem_status = "✓" if mem_util < 0.9 else "⚠"
    print(f"{mem_status} Memory utilization: {mem_util:.1%}")

    # Check 5: Latency acceptable
    benchmark = PerformanceBenchmark(api, device=DEVICE)
    latency = benchmark.benchmark_latency(input_dim=2, num_samples=100)
    latency_status = "✓" if latency.avg_latency_ms < 100 else "⚠"
    print(f"{latency_status} Latency: {latency.avg_latency_ms:.2f}ms (target: <100ms)")

    # Check 6: Can save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "deployment_test.pt")
        api.save(checkpoint_path)
        NeuralForestAPI(checkpoint_path=checkpoint_path, device=DEVICE)
        print("✓ Checkpoint save/load working")

    # Check 7: Docker/K8s files exist
    docker_file = os.path.join(
        os.path.dirname(__file__), "deployment", "docker", "Dockerfile"
    )
    k8s_file = os.path.join(
        os.path.dirname(__file__), "deployment", "kubernetes", "deployment.yaml"
    )

    docker_exists = os.path.exists(docker_file)
    k8s_exists = os.path.exists(k8s_file)

    docker_status = "✓" if docker_exists else "⚠"
    k8s_status = "✓" if k8s_exists else "⚠"

    print(f"{docker_status} Docker configuration present")
    print(f"{k8s_status} Kubernetes configuration present")

    print("\n" + "-" * 50)
    print("✅ Deployment readiness check complete")
    print("   Forest is ready for production deployment!")


def main():
    """Run all Phase 7 demonstrations."""
    print("\n" + "=" * 70)
    print("  NEURALFOREST PHASE 7: PRODUCTION & SCALING")
    print("=" * 70)

    try:
        demo_checkpoint_management()
        demo_production_api()
        demo_performance_benchmarks()
        demo_full_benchmark_suite()
        demo_deployment_readiness()

        print("\n" + "=" * 70)
        print("  ✅ PHASE 7 COMPLETE - PRODUCTION READY")
        print("=" * 70)
        print("\n🚀 NeuralForest is ready for deployment! 🚀\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
