import time
import numpy as np
from kernel.culling_mod import cull_back_faces


class CullBackFacesBenchmark:
    """Benchmark class for measuring performance of back-face culling operations."""

    def __init__(self):
        # Initialize with JIT warm-up using a simple triangle case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertices = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        )
        warm_up_indices = np.array([[0, 1, 2]], dtype=np.int32)

        print("\nWarming up JIT compilation...")
        for _ in range(100):  # Increased warm-up iterations for consistency
            cull_back_faces(warm_up_vertices, warm_up_indices)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices, indices):
        """Run performance benchmark on back-face culling operation.

        Performs 10 loops of 1000 culling operations each to get statistically significant timing data.

        Args:
            vertices: Vertex positions array in camera space
            indices: Triangle indices array

        Returns:
            float: Average time in milliseconds for 1000 culling operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                cull_back_faces(vertices, indices)
            end_time = time.perf_counter()
            total_times.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds

        avg_time = sum(total_times) / len(total_times)
        return avg_time


def run_benchmark(benchmark, vertices, indices, name):
    """Run a single benchmark case and print results.

    Args:
        benchmark: CullBackFacesBenchmark instance
        vertices: Vertex positions array
        indices: Triangle indices array
        name: Description of the benchmark case
    """
    avg_time = benchmark.benchmark_calculation(vertices, indices)
    print(f"{name}:")
    print(f"  Average time for 1000 culling operations: {avg_time:.3f} ms")
    print()


def run_benchmarks():
    """Run a series of benchmark tests covering different back-face culling scenarios."""
    benchmark = CullBackFacesBenchmark()

    # Test Case 1: Single front-facing triangle
    vertices = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
    )
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    run_benchmark(benchmark, vertices, indices, "Single Front-facing Triangle")

    # Test Case 2: Single back-facing triangle
    vertices = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32
    )
    run_benchmark(benchmark, vertices, indices, "Single Back-facing Triangle")

    # Test Case 3: Mixed triangles (some front, some back)
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],  # Front-facing
            [3, 5, 4],  # Back-facing
        ],
        dtype=np.int32,
    )
    run_benchmark(benchmark, vertices, indices, "Mixed Front and Back-facing Triangles")

    # Test Case 4: Edge case - Degenerate triangle
    vertices = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32
    )
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    run_benchmark(benchmark, vertices, indices, "Degenerate Triangle")


if __name__ == "__main__":
    run_benchmarks()
