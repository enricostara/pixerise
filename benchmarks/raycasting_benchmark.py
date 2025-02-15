import time
import numpy as np
from kernel.raycasting_mod import check_ray_triangle_intersection


class RayTriangleIntersectionBenchmark:
    """Benchmark class for measuring performance of ray-triangle intersection tests."""

    def __init__(self):
        # Initialize with JIT warm-up using a simple ray-triangle case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_ray_origin = np.array([0.0, 0.0, -1.0])
        warm_up_ray_dir = np.array([0.0, 0.0, 1.0])
        warm_up_triangle = (
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        )

        print("\nWarming up JIT compilation...")
        for _ in range(100):
            check_ray_triangle_intersection(
                warm_up_ray_origin, warm_up_ray_dir, *warm_up_triangle
            )
        print("Warm-up complete.\n")

    def benchmark_calculation(self, ray_origin, ray_dir, v0, v1, v2):
        """Run performance benchmark on ray-triangle intersection test.

        Performs 10 loops of 1000 intersection tests each to get statistically significant timing data.

        Args:
            ray_origin: Origin point of ray
            ray_dir: Direction vector of ray (normalized)
            v0, v1, v2: Vertices of triangle to test against

        Returns:
            float: Average time in milliseconds for 1000 intersection tests
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                check_ray_triangle_intersection(ray_origin, ray_dir, v0, v1, v2)
            end_time = time.perf_counter()
            total_times.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds

        avg_time = sum(total_times) / len(total_times)
        return avg_time


def run_benchmark(benchmark, ray_origin, ray_dir, triangle_vertices, name):
    """Run a single benchmark case and print results.

    Args:
        benchmark: RayTriangleIntersectionBenchmark instance
        ray_origin: Origin point of ray
        ray_dir: Direction vector of ray
        triangle_vertices: Tuple of three vertices defining the triangle
        name: Description of the benchmark case
    """
    avg_time = benchmark.benchmark_calculation(ray_origin, ray_dir, *triangle_vertices)
    print(f"{name}:")
    print(f"  Average time for 1000 intersection tests: {avg_time:.3f} ms")
    print()


def run_benchmarks():
    """Run a series of benchmark tests covering different ray-triangle intersection scenarios."""
    benchmark = RayTriangleIntersectionBenchmark()

    # Test Case 1: Ray hits triangle center
    ray_origin = np.array([0.0, 0.0, -1.0])
    ray_dir = np.array([0.0, 0.0, 1.0])
    triangle = (
        np.array([1.0, 1.0, 0.0]),
        np.array([-1.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    run_benchmark(
        benchmark, ray_origin, ray_dir, triangle, "Direct Hit at Triangle Center"
    )

    # Test Case 2: Ray misses triangle
    ray_origin = np.array([2.0, 2.0, -1.0])
    ray_dir = np.array([0.0, 0.0, 1.0])
    run_benchmark(benchmark, ray_origin, ray_dir, triangle, "Ray Misses Triangle")

    # Test Case 3: Ray parallel to triangle
    ray_origin = np.array([0.0, 0.0, 1.0])
    ray_dir = np.array([1.0, 0.0, 0.0])
    run_benchmark(benchmark, ray_origin, ray_dir, triangle, "Ray Parallel to Triangle")

    # Test Case 4: Ray hits triangle edge
    ray_origin = np.array([1.0, 1.0, -1.0])
    ray_dir = np.array([0.0, 0.0, 1.0])
    run_benchmark(benchmark, ray_origin, ray_dir, triangle, "Ray Hits Triangle Edge")

    # Test Case 5: Ray hits triangle vertex
    ray_origin = np.array([0.0, -1.0, -1.0])
    ray_dir = np.array([0.0, 0.0, 1.0])
    run_benchmark(benchmark, ray_origin, ray_dir, triangle, "Ray Hits Triangle Vertex")


if __name__ == "__main__":
    run_benchmarks()
