import time
import numpy as np
from kernel.clipping_mod import clip_triangle, clip_triangle_and_normals


class ClipTriangleBenchmark:
    """Benchmark class for measuring performance of triangle clipping operations against a plane."""

    def __init__(self):
        # Initialize with JIT warm-up using a simple triangle case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertices = np.array(
            [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [0.0, -1.0, 1.0]]
        )
        warm_up_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        warm_up_normal = np.array([0.0, 0.0, 1.0])
        print("\nWarming up JIT compilation...")
        for _ in range(100):
            clip_triangle(warm_up_vertices, warm_up_normal)
            clip_triangle_and_normals(warm_up_vertices, warm_up_normals, warm_up_normal)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, plane_normal, vertices, vertex_normals=None):
        """Run performance benchmark on triangle clipping operation.

        Performs 10 loops of 1000 clipping operations each to get statistically significant timing data.

        Args:
            plane_normal: Normal vector of the clipping plane
            vertices: Triangle vertices to be clipped
            vertex_normals: Optional vertex normals for interpolation benchmark

        Returns:
            float: Average time in milliseconds for 1000 clipping operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        if vertex_normals is None:
            # Basic clipping without normal interpolation
            for _ in range(NUM_LOOPS):
                start_time = time.perf_counter()
                for _ in range(CALLS_PER_LOOP):
                    clip_triangle(vertices, plane_normal)
                end_time = time.perf_counter()
                total_times.append(
                    (end_time - start_time) * 1000
                )  # Convert to milliseconds
        else:
            # Clipping with normal interpolation
            for _ in range(NUM_LOOPS):
                start_time = time.perf_counter()
                for _ in range(CALLS_PER_LOOP):
                    clip_triangle_and_normals(vertices, vertex_normals, plane_normal)
                end_time = time.perf_counter()
                total_times.append(
                    (end_time - start_time) * 1000
                )  # Convert to milliseconds

        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, plane_normal, vertices, name, vertex_normals=None):
    """Run benchmark for triangle clipping"""
    avg_time = benchmark.benchmark_calculation(plane_normal, vertices, vertex_normals)

    if vertex_normals is None:
        print(f"\n{name} (Basic Clipping)")
    else:
        print(f"\n{name} (With Normal Interpolation)")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def run_benchmarks():
    """Run a series of benchmark tests covering different triangle clipping scenarios."""
    print("Running Triangle Clipping Benchmarks...")
    print("Each test runs 10 loops of 1000 clipping operations")
    print("Results show the average time taken for 1000 operations")
    benchmark = ClipTriangleBenchmark()

    # Define clipping plane (XY plane, normal pointing in +Z direction)
    plane_normal = np.array([0.0, 0.0, 1.0])

    # Test case 1: Triangle intersecting plane
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],  # Above plane
            [-1.0, 1.0, -1.0],  # Below plane
            [0.0, -1.0, 0.0],  # On plane
        ]
    )
    vertex_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    run_benchmark(benchmark, plane_normal, vertices, "Intersecting Triangle")
    run_benchmark(
        benchmark, plane_normal, vertices, "Intersecting Triangle", vertex_normals
    )

    # Test case 2: Triangle completely above plane
    vertices = np.array([[1.0, 1.0, 2.0], [-1.0, 1.0, 2.0], [0.0, -1.0, 2.0]])
    vertex_normals = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    run_benchmark(benchmark, plane_normal, vertices, "Triangle Above Plane")
    run_benchmark(
        benchmark, plane_normal, vertices, "Triangle Above Plane", vertex_normals
    )

    # Test case 3: Triangle exactly on plane
    vertices = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    vertex_normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    run_benchmark(benchmark, plane_normal, vertices, "Triangle On Plane")
    run_benchmark(
        benchmark, plane_normal, vertices, "Triangle On Plane", vertex_normals
    )

    # Test case 4: Triangle with one vertex below and two above
    vertices = np.array(
        [
            [0.0, 0.0, -1.0],  # Below plane
            [1.0, 0.0, 1.0],  # Above plane
            [-1.0, 0.0, 1.0],  # Above plane
        ]
    )
    vertex_normals = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    run_benchmark(benchmark, plane_normal, vertices, "Split Triangle (Two Output)")
    run_benchmark(
        benchmark, plane_normal, vertices, "Split Triangle (Two Output)", vertex_normals
    )


if __name__ == "__main__":
    run_benchmarks()
