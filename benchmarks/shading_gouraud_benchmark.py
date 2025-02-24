import time
import numpy as np
from pixerise.kernel.shading_mod import triangle_gouraud_shading


class GouraudShadingBenchmark:
    """Benchmark class for measuring performance of Gouraud shading computations."""

    def __init__(self):
        # Initialize with JIT warm-up using a simple case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertex_normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        warm_up_light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        warm_up_ambient = 0.1

        print("\nWarming up JIT compilation...")
        for _ in range(100):
            triangle_gouraud_shading(
                warm_up_vertex_normals, warm_up_light_dir, warm_up_ambient
            )
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertex_normals, light_dir, ambient):
        """Run performance benchmark on Gouraud shading computation.

        Performs 10 loops of 1000 shading operations each to get statistically significant timing data.

        Args:
            vertex_normals: Array of normal vectors for each vertex
            light_dir: Direction vector of the light source
            ambient: Ambient light intensity

        Returns:
            float: Average time in milliseconds for 1000 shading operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                triangle_gouraud_shading(vertex_normals, light_dir, ambient)
            end_time = time.perf_counter()
            total_times.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds

        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, vertex_normals, light_dir, ambient, name):
    """Run benchmark for Gouraud shading computation"""
    avg_time = benchmark.benchmark_calculation(vertex_normals, light_dir, ambient)
    print(f"\n{name}")
    print(f"Average time for 1000 shading operations: {avg_time:.2f} ms")


def run_benchmarks():
    """Run a series of benchmark tests covering different Gouraud shading scenarios."""
    benchmark = GouraudShadingBenchmark()

    # Test case 1: Uniform normals pointing up
    vertex_normals = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ambient = 0.1

    run_benchmark(
        benchmark,
        vertex_normals,
        light_dir,
        ambient,
        "Triangle with Uniform Normals Pointing Up",
    )

    # Test case 2: Varying normals at 45 degrees
    s = np.sqrt(2.0) / 2.0  # sin/cos of 45 degrees
    vertex_normals = np.array(
        [[s, 0.0, s], [0.0, s, s], [-s, 0.0, s]], dtype=np.float32
    )

    run_benchmark(
        benchmark,
        vertex_normals,
        light_dir,
        ambient,
        "Triangle with Varying 45-Degree Normals",
    )

    # Test case 3: Complex varying normals
    vertex_normals = np.array(
        [
            [0.577, 0.577, 0.577],  # Equal components
            [0.0, 0.0, 1.0],  # Straight up
            [1.0, 0.0, 0.0],  # Right facing
        ],
        dtype=np.float32,
    )

    run_benchmark(
        benchmark,
        vertex_normals,
        light_dir,
        ambient,
        "Triangle with Complex Varying Normals",
    )

    # Test case 4: Light from side with varying normals
    light_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Light from right side
    vertex_normals = np.array(
        [
            [1.0, 0.0, 0.0],  # Facing light
            [s, 0.0, s],  # 45 degrees
            [0.0, 0.0, 1.0],  # Perpendicular
        ],
        dtype=np.float32,
    )

    run_benchmark(
        benchmark, vertex_normals, light_dir, ambient, "Side Light with Varying Normals"
    )


if __name__ == "__main__":
    run_benchmarks()
