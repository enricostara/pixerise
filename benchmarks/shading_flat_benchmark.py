import time
import numpy as np
from kernel.shading_mod import triangle_flat_shading

class ShadingBenchmark:
    """Benchmark class for measuring performance of flat shading computations."""
    
    def __init__(self):
        # Initialize with JIT warm-up using a simple case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        warm_up_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        warm_up_light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        warm_up_material_color = np.array([255, 0, 0], dtype=np.uint8)
        warm_up_ambient = 0.1

        print("\nWarming up JIT compilation...")
        for _ in range(100):
            triangle_flat_shading(
                warm_up_vertices,
                warm_up_normal,
                warm_up_light_dir,
                warm_up_material_color,
                warm_up_ambient
            )
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices, normal, light_dir, material_color, ambient):
        """Run performance benchmark on flat shading computation.
        
        Performs 10 loops of 1000 shading operations each to get statistically significant timing data.
        
        Args:
            vertices: Array of vertex positions for a single triangle
            normal: Normal vector for the triangle
            light_dir: Direction vector of the light source
            material_color: Color of the material (uint8 RGB values)
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
                triangle_flat_shading(vertices, normal, light_dir, material_color, ambient)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls

def run_benchmark(benchmark, vertices, normal, light_dir, material_color, ambient, name):
    """Run benchmark for flat shading computation"""
    avg_time = benchmark.benchmark_calculation(vertices, normal, light_dir, material_color, ambient)
    print(f"\n{name}")
    print(f"Average time for 1000 shading operations: {avg_time:.2f} ms")

def run_benchmarks():
    """Run a series of benchmark tests covering different flat shading scenarios."""
    benchmark = ShadingBenchmark()

    # Test case 1: Single triangle with normal pointing up
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    material_color = np.array([255, 0, 0], dtype=np.uint8)
    ambient = 0.1

    run_benchmark(
        benchmark, vertices, normal, light_dir, material_color, ambient,
        "Triangle with Normal Pointing Up"
    )

    # Test case 2: Single triangle with normal at 45 degrees
    normal = np.array([0.707, 0.0, 0.707], dtype=np.float64)  # 45 degrees to Z axis
    
    run_benchmark(
        benchmark, vertices, normal, light_dir, material_color, ambient,
        "Triangle with Normal at 45 Degrees"
    )

    # Test case 3: Single triangle with complex normal
    normal = np.array([0.577, 0.577, 0.577], dtype=np.float64)  # Equal components
    
    run_benchmark(
        benchmark, vertices, normal, light_dir, material_color, ambient,
        "Triangle with Complex Normal"
    )

    # Test case 4: Single triangle with grazing light
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    light_dir = np.array([0.0, 0.999, 0.001], dtype=np.float64)  # Almost horizontal light
    
    run_benchmark(
        benchmark, vertices, normal, light_dir, material_color, ambient,
        "Triangle with Grazing Light"
    )

if __name__ == "__main__":
    run_benchmarks()
