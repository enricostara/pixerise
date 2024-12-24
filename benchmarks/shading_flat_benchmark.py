import time
import numpy as np
from kernel.shading_mod import compute_flat_shading

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
        warm_up_indices = np.array([[0, 1, 2]], dtype=np.int32)
        warm_up_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        warm_up_light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        warm_up_material_color = np.array([255, 0, 0], dtype=np.uint8)
        warm_up_ambient = 0.1

        print("\nWarming up JIT compilation...")
        for _ in range(100):
            compute_flat_shading(
                warm_up_vertices,
                warm_up_indices,
                warm_up_normals,
                warm_up_light_dir,
                warm_up_material_color,
                warm_up_ambient
            )
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices, indices, normals, light_dir, material_color, ambient):
        """Run performance benchmark on flat shading computation.
        
        Performs 10 loops of 1000 shading operations each to get statistically significant timing data.
        
        Args:
            vertices: Array of vertex positions
            indices: Array of triangle indices
            normals: Array of triangle normals
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
                compute_flat_shading(vertices, indices, normals, light_dir, material_color, ambient)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls

def run_benchmark(benchmark, vertices, indices, normals, light_dir, material_color, ambient, name):
    """Run benchmark for flat shading computation"""
    avg_time = benchmark.benchmark_calculation(vertices, indices, normals, light_dir, material_color, ambient)
    print(f"\n{name}")
    print(f"Average time for 1000 shading operations: {avg_time:.2f} ms")

def run_benchmarks():
    """Run a series of benchmark tests covering different flat shading scenarios."""
    benchmark = ShadingBenchmark()

    # Test case 1: Single triangle
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    material_color = np.array([255, 0, 0], dtype=np.uint8)
    ambient = 0.1

    run_benchmark(
        benchmark, vertices, indices, normals, light_dir, material_color, ambient,
        "Single Triangle Test"
    )

    # Test case 2: Multiple triangles (cube - 12 triangles)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Front face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Back face
    ], dtype=np.float64)
    indices = np.array([
        [0, 1, 2], [0, 2, 3],  # Front face
        [1, 5, 6], [1, 6, 2],  # Right face
        [5, 4, 7], [5, 7, 6],  # Back face
        [4, 0, 3], [4, 3, 7],  # Left face
        [3, 2, 6], [3, 6, 7],  # Top face
        [4, 5, 1], [4, 1, 0]   # Bottom face
    ], dtype=np.int32)

    # Pre-compute normals for each triangle
    normals = np.zeros((12, 3), dtype=np.float64)
    for i in range(12):
        v0, v1, v2 = vertices[indices[i]]
        normal = np.cross(v1 - v0, v2 - v0)
        normals[i] = normal / np.linalg.norm(normal)

    run_benchmark(
        benchmark, vertices, indices, normals, light_dir, material_color, ambient,
        "Cube Test (12 triangles)"
    )

    # Test case 3: Large mesh (1000 triangles)
    num_triangles = 1000
    vertices = np.random.rand(num_triangles * 3, 3)  # Random vertices
    indices = np.arange(num_triangles * 3).reshape(-1, 3)
    normals = np.random.rand(num_triangles, 3)  # Random normals
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize

    run_benchmark(
        benchmark, vertices, indices, normals, light_dir, material_color, ambient,
        "Large Mesh Test (1000 triangles)"
    )

if __name__ == "__main__":
    run_benchmarks()
