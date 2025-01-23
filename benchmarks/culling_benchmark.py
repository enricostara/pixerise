import time
import numpy as np
from kernel.culling_mod import cull_back_faces

class CullBackFacesBenchmark:
    """Benchmark class for measuring performance of back-face culling operations."""
    
    def __init__(self):
        # Initialize with JIT warm-up using a simple triangle
        warm_up_vertices = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)
        warm_up_indices = np.array([[0, 1, 2]], dtype=np.int32)
        
        print("\nWarming up JIT compilation...")
        for _ in range(10):
            cull_back_faces(warm_up_vertices, warm_up_indices)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices, indices, camera_pos):
        """Run performance benchmark on back-face culling operation.
        
        Performs 10 loops of 1000 culling operations each to get statistically significant timing data.
        Vertices should be in camera space (relative to camera at origin).
        
        Args:
            vertices: Vertex positions array in camera space
            indices: Triangle indices array
            camera_pos: Ignored, vertices should already be in camera space
            
        Returns:
            float: Average time in milliseconds for 100 culling operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        # Transform vertices to camera space (camera at origin)
        vertices_camera = vertices - camera_pos

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                cull_back_faces(vertices_camera, indices)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return sum(total_times) / len(total_times)

def run_benchmark(benchmark, vertices, indices, camera_pos, name):
    """Run a single benchmark test and print results."""
    time_ms = benchmark.benchmark_calculation(vertices, indices, camera_pos)
    print(f"{name}: {time_ms:.2f}ms")

def run_benchmarks():
    """Run a series of benchmark tests covering different back-face culling scenarios."""
    print("Running Back-Face Culling Benchmarks...")
    print("Each test runs 10 loops of 1000 culling operations")
    print("Results show the average time taken for 1000 operations")
    
    benchmark = CullBackFacesBenchmark()
    
    # Test 1: Simple triangle
    vertices = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    camera_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    run_benchmark(benchmark, vertices, indices, camera_pos, "Single triangle")
    
    # Test 2: Small cube (6 faces)
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)
    indices = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [1, 5, 6], [1, 6, 2],  # Right
        [5, 4, 7], [5, 7, 6],  # Back
        [4, 0, 3], [4, 3, 7],  # Left
        [3, 2, 6], [3, 6, 7],  # Top
        [4, 5, 1], [4, 1, 0]   # Bottom
    ], dtype=np.int32)
    camera_pos = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    run_benchmark(benchmark, vertices, indices, camera_pos, "Simple cube")
    
    # Test 3: Small grid (4x4)
    grid_size = 4
    vertices = []
    indices = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2) * 0.5
            z = (j - grid_size/2) * 0.5
            y = np.sin(x) * np.cos(z) * 0.2
            vertices.append([x, y, z])
            
            if i < grid_size - 1 and j < grid_size - 1:
                idx = i * grid_size + j
                indices.append([idx, idx + 1, idx + grid_size])
                indices.append([idx + 1, idx + grid_size + 1, idx + grid_size])
    
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    camera_pos = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    run_benchmark(benchmark, vertices, indices, camera_pos, "Small grid")

if __name__ == "__main__":
    run_benchmarks()
