import time
import numpy as np
from kernel.culling_mod import cull_back_faces

class CullBackFacesBenchmark:
    """Benchmark class for measuring performance of back-face culling operations."""
    
    def __init__(self):
        # Initialize with JIT warm-up using a simple cube case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertices = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float32)
        warm_up_indices = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32)
        warm_up_camera = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        
        print("\nWarming up JIT compilation...")
        for _ in range(100):
            cull_back_faces(warm_up_vertices, warm_up_indices, warm_up_camera)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices, indices, camera_pos):
        """Run performance benchmark on back-face culling operation.
        
        Performs 10 loops of 1000 culling operations each to get statistically significant timing data.
        
        Args:
            vertices: Vertex positions array
            indices: Triangle indices array
            camera_pos: Camera position vector
            
        Returns:
            float: Average time in milliseconds for 1000 culling operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                cull_back_faces(vertices, indices, camera_pos)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls

def run_benchmark(benchmark, vertices, indices, camera_pos, name):
    """Run benchmark for back-face culling"""
    avg_time = benchmark.benchmark_calculation(vertices, indices, camera_pos)
    
    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")

def run_benchmarks():
    """Run a series of benchmark tests covering different back-face culling scenarios."""
    print("Running Back-Face Culling Benchmarks...")
    print("Each test runs 10 loops of 1000 culling operations")
    print("Results show the average time taken for 1000 operations")
    benchmark = CullBackFacesBenchmark()
    
    # Test case 1: Simple cube (6 faces, 12 triangles)
    vertices = np.array([
        # Front face
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
        # Back face
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
    ], dtype=np.float32)
    
    indices = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [4, 6, 5], [4, 7, 6],
        # Left face
        [0, 3, 7], [0, 7, 4],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [0, 4, 5], [0, 5, 1]
    ], dtype=np.int32)
    
    # Test from different camera positions
    camera_front = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # Should see front face
    run_benchmark(benchmark, vertices, indices, camera_front, "Camera Front View")
    
    camera_back = np.array([0.0, 0.0, -5.0], dtype=np.float32)  # Should see back face
    run_benchmark(benchmark, vertices, indices, camera_back, "Camera Back View")
    
    camera_angle = np.array([5.0, 5.0, 5.0], dtype=np.float32)  # Should see multiple faces
    run_benchmark(benchmark, vertices, indices, camera_angle, "Camera Angle View")
    
    # Test case 2: Dense mesh (100 faces)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X * np.pi) * np.cos(Y * np.pi)
    
    dense_vertices = []
    dense_indices = []
    idx = 0
    
    for i in range(9):
        for j in range(9):
            # Add 4 vertices for each grid cell
            v1 = [X[i,j], Y[i,j], Z[i,j]]
            v2 = [X[i,j+1], Y[i,j+1], Z[i,j+1]]
            v3 = [X[i+1,j+1], Y[i+1,j+1], Z[i+1,j+1]]
            v4 = [X[i+1,j], Y[i+1,j], Z[i+1,j]]
            
            dense_vertices.extend([v1, v2, v3, v4])
            
            # Add 2 triangles
            dense_indices.extend([[idx, idx+1, idx+2], [idx, idx+2, idx+3]])
            idx += 4
    
    dense_vertices = np.array(dense_vertices, dtype=np.float32)
    dense_indices = np.array(dense_indices, dtype=np.int32)
    
    run_benchmark(benchmark, dense_vertices, dense_indices, camera_front, "Dense Mesh (100 faces)")

if __name__ == "__main__":
    run_benchmarks()
