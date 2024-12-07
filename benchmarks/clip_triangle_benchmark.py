import time
import numpy as np
from kernel.clipping_mod import clip_triangle

class ClipTriangleBenchmark:
    """Benchmark class for measuring performance of triangle clipping operations against a plane."""
    
    def __init__(self):
        # Initialize with JIT warm-up using a simple triangle case
        # This ensures subsequent benchmarks measure actual performance, not JIT compilation time
        warm_up_vertices = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, -1.0, 1.0]
        ])
        warm_up_normal = np.array([0.0, 0.0, 1.0])
        print("\nWarming up JIT compilation...")
        for _ in range(100):
            clip_triangle(warm_up_normal, warm_up_vertices)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, plane_normal, vertices):
        """Run performance benchmark on triangle clipping operation.
        
        Performs 10 loops of 1000 clipping operations each to get statistically significant timing data.
        
        Args:
            plane_normal: Normal vector of the clipping plane
            vertices: Triangle vertices to be clipped
            
        Returns:
            float: Average time in milliseconds for 1000 clipping operations
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                clip_triangle(plane_normal, vertices)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls

def run_benchmarks():
    """Run a series of benchmark tests covering different triangle clipping scenarios."""
    benchmark = ClipTriangleBenchmark()
    
    # Define clipping plane (XY plane, normal pointing in +Z direction)
    plane_normal1 = np.array([0.0, 0.0, 1.0])
    
    # Test case 1: Triangle intersecting plane
    # This tests the most common case where the triangle straddles the clipping plane
    # Expected result: One triangle after clipping
    vertices1 = np.array([
        [1.0, 1.0, 1.0],    # Above plane
        [-1.0, 1.0, -1.0],  # Below plane
        [0.0, -1.0, 0.0]    # On plane
    ])
    time1 = benchmark.benchmark_calculation(plane_normal1, vertices1)
    print(f"Intersecting triangle case: {time1:.2f}ms for 1000 calls")

    # Test case 2: Triangle completely above plane
    # Tests the simple case where no clipping is needed
    # Expected result: Original triangle unchanged
    vertices2 = np.array([
        [1.0, 1.0, 2.0],
        [-1.0, 1.0, 2.0],
        [0.0, -1.0, 2.0]
    ])
    time2 = benchmark.benchmark_calculation(plane_normal1, vertices2)
    print(f"Above plane case: {time2:.2f}ms for 1000 calls")

    # Test case 3: Triangle exactly on plane
    # Tests edge case handling when triangle lies on the clipping plane
    # Expected result: Original triangle unchanged
    vertices3 = np.array([
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    time3 = benchmark.benchmark_calculation(plane_normal1, vertices3)
    print(f"On plane case: {time3:.2f}ms for 1000 calls")

    # Test case 4: Triangle with one vertex below and two above
    # Tests the complex case where clipping creates two new triangles
    # Expected result: Two triangles after clipping
    vertices4 = np.array([
        [0.0, 0.0, -1.0],  # Below plane
        [1.0, 0.0, 1.0],   # Above plane
        [-1.0, 0.0, 1.0]   # Above plane
    ])
    time4 = benchmark.benchmark_calculation(plane_normal1, vertices4)
    print(f"Two triangles case: {time4:.2f}ms for 1000 calls")

if __name__ == "__main__":
    run_benchmarks()
