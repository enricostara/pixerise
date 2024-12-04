import time
import numpy as np
from kernel import calculate_bounding_sphere


class BoundingSphereBenchmark:
    def __init__(self):
        # Warm up JIT compilation with a simple case
        warm_up_vertices = np.array([[1.0, 1.0, 1.0]])
        print("\nWarming up JIT compilation...")
        for _ in range(100):
            calculate_bounding_sphere(warm_up_vertices)
        print("Warm-up complete.\n")

    def benchmark_calculation(self, vertices):
        """Run 10 loops of 1000 bounding sphere calculations and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                calculate_bounding_sphere(vertices)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, vertices, name):
    """Run benchmark for bounding sphere calculation"""
    avg_time = benchmark.benchmark_calculation(vertices)
    
    print(f"\n{name}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Bounding Sphere Calculation Benchmarks...")
    print("Each test runs 10 loops of 1000 calculations")
    print("Results show the average time taken for 1000 calculations")
    benchmark = BoundingSphereBenchmark()
    
    # Test with single vertex
    vertices_single = np.array([[1.0, 2.0, 3.0]])
    run_benchmark(benchmark, vertices_single, "Single Vertex")
    
    # Test with cube vertices (8 points)
    vertices_cube = np.array([
        [0.0, 0.0, 0.0],  # Origin
        [1.0, 0.0, 0.0],  # Right
        [0.0, 1.0, 0.0],  # Up
        [0.0, 0.0, 1.0],  # Forward
        [1.0, 1.0, 0.0],  # Right-Up
        [1.0, 0.0, 1.0],  # Right-Forward
        [0.0, 1.0, 1.0],  # Up-Forward
        [1.0, 1.0, 1.0]   # Right-Up-Forward
    ])
    run_benchmark(benchmark, vertices_cube, "Cube (8 vertices)")
    
    # Test with sphere approximation (100 points)
    phi = np.linspace(0, 2*np.pi, 10)
    theta = np.linspace(0, np.pi, 10)
    vertices_sphere = []
    for p in phi:
        for t in theta:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            vertices_sphere.append([x, y, z])
    vertices_sphere = np.array(vertices_sphere)
    run_benchmark(benchmark, vertices_sphere, "Sphere Approximation (100 vertices)")
    
    # Test with large mesh (1000 random points)
    vertices_large = np.random.rand(1000, 3) * 10 - 5  # Points between -5 and 5
    run_benchmark(benchmark, vertices_large, "Large Mesh (1000 vertices)")
    
    # Test with very large mesh (10000 random points)
    vertices_very_large = np.random.rand(10000, 3) * 10 - 5  # Points between -5 and 5
    run_benchmark(benchmark, vertices_very_large, "Very Large Mesh (10000 vertices)")


if __name__ == "__main__":
    main()
