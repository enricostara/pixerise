import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer
from kernel.transforming_mod import transform_vertex_normal


class TransformNormalBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.renderer = Renderer(self.canvas, self.viewport, self.scene)
        
        # Warm up JIT compilation
        print("Warming up JIT compilation...")
        normal = np.array([0.0, 1.0, 0.0])  # Unit normal pointing up
        rotation = np.array([0.1, 0.2, 0.3])
        camera_rotation = np.array([0.1, 0.1, 0.1])
        
        for _ in range(100):
            # Run all variants to ensure everything is compiled
            transform_vertex_normal(normal, rotation, camera_rotation, False)  # Without camera
            transform_vertex_normal(normal, rotation, camera_rotation, True)   # With camera
        print("JIT compilation completed")

        # Additional warm-up with simple case
        print("\nWarming up JIT compilation...")
        warm_up_normal = np.array([0.0, 1.0, 0.0])
        warm_up_rotation = np.zeros(3)
        warm_up_camera_rotation = np.zeros(3)
        for _ in range(100):
            transform_vertex_normal(warm_up_normal, warm_up_rotation, warm_up_camera_rotation, False)
        print("Warm-up complete.\n")

    def benchmark_transform(self, normal, rotation, camera_rotation, has_camera):
        """Run 10 loops of 1000 normal transforms and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times_jit = []

        for _ in range(NUM_LOOPS):
            # Benchmark JIT method
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                transform_vertex_normal(normal, rotation, camera_rotation, has_camera)
            end_time = time.perf_counter()
            total_times_jit.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return {
            'jit': np.mean(total_times_jit)
        }


def run_benchmark(benchmark, normal, rotation, camera_rotation, has_camera, name):
    """Run benchmark for normal transformation"""
    avg_times = benchmark.benchmark_transform(normal, rotation, camera_rotation, has_camera)
    
    print(f"\n{name}")
    print(f"JIT method: {avg_times['jit']:.3f}ms for 1000 calls")


def main():
    print("Running Normal Transform Benchmarks...")
    print("Each test runs 10 loops of 1000 normal transforms")
    print("Results show the average time taken for 1000 transforms")
    benchmark = TransformNormalBenchmark()
    
    # Test cases with a unit normal pointing up
    normal = np.array([0.0, 1.0, 0.0])
    
    # Test with rotation only, no camera
    rotation = np.array([np.pi/4, np.pi/4, np.pi/4])
    camera_rotation = np.zeros(3)
    run_benchmark(benchmark, normal, rotation, camera_rotation, False, "Rotation Only (No Camera)")
    
    # Test with rotation and camera
    camera_rotation = np.array([np.pi/6, np.pi/6, np.pi/6])
    run_benchmark(benchmark, normal, rotation, camera_rotation, True, "Rotation with Camera")
    
    # Test with extreme rotations
    rotation = np.array([np.pi, np.pi/2, np.pi/3])
    camera_rotation = np.array([np.pi/2, np.pi/3, np.pi/4])
    run_benchmark(benchmark, normal, rotation, camera_rotation, True, "Extreme Rotations with Camera")
    
    # Test with non-unit normal
    normal = np.array([2.0, 3.0, 4.0])  # Will be normalized during transformation
    run_benchmark(benchmark, normal, rotation, camera_rotation, True, "Non-unit Normal with Camera")


if __name__ == "__main__":
    main()
