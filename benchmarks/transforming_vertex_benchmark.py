import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer
from kernel.transforming_mod import transform_vertex


class TransformVertexBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.renderer = Renderer(self.canvas, self.viewport, self.scene)
        
        # Warm up JIT compilation
        print("Warming up JIT compilation...")
        vertex = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        translation = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rotation = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        camera_translation = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        camera_rotation = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        for _ in range(100):
            # Run all variants to ensure everything is compiled
            transform_vertex(vertex, translation, np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3), False)  # Translation only
            transform_vertex(vertex, np.zeros(3), rotation, np.ones(3), np.zeros(3), np.zeros(3), False)  # Rotation only
            transform_vertex(vertex, np.zeros(3), np.zeros(3), scale, np.zeros(3), np.zeros(3), False)  # Scale only
            transform_vertex(vertex, translation, rotation, scale, np.zeros(3), np.zeros(3), False)  # Combined transform
            transform_vertex(vertex, translation, rotation, scale, camera_translation, camera_rotation, True)  # With camera
        print("JIT compilation completed")

        # Additional warm-up with simple case
        print("\nWarming up JIT compilation...")
        warm_up_vertex = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        warm_up_translation = np.zeros(3, dtype=np.float32)
        warm_up_rotation = np.zeros(3, dtype=np.float32)
        warm_up_scale = np.ones(3, dtype=np.float32)
        for _ in range(100):
            transform_vertex(warm_up_vertex, warm_up_translation, warm_up_rotation, warm_up_scale, np.zeros(3), np.zeros(3), False)
        print("Warm-up complete.\n")

    def benchmark_transform(self, vertex, translation, rotation, scale, camera_translation, camera_rotation, has_camera):
        """Run 10 loops of 1000 vertex transforms and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times_jit = []

        for _ in range(NUM_LOOPS):
            # Benchmark JIT method
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                transform_vertex(vertex, translation, rotation, scale, camera_translation, camera_rotation, has_camera)
            end_time = time.perf_counter()
            total_times_jit.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return {
            'jit': np.mean(total_times_jit)
        }


def run_benchmark(benchmark, vertex, translation, rotation, scale, camera_translation, camera_rotation, has_camera, name):
    """Run benchmark for vertex transformation"""
    avg_times = benchmark.benchmark_transform(vertex, translation, rotation, scale, camera_translation, camera_rotation, has_camera)
    
    print(f"\n{name}")
    print(f"JIT method: {avg_times['jit']:.3f}ms for 1000 calls")


def main():
    print("Running Vertex Transform Benchmarks...")
    print("Each test runs 10 loops of 1000 vertex transforms")
    print("Results show the average time taken for 1000 transforms")
    benchmark = TransformVertexBenchmark()
    
    # Test vertex with various transformations
    vertex = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # Test with translation only
    translation = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    run_benchmark(benchmark, vertex, translation, np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3), False, "Translation Only")
    
    # Test with rotation only
    rotation = np.array([np.pi/4, np.pi/4, np.pi/4], dtype=np.float32)
    run_benchmark(benchmark, vertex, np.zeros(3), rotation, np.ones(3), np.zeros(3), np.zeros(3), False, "Rotation Only")
    
    # Test with scale only
    scale = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    run_benchmark(benchmark, vertex, np.zeros(3), np.zeros(3), scale, np.zeros(3), np.zeros(3), False, "Scale Only")
    
    # Test with combined transformations
    run_benchmark(benchmark, vertex, translation, rotation, scale, np.zeros(3), np.zeros(3), False, "Combined Transformations")
    
    # Test with camera transform
    camera_translation = np.array([0.0, 0.0, -10.0], dtype=np.float32)
    camera_rotation = np.array([0.0, np.pi/4, 0.0], dtype=np.float32)
    run_benchmark(benchmark, vertex, translation, rotation, scale, camera_translation, camera_rotation, True, "With Camera Transform")


if __name__ == "__main__":
    main()
