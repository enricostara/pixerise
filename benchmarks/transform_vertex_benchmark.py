import time
import numpy as np
from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort
from pixerise.rasterizer import Rasterizer


class TransformVertexBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.rasterizer = Rasterizer(self.canvas, self.viewport, self.scene)

    def benchmark_transform(self, vertex, transform):
        """Run 10 loops of 1000 vertex transforms and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.rasterizer._transform_vertex(vertex, transform)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, vertex, transform, name):
    """Run benchmark for vertex transformation"""
    avg_time = benchmark.benchmark_transform(vertex, transform)
    
    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Vertex Transform Benchmarks...")
    print("Each test runs 10 loops of 1000 vertex transforms")
    print("Results show the average time taken for 1000 transforms")
    benchmark = TransformVertexBenchmark()
    
    # Test vertex with translation only
    vertex = np.array([1.0, 2.0, 3.0])
    transform_translation = {
        'translation': np.array([10.0, 20.0, 30.0])
    }
    run_benchmark(benchmark, vertex, transform_translation, "Translation Only")
    
    # Test vertex with rotation only
    transform_rotation = {
        'rotation': np.array([np.pi/4, np.pi/4, np.pi/4])
    }
    run_benchmark(benchmark, vertex, transform_rotation, "Rotation Only")
    
    # Test vertex with scale only
    transform_scale = {
        'scale': np.array([2.0, 2.0, 2.0])
    }
    run_benchmark(benchmark, vertex, transform_scale, "Scale Only")
    
    # Test vertex with all transformations
    transform_all = {
        'translation': np.array([10.0, 20.0, 30.0]),
        'rotation': np.array([np.pi/4, np.pi/4, np.pi/4]),
        'scale': np.array([2.0, 2.0, 2.0])
    }
    run_benchmark(benchmark, vertex, transform_all, "All Transformations")
    
    # Test with camera transform
    benchmark.scene['camera'] = {
        'transform': {
            'translation': np.array([0.0, 0.0, -10.0]),
            'rotation': np.array([0.0, np.pi/4, 0.0])
        }
    }
    run_benchmark(benchmark, vertex, transform_all, "With Camera Transform")


if __name__ == "__main__":
    main()
