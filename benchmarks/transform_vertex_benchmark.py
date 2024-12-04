import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class TransformVertexBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.renderer = Renderer(self.canvas, self.viewport, self.scene)
        
        # Warm up JIT compilation
        print("Warming up JIT compilation...")
        vertex = np.array([1.0, 2.0, 3.0])
        translation = np.array([4.0, 5.0, 6.0])
        rotation = np.array([0.1, 0.2, 0.3])
        scale = np.array([2.0, 2.0, 2.0])
        camera_translation = np.array([1.0, 1.0, 1.0])
        camera_rotation = np.array([0.1, 0.1, 0.1])
        
        for _ in range(100):
            # Run all variants to ensure everything is compiled
            self.renderer._transform_vertex(vertex, {'translation': translation})  # Translation only
            self.renderer._transform_vertex(vertex, {'rotation': rotation})  # Rotation only
            self.renderer._transform_vertex(vertex, {'scale': scale})  # Scale only
            self.renderer._transform_vertex(vertex, {'translation': translation, 'rotation': rotation, 'scale': scale})  # All transforms
            self.renderer._transform_vertex(vertex, {'translation': translation, 'rotation': rotation, 'scale': scale})  # With camera
        print("JIT compilation completed")

    def benchmark_transform(self, vertex, transform):
        """Run 10 loops of 1000 vertex transforms and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times_jit = []
        total_times_matrix = []

        # Pre-compute matrices for matrix-based method
        model_matrix = self.renderer.create_model_matrix(transform)
        camera_matrix = None
        if 'camera' in self.renderer._scene:
            camera_transform = self.renderer._scene['camera'].get('transform', {})
            camera_matrix = self.renderer.create_camera_matrix(camera_transform)

        for _ in range(NUM_LOOPS):
            # Benchmark JIT method
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer._transform_vertex(vertex, transform)
            end_time = time.perf_counter()
            total_times_jit.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Benchmark matrix method
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer._transform_vertex_with_matrices(vertex, transform, model_matrix, camera_matrix)
            end_time = time.perf_counter()
            total_times_matrix.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return {
            'jit': np.mean(total_times_jit),
            'matrix': np.mean(total_times_matrix)
        }


def run_benchmark(benchmark, vertex, transform, name):
    """Run benchmark for vertex transformation"""
    avg_times = benchmark.benchmark_transform(vertex, transform)
    
    print(f"\n{name}")
    print(f"JIT method:     {avg_times['jit']:.3f}ms for 1000 calls")
    print(f"Matrix method:  {avg_times['matrix']:.3f}ms for 1000 calls")
    print(f"Matrix/JIT ratio: {avg_times['matrix']/avg_times['jit']:.2f}x")


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
