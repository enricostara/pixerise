import time
import numpy as np
import pygame
from pixerise import Canvas, ViewPort, Renderer
from kernel.transforming_mod import transform_vertex


class TransformVertexBenchmark:
    def __init__(self):
        """Initialize benchmark setup"""
        canvas = Canvas((800, 600))
        viewport = ViewPort((800, 600), 1, canvas)
        scene = {}
        self.renderer = Renderer(canvas, viewport, scene)

        # Warm up JIT compilation
        vertex = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        translation = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rotation = np.array([np.pi/4, np.pi/4, np.pi/4], dtype=np.float32)
        scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        camera_translation = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        camera_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Run each transformation type once to compile
        transform_vertex(vertex, translation, np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3), False)  # Translation only
        transform_vertex(vertex, np.zeros(3), rotation, np.ones(3), np.zeros(3), np.zeros(3), False)  # Rotation only
        transform_vertex(vertex, np.zeros(3), np.zeros(3), scale, np.zeros(3), np.zeros(3), False)  # Scale only
        transform_vertex(vertex, translation, rotation, scale, np.zeros(3), np.zeros(3), False)  # Combined transform
        transform_vertex(vertex, translation, rotation, scale, camera_translation, camera_rotation, True)  # With camera

    def run_benchmark(self, test_name, transform_params):
        """Run a specific benchmark test"""
        vertex = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        translation = transform_params.get('translation', np.zeros(3, dtype=np.float32))
        rotation = transform_params.get('rotation', np.zeros(3, dtype=np.float32))
        scale = transform_params.get('scale', np.ones(3, dtype=np.float32))
        camera_translation = transform_params.get('camera_translation', np.zeros(3, dtype=np.float32))
        camera_rotation = transform_params.get('camera_rotation', np.zeros(3, dtype=np.float32))
        use_camera = transform_params.get('use_camera', False)

        times = []
        for _ in range(10):  # Run 10 loops
            start_time = time.time()
            for _ in range(1000):  # Transform 1000 vertices per loop
                transform_vertex(vertex, translation, rotation, scale, camera_translation, camera_rotation, use_camera)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        print(f"{test_name}: {avg_time:.6f} seconds")


def main():
    print("Running Vertex Transform Benchmarks...")
    print("Each test runs 10 loops of 1000 vertex transforms")
    print("Results show the average time taken for 1000 transforms")
    print("Warming up JIT compilation...")

    benchmark = TransformVertexBenchmark()
    print("\nRunning benchmarks...")

    # Test translation only
    benchmark.run_benchmark("Translation only", {
        'translation': np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Test rotation only
    benchmark.run_benchmark("Rotation only", {
        'rotation': np.array([np.pi/4, np.pi/4, np.pi/4], dtype=np.float32)
    })

    # Test scale only
    benchmark.run_benchmark("Scale only", {
        'scale': np.array([2.0, 2.0, 2.0], dtype=np.float32)
    })

    # Test combined transform
    benchmark.run_benchmark("Combined transform", {
        'translation': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'rotation': np.array([np.pi/4, np.pi/4, np.pi/4], dtype=np.float32),
        'scale': np.array([2.0, 2.0, 2.0], dtype=np.float32)
    })

    # Test with camera transform
    benchmark.run_benchmark("With camera transform", {
        'translation': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'rotation': np.array([np.pi/4, np.pi/4, np.pi/4], dtype=np.float32),
        'scale': np.array([2.0, 2.0, 2.0], dtype=np.float32),
        'camera_translation': np.array([0.0, 0.0, -10.0], dtype=np.float32),
        'camera_rotation': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'use_camera': True
    })


if __name__ == "__main__":
    main()
