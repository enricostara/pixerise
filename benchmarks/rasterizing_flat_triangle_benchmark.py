import time
import numpy as np
from pixerise.pixerise import Canvas, ViewPort, Renderer


class TriangleBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.renderer = Renderer(self.canvas, self.viewport)

        # Warm up JIT compilation with a simple case
        print("\nWarming up JIT compilation...")
        p1, p2, p3 = (0, 0, 0), (50, 0, 0), (25, 50, 0)
        color = (255, 255, 255)
        for _ in range(100):
            self.renderer.draw_triangle(p1, p2, p3, color)
        print("Warm-up complete.\n")

    def benchmark_triangle(self, p1, p2, p3, color=(255, 255, 255)):
        """Run 10 loops of 1000 triangle draws and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer.draw_triangle(p1, p2, p3, color)
            end_time = time.perf_counter()
            total_times.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds

        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, p1, p2, p3, name):
    """Run benchmark for triangle drawing"""
    avg_time = benchmark.benchmark_triangle(p1, p2, p3)

    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Triangle Drawing Benchmarks...")
    print("Each test runs 10 loops of 1000 triangle draws")
    print("Results show the average time taken for 1000 draws")
    benchmark = TriangleBenchmark()

    # Test small equilateral triangle
    radius = 50
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    points = [(radius * np.cos(a), radius * np.sin(a), 0) for a in angles]
    run_benchmark(
        benchmark, points[0], points[1], points[2], "Small Equilateral Triangle"
    )

    # Test large equilateral triangle
    radius = 300
    points = [(radius * np.cos(a), radius * np.sin(a), 0) for a in angles]
    run_benchmark(
        benchmark, points[0], points[1], points[2], "Large Equilateral Triangle"
    )

    # Test flat bottom triangle
    run_benchmark(
        benchmark, (0, 200, 0), (-200, -200, 0), (200, -200, 0), "Flat Bottom Triangle"
    )

    # Test flat top triangle
    run_benchmark(
        benchmark, (-200, 200, 0), (200, 200, 0), (0, -200, 0), "Flat Top Triangle"
    )

    # Test very thin triangle
    run_benchmark(
        benchmark, (0, 300, 0), (20, -300, 0), (40, 300, 0), "Very Thin Triangle"
    )

    # Test very small triangle
    run_benchmark(benchmark, (0, 0, 0), (2, 2, 0), (0, 2, 0), "Very Small Triangle")

    # Test right triangle
    run_benchmark(benchmark, (0, 0, 0), (0, 200, 0), (200, 0, 0), "Right Triangle")

    # Test obtuse triangle
    run_benchmark(benchmark, (-300, 0, 0), (300, 0, 0), (0, 100, 0), "Obtuse Triangle")


if __name__ == "__main__":
    main()
