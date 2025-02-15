import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class ShadedTriangleBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.renderer = Renderer(self.canvas, self.viewport)

        # Warm up JIT compilation with a simple case
        print("\nWarming up JIT compilation...")
        p1, p2, p3 = (0, 0, 0.5), (50, 0, 0.5), (25, 50, 0.5)
        color = (255, 255, 255)
        for _ in range(100):
            self.renderer.draw_shaded_triangle(p1, p2, p3, color, 1.0, 1.0, 1.0)
        print("Warm-up complete.\n")

    def benchmark_shaded_triangle(
        self, p1, p2, p3, color=(255, 255, 255), i1=1.0, i2=1.0, i3=1.0
    ):
        """Run 10 loops of 1000 shaded triangle draws and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer.draw_shaded_triangle(p1, p2, p3, color, i1, i2, i3)
            end_time = time.perf_counter()
            total_times.append(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds

        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, p1, p2, p3, intensities, name):
    """Run benchmark for shaded triangle drawing"""
    avg_time = benchmark.benchmark_shaded_triangle(
        p1, p2, p3, i1=intensities[0], i2=intensities[1], i3=intensities[2]
    )

    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Shaded Triangle Drawing Benchmarks...")
    print("Each test runs 10 loops of 1000 shaded triangle draws")
    print("Results show the average time taken for 1000 draws")
    benchmark = ShadedTriangleBenchmark()

    # Test small equilateral triangle with uniform intensity
    radius = 50
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    points = [(radius * np.cos(a), radius * np.sin(a), 0.5) for a in angles]
    run_benchmark(
        benchmark,
        points[0],
        points[1],
        points[2],
        [1.0, 1.0, 1.0],
        "Small Equilateral Triangle (Uniform Intensity)",
    )

    # Test large equilateral triangle with gradient intensity
    radius = 300
    points = [(radius * np.cos(a), radius * np.sin(a), 0.5) for a in angles]
    run_benchmark(
        benchmark,
        points[0],
        points[1],
        points[2],
        [1.0, 0.5, 0.0],
        "Large Equilateral Triangle (Gradient Intensity)",
    )

    # Test flat bottom triangle with spotlight effect and varying depth
    run_benchmark(
        benchmark,
        (0, 200, 0.3),
        (-200, -200, 0.5),
        (200, -200, 0.7),
        [0.2, 1.0, 0.2],
        "Flat Bottom Triangle (Spotlight Effect, Varying Depth)",
    )

    # Test flat top triangle with inverse spotlight and steep depth gradient
    run_benchmark(
        benchmark,
        (-200, 200, 0.1),
        (200, 200, 0.9),
        (0, -200, 0.5),
        [1.0, 1.0, 0.0],
        "Flat Top Triangle (Inverse Spotlight, Steep Depth)",
    )

    # Test very thin triangle with alternating intensity and uniform depth
    run_benchmark(
        benchmark,
        (0, 300, 0.5),
        (20, -300, 0.5),
        (40, 300, 0.5),
        [1.0, 0.0, 1.0],
        "Very Thin Triangle (Alternating Intensity, Uniform Depth)",
    )

    # Test very small triangle with low intensity and near depth
    run_benchmark(
        benchmark,
        (0, 0, 0.1),
        (2, 2, 0.1),
        (0, 2, 0.1),
        [0.2, 0.2, 0.2],
        "Very Small Triangle (Low Intensity, Near Depth)",
    )

    # Test right triangle with horizontal gradient and depth gradient
    run_benchmark(
        benchmark,
        (0, 0, 0.2),
        (0, 200, 0.5),
        (200, 0, 0.8),
        [0.0, 1.0, 1.0],
        "Right Triangle (Horizontal Gradient, Depth Gradient)",
    )

    # Test obtuse triangle with vertical gradient and uniform far depth
    run_benchmark(
        benchmark,
        (-300, 0, 0.9),
        (300, 0, 0.9),
        (0, 100, 0.9),
        [0.0, 0.0, 1.0],
        "Obtuse Triangle (Vertical Gradient, Far Depth)",
    )

    # Test degenerate line case with varying depth
    run_benchmark(
        benchmark,
        (0, 0, 0.1),
        (1, 1, 0.5),
        (2, 2, 0.9),
        [1.0, 0.5, 0.0],
        "Degenerate Line Case (Varying Depth)",
    )

    # Test overlapping triangles with different depths
    print("\nBenchmarking overlapping triangles with z-buffer...")
    start_time = time.perf_counter()
    for _ in range(1000):
        # Draw back triangle
        benchmark.renderer.draw_shaded_triangle(
            (-100, -100, 0.8),
            (100, -100, 0.8),
            (0, 100, 0.8),
            (255, 0, 0),
            1.0,
            1.0,
            1.0,
        )
        # Draw front triangle
        benchmark.renderer.draw_shaded_triangle(
            (-50, -50, 0.2), (50, -50, 0.2), (0, 50, 0.2), (0, 255, 0), 1.0, 1.0, 1.0
        )
    end_time = time.perf_counter()
    print(
        f"Average time for 1000 overlapping triangle pairs: {(end_time - start_time) * 1000:.3f}ms"
    )


if __name__ == "__main__":
    main()
