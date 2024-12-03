import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class ShadedTriangleBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.renderer = Renderer(self.canvas, self.viewport, self.scene)

    def benchmark_shaded_triangle(self, p1, p2, p3, color=(255, 255, 255), i1=1.0, i2=1.0, i3=1.0):
        """Run 10 loops of 1000 shaded triangle draws and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer.draw_shaded_triangle(p1, p2, p3, color, i1, i2, i3)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, p1, p2, p3, intensities, name):
    """Run benchmark for shaded triangle drawing"""
    avg_time = benchmark.benchmark_shaded_triangle(p1, p2, p3, i1=intensities[0], i2=intensities[1], i3=intensities[2])
    
    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Shaded Triangle Drawing Benchmarks...")
    print("Each test runs 10 loops of 1000 shaded triangle draws")
    print("Results show the average time taken for 1000 draws")
    benchmark = ShadedTriangleBenchmark()
    
    # Test small equilateral triangle with uniform intensity
    radius = 50
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    run_benchmark(benchmark, points[0], points[1], points[2], [1.0, 1.0, 1.0], 
                 "Small Equilateral Triangle (Uniform Intensity)")
    
    # Test large equilateral triangle with gradient intensity
    radius = 300
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    run_benchmark(benchmark, points[0], points[1], points[2], [1.0, 0.5, 0.0],
                 "Large Equilateral Triangle (Gradient Intensity)")
    
    # Test flat bottom triangle with spotlight effect
    run_benchmark(benchmark, 
                 (0, 200), (-200, -200), (200, -200),
                 [0.2, 1.0, 0.2],
                 "Flat Bottom Triangle (Spotlight Effect)")
    
    # Test flat top triangle with inverse spotlight
    run_benchmark(benchmark,
                 (-200, 200), (200, 200), (0, -200),
                 [1.0, 1.0, 0.0],
                 "Flat Top Triangle (Inverse Spotlight)")
    
    # Test very thin triangle with alternating intensity
    run_benchmark(benchmark,
                 (0, 300), (20, -300), (40, 300),
                 [1.0, 0.0, 1.0],
                 "Very Thin Triangle (Alternating Intensity)")
    
    # Test very small triangle with low intensity
    run_benchmark(benchmark,
                 (0, 0), (2, 2), (0, 2),
                 [0.2, 0.2, 0.2],
                 "Very Small Triangle (Low Intensity)")
    
    # Test right triangle with horizontal gradient
    run_benchmark(benchmark,
                 (0, 0), (0, 200), (200, 0),
                 [0.0, 1.0, 1.0],
                 "Right Triangle (Horizontal Gradient)")
    
    # Test obtuse triangle with vertical gradient
    run_benchmark(benchmark,
                 (-300, 0), (300, 0), (0, 100),
                 [0.0, 0.0, 1.0],
                 "Obtuse Triangle (Vertical Gradient)")
    
    # Test degenerate line case
    run_benchmark(benchmark,
                 (0, 0), (100, 100), (200, 200),
                 [1.0, 0.5, 0.0],
                 "Degenerate Line (Gradient Intensity)")
    
    # Test degenerate point case
    run_benchmark(benchmark,
                 (0, 0), (0, 0), (0, 0),
                 [1.0, 1.0, 1.0],
                 "Degenerate Point (Uniform Intensity)")


if __name__ == "__main__":
    main()
