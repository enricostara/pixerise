import time
import numpy as np
from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort
from pixerise.rasterizer import Rasterizer


class LineBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.scene = {}
        self.rasterizer = Rasterizer(self.canvas, self.viewport, self.scene)

    def benchmark_line(self, start, end, color=(255, 255, 255), iterations=1000):
        """Benchmark drawing a single line multiple times and return average time"""
        total_time = 0
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.rasterizer.draw_line(start, end, color)
            end_time = time.perf_counter()
            total_time += (end_time - start_time) * 1000  # Convert to milliseconds
            
        return total_time / iterations  # Return average time per line


def run_benchmark(benchmark, start, end, name):
    """Run benchmark for line drawing"""
    time_bresenham = benchmark.benchmark_line(start, end)
    
    print(f"\n{name}")
    print(f"Average time: {time_bresenham:.3f}ms")


def main():
    print("Running Line Drawing Benchmarks...")
    print(f"Each test draws the line 1000 times and reports the average time")
    benchmark = LineBenchmark()
    
    # Test horizontal line
    run_benchmark(benchmark, (-450, 0), (450, 0), "Horizontal Line")
    
    # Test vertical line
    run_benchmark(benchmark, (0, -250), (0, 250), "Vertical Line")
    
    # Test diagonal line
    run_benchmark(benchmark, (-250, -250), (250, 250), "Diagonal Line")
    
    # Test short line
    run_benchmark(benchmark, (-50, -50), (50, 50), "Short Line")
    
    # Test long diagonal line
    run_benchmark(benchmark, (-450, -250), (450, 250), "Long Diagonal Line")


if __name__ == "__main__":
    main()
