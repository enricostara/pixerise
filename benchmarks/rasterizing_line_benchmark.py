import time
import numpy as np
from pixerise import Canvas, ViewPort, Renderer


class LineBenchmark:
    def __init__(self, width=1920, height=1080):
        self.canvas = Canvas((width, height))
        self.viewport = ViewPort((width, height), 1, self.canvas)
        self.renderer = Renderer(self.canvas, self.viewport)

        # Warm up JIT compilation with a simple case
        print("\nWarming up JIT compilation...")
        for _ in range(100):
            self.renderer.draw_line((0, 0, 0), (50, 50, 0), (255, 255, 255))
        print("Warm-up complete.\n")

    def benchmark_line(self, start, end, color=(255, 255, 255)):
        """Run 10 loops of 1000 line draws and return the average total time for 1000 calls"""
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 1000
        total_times = []

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                self.renderer.draw_line(start, end, color)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for 1000 calls


def run_benchmark(benchmark, p1, p2, name):
    """Run benchmark for line drawing"""
    avg_time = benchmark.benchmark_line(p1, p2)
    
    print(f"\n{name}")
    print(f"Average time for 1000 calls: {avg_time:.3f}ms")


def main():
    print("Running Line Drawing Benchmarks...")
    print("Each test runs 10 loops of 1000 line draws")
    print("Results show the average time taken for 1000 draws")
    benchmark = LineBenchmark()
    
    # Test horizontal line
    run_benchmark(benchmark, 
                 (-200, 0, 0), (200, 0, 0),
                 "Horizontal Line")
    
    # Test vertical line
    run_benchmark(benchmark,
                 (0, -200, 0), (0, 200, 0),
                 "Vertical Line")
    
    # Test diagonal line
    run_benchmark(benchmark,
                 (-100, -100, 0), (100, 100, 0),
                 "Diagonal Line")
    
    # Test short line
    run_benchmark(benchmark,
                 (0, 0, 0), (10, 10, 0),
                 "Short Line")
    
    # Test long diagonal line
    run_benchmark(benchmark,
                 (-300, -300, 0), (300, 300, 0),
                 "Long Diagonal Line")


if __name__ == "__main__":
    main()
