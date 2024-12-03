import time
import numpy as np
import sys
from linalg import matmul


class LinAlgBenchmark:
    def benchmark_matmul(self, size: int, use_numpy: bool = False):
        """Run 10 loops of matrix multiplication and return the average total time
        
        Args:
            size: Size of the square matrices to multiply (size x size)
            use_numpy: If True, use numpy's matmul instead of our implementation
        """
        NUM_LOOPS = 10
        CALLS_PER_LOOP = 100  # Reduced from 1000 since matrix mult is more intensive
        total_times = []

        # Create random matrices
        if use_numpy:
            matrix_a = np.random.random((size, size))
            matrix_b = np.random.random((size, size))
        else:
            matrix_a = [[np.random.random() for _ in range(size)] for _ in range(size)]
            matrix_b = [[np.random.random() for _ in range(size)] for _ in range(size)]

        for _ in range(NUM_LOOPS):
            start_time = time.perf_counter()
            for _ in range(CALLS_PER_LOOP):
                if use_numpy:
                    np.matmul(matrix_a, matrix_b)
                else:
                    matmul(matrix_a, matrix_b)
            end_time = time.perf_counter()
            total_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        return np.mean(total_times)  # Return average time for CALLS_PER_LOOP calls


def run_benchmark(benchmark, size: int):
    """Run benchmark comparing our implementation vs numpy"""
    our_time = benchmark.benchmark_matmul(size, use_numpy=False)
    numpy_time = benchmark.benchmark_matmul(size, use_numpy=True)
    
    print(f"\nMatrix size: {size}x{size}")
    print(f"Our implementation:")
    print(f"  Average time for 100 multiplications: {our_time:.3f}ms")
    print(f"  Average time per multiplication: {our_time/100:.3f}ms")
    print(f"NumPy implementation:")
    print(f"  Average time for 100 multiplications: {numpy_time:.3f}ms")
    print(f"  Average time per multiplication: {numpy_time/100:.3f}ms")
    print(f"Performance ratio (NumPy is {our_time/numpy_time:.1f}x faster)")


def main():
    print("Running Linear Algebra Benchmarks...")
    print("Comparing our implementation vs NumPy")
    print("Each test runs 10 loops of 100 matrix multiplications")
    benchmark = LinAlgBenchmark()
    
    # Test different matrix sizes
    run_benchmark(benchmark, 2)   # Small matrices
    run_benchmark(benchmark, 4)   # 4x4 matrices
    run_benchmark(benchmark, 5)   # Medium matrices
    run_benchmark(benchmark, 10)  # Large matrices
    run_benchmark(benchmark, 20)  # Very large matrices
    run_benchmark(benchmark, 50)  # Huge matrices


if __name__ == "__main__":
    main()
