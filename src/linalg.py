"""
Linear algebra functions implemented in pure Python.
"""

def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 2D matrices.
    
    Args:
        a: First matrix as list of lists, shape (M, K)
        b: Second matrix as list of lists, shape (K, N)
        
    Returns:
        Result matrix as list of lists, shape (M, N)
        
    Raises:
        ValueError: If matrices have incompatible dimensions
    """
    if not a or not a[0] or not b or not b[0]:
        raise ValueError("Empty matrices are not allowed")
        
    m = len(a)      # rows in first matrix
    k = len(a[0])   # cols in first matrix
    n = len(b[0])   # cols in second matrix
    
    # Check if matrices can be multiplied
    if len(b) != k:
        raise ValueError(f"Matrix dimensions don't match: ({m},{k}) and ({len(b)},{n})")
    
    # Initialize result matrix with zeros
    result = [[0.0] * n for _ in range(m)]
    
    # Perform matrix multiplication using explicit loops
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            result[i][j] = total
            
    return result