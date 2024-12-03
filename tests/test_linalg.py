import unittest
from linalg import matmul

class TestMatrixMultiplication(unittest.TestCase):
    def test_empty_matrices(self):
        """Test handling of empty matrices"""
        with self.assertRaises(ValueError):
            matmul([], [])
        with self.assertRaises(ValueError):
            matmul([[]], [[]])
            
    def test_incompatible_dimensions(self):
        """Test matrices with incompatible dimensions"""
        a = [[1, 2], [3, 4]]  # 2x2
        b = [[1], [2], [3]]   # 3x1
        with self.assertRaises(ValueError):
            matmul(a, b)
            
    def test_basic_multiplication(self):
        """Test basic 2x2 matrix multiplication"""
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        expected = [[19, 22], [43, 50]]
        result = matmul(a, b)
        self.assertEqual(result, expected)
        
    def test_non_square_matrices(self):
        """Test multiplication of non-square matrices"""
        a = [[1, 2, 3]]           # 1x3
        b = [[4], [5], [6]]       # 3x1
        expected = [[32]]         # 1x1
        result = matmul(a, b)
        self.assertEqual(result, expected)
        
    def test_floating_point(self):
        """Test multiplication with floating point numbers"""
        a = [[0.5, 1.5], [2.5, 3.5]]
        b = [[1.0, 2.0], [3.0, 4.0]]
        expected = [[5.0, 7.0], [13.0, 19.0]]
        result = matmul(a, b)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
