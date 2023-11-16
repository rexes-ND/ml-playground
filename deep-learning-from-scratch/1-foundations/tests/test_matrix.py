import unittest

import numpy as np

from functions import sigmoid
from matrix import (
    matmul_backward_first,
    matrix_forward_extra,
    matrix_function_backward_1,
)


class TestMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(190203)
        self.X = np.random.randn(1, 3)
        self.W = np.random.randn(3, 1)

    def test_matmul_backward_first(self):
        self.assertTrue(
            np.allclose(
                matmul_backward_first(self.X, self.W),
                np.transpose(self.W, (1, 0)),
            )
        )

    def test_matmul_function_backward_1(self):
        self.assertAlmostEqual(
            (
                (
                    matrix_forward_extra(
                        self.X + np.array([[0.001, 0, 0]]), self.W, sigmoid
                    )
                    - matrix_forward_extra(
                        self.X - np.array([[0.001, 0, 0]]), self.W, sigmoid
                    )
                )
                / 0.002
            )[0, 0],
            matrix_function_backward_1(self.X, self.W, sigmoid)[0, 0],
            4,
        )
