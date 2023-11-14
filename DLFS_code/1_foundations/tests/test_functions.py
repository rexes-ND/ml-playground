import unittest

import numpy as np

from functions import (
    square,
    sigmoid,
    leaky_relu,
    deriv,
    chain,
)


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.input_range = np.arange(-2, 2, 0.01)

    def test_square(self):
        self.assertTrue(np.all(square(np.array([1, 2, 3])) == np.array([1, 4, 9])))

    def test_sigmoid(self):
        pass

    def test_leaky_relu(self):
        pass

    def test_deriv(self):
        # Testing if the derivative of square of x is close to 2*x.
        self.assertTrue(
            np.allclose(
                deriv(square, self.input_range),
                2 * self.input_range,
            )
        )

    def test_chain(self):
        self.assertTrue(
            np.allclose(
                chain([square, square], self.input_range),
                np.power(self.input_range, 4),
            )
        )


if __name__ == "__main__":
    print(__package__)
    unittest.main()
