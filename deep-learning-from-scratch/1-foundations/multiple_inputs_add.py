import numpy as np
from numpy import ndarray

from derivative import deriv
from functions.sigmoid import sigmoid
from functions.square import square
from nested_func import Array_Function


def multiple_inputs_add(x: ndarray, y: ndarray, sigma: Array_Function) -> float:
    assert x.shape == y.shape

    a = x + y
    return sigma(a)


def multiple_inputs_add_backward(
    x: ndarray, y: ndarray, sigma: Array_Function
) -> float:
    a = x + y

    dsda = deriv(sigma, a)
    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    assert np.allclose(
        multiple_inputs_add(input_range, input_range, sigmoid), sigmoid(2 * input_range)
    )
    res = multiple_inputs_add_backward(input_range, input_range, square)
    assert np.allclose(res[0], 4 * input_range)
    assert np.allclose(res[1], 4 * input_range)
