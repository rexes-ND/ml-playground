from typing import Callable, List

import numpy as np
from numpy import ndarray

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]


def square(x: ndarray) -> ndarray:
    return np.power(x, 2)


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def leaky_relu(x: ndarray, alpha: float = 0.2) -> ndarray:
    return np.maximum(alpha * x, x)


def deriv(f: Array_Function, input_: ndarray, delta: float = 0.001) -> ndarray:
    return (f(input_ + delta) - f(input_ - delta)) / (2 * delta)


def chain(cs: Chain, x: ndarray) -> ndarray:
    y = x.copy()
    for c in cs:
        y = c(y)
    return y


def chain_deriv(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x)))' = f2'(f1(x)) * f1'(x)
    (fn(...(f2(f1(x)))))' = fn'(...) * ... * f2'(f1(x)) * f1'(x)
    """
    res = 1
    cur_input_range = input_range.copy()
    for c in chain:
        res *= deriv(c, cur_input_range)
        cur_input_range = c(cur_input_range)
    return res


def multiple_inputs_add(
    x: ndarray,
    y: ndarray,
    sigma: Array_Function,
) -> float:
    """
    Function with multiple inputs and addition, forward pass.
    a = alpha(x, y) = x + y
    s = sigma(a)
    """
    assert x.shape == y.shape

    a = x + y
    return sigma(a)


def multiple_inputs_add_backward(
    x: ndarray,
    y: ndarray,
    sigma: Array_Function,
):
    """
    Computes the derivative of this simple function with respect to both inputs.
    a = alpha(x, y) = x + y
    s = sigma(a)

    d(sigma(alpha(x, y)))/dx = sigma'(a) * d(alpha(x, y))/dx
    d(alpha(x, y))/dx = d(x + y)/dx = 1
    """
    a = x + y
    dsda = deriv(sigma, a)
    dadx, dady = 1, 1
    return dsda * dadx, dsda * dady
