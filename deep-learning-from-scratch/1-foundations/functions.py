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


def deriv(
    f: Array_Function,
    input_: ndarray,
    delta: float = 0.001,
) -> ndarray:
    return (f(input_ + delta) - f(input_ - delta)) / (2 * delta)


def chain(cs: Chain, x: ndarray) -> ndarray:
    y = x.copy()
    for c in cs:
        y = c(y)
    return y
