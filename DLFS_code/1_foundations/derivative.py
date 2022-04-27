import numpy as np
from numpy import ndarray
from typing import Callable

from square import square

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    assert np.allclose(deriv(square, input_range), 2 * input_range) 
