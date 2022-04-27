import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable

from square import square

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    plt.plot(input_range, deriv(square, input_range))
    plt.plot(input_range, 2*input_range)
    plt.legend(["Derivative of square function", "Linear function with coef 2"])
    plt.show()
