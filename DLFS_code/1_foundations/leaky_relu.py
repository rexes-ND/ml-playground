from re import I
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def leaky_relu(x: ndarray, alpha: float = 0.2) -> ndarray:
    return np.maximum(alpha * x, x)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    plt.plot(input_range, leaky_relu(input_range))
    plt.show()
