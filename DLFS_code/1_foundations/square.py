import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    plt.plot(input_range, square(input_range))
    plt.show()
