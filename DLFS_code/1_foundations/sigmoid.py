import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    plt.plot(input_range, sigmoid(input_range))
    plt.show()
