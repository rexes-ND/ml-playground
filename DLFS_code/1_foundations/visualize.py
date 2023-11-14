import numpy as np
import matplotlib.pyplot as plt

from functions import (
    square,
    sigmoid,
    leaky_relu,
)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)
    plt.plot(input_range, square(input_range), label="square")
    plt.plot(input_range, sigmoid(input_range), label="sigmoid")
    plt.plot(input_range, leaky_relu(input_range), label="leaky_relu")
    plt.legend()
    plt.show()
