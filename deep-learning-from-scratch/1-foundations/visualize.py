import numpy as np
import matplotlib.pyplot as plt

from functions import (
    square,
    sigmoid,
    leaky_relu,
    chain,
    chain_deriv,
)

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)

    plt.figure(figsize=(24, 15))
    plt.subplot(2, 3, 1)
    plt.title("Square function")
    plt.plot(input_range, square(input_range))

    plt.subplot(2, 3, 2)
    plt.title("Sigmoid function")
    plt.plot(input_range, sigmoid(input_range), label="sigmoid")

    plt.subplot(2, 3, 3)
    plt.title("ReLU function")
    plt.plot(input_range, leaky_relu(input_range))

    input_range = np.arange(-3, 3, 0.01)

    plt.subplot(2, 3, 4)
    plt.title("Function and derivative for f(x) = sigmoid(square(x))")
    plt.plot(
        input_range,
        chain([square, sigmoid], input_range),
    )
    plt.plot(
        input_range,
        chain_deriv([square, sigmoid], input_range),
    )

    plt.subplot(2, 3, 5)
    plt.title("Function and derivative for f(x) = square(sigmoid(x))")
    plt.plot(
        input_range,
        chain([sigmoid, square], input_range),
    )
    plt.plot(
        input_range,
        chain_deriv([sigmoid, square], input_range),
    )

    plt.subplot(2, 3, 6)
    plt.title("Function and derivative for f(x) = square(sigmoid(leaky_relu(x)))")
    plt.plot(
        input_range,
        chain([leaky_relu, sigmoid, square], input_range),
    )
    plt.plot(
        input_range,
        chain_deriv([leaky_relu, sigmoid, square], input_range),
    )

    # plt.show()
    plt.savefig("visualization.png")
