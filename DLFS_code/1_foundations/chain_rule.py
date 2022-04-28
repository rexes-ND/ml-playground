import numpy as np
from numpy import ndarray
from nested_func import Chain
from derivative import deriv
from square import square
from sigmoid import sigmoid
from leaky_relu import leaky_relu
import matplotlib.pyplot as plt

def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 2
    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]

    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du

def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 3
    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    df1dx = deriv(f1, input_range)
    f1_of_x = f1(input_range)
    df2du = deriv(f2, f1_of_x)
    f2_of_f1_of_x = f2(f1_of_x)
    df3du = deriv(f3, f2_of_f1_of_x)

    return df1dx * df2du * df3du

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)

    # chain = [square, sigmoid]
    # plt.plot(input_range, sigmoid(square(input_range)))
    # plt.plot(input_range, chain_deriv_2(chain, input_range))
    # plt.legend(["Function for sigmoid(square(x)", "Derivative for sigmoid(square(x)"])
    # plt.show()

    chain = [leaky_relu, square, sigmoid]
    plt.plot(input_range, sigmoid(square(leaky_relu(input_range))))
    plt.plot(input_range, chain_deriv_3(chain, input_range))
    plt.legend(["Function for sigmoid(square(leaky_relu(x)))", "Derivative for sigmoid(square(leaky_relu(x)))"])
    plt.show()
    