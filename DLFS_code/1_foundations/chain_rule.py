import numpy as np
from numpy import ndarray
from nested_func import Chain
from derivative import deriv
from square import square
from sigmoid import sigmoid
import matplotlib.pyplot as plt

def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 2
    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]

    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du

if __name__ == "__main__":
    input_range = np.arange(-2, 2, 0.01)

    chain = [square, sigmoid]
    plt.plot(input_range, sigmoid(square(input_range)))
    plt.plot(input_range, chain_deriv_2(chain, input_range))
    plt.legend(["Function for sigmoid(square(x)", "Derivative for sigmoid(square(x)"])
    plt.show()
