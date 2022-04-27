from typing import List

import numpy as np
from numpy import ndarray
from typing import Callable
from square import square

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

if __name__ == "__main__":
    chain = [square, square]
    input_range = np.arange(-2, 2, 0.01)
    assert np.allclose(chain_length_2(chain, input_range), np.power(input_range, 4))
