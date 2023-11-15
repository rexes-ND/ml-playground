import numpy as np
from numpy import ndarray

from nested_func import Array_Function
from derivative import deriv
from functions.square import square
from functions.sigmoid import sigmoid


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    return N


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    dNdX = np.transpose(W, (1, 0))
    return dNdX


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    return S


def matrix_function_backward_1(
    X: ndarray, W: ndarray, sigma: Array_Function
) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)

    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)


def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_Function) -> float:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)

    return L


def matrix_function_backward_sum_1(
    X: ndarray, W: ndarray, sigma: Array_Function
) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    # L = np.sum(S)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    # dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dSdN, dNdX)

    return dLdX


if __name__ == "__main__":
    np.random.seed(190204)
    X = np.random.rand(1, 3)
    W = np.random.rand(3, 1)

    assert np.allclose(
        2 * np.transpose(W) * matmul_forward(X, W),
        matrix_function_backward_1(X, W, square),
    )

    X = np.random.randn(3, 3)
    W = np.random.randn(3, 2)

    print(np.round(matrix_function_backward_sum_1(X, W, sigmoid), 4))

    X1 = X.copy()
    X1[0, 0] += 0.001

    print(
        round(
            (
                matrix_function_forward_sum(X1, W, sigmoid)
                - matrix_function_forward_sum(X, W, sigmoid)
            )
            / 0.001,
            4,
        )
    )
