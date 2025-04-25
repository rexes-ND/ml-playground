import numpy as np
from numpy import ndarray

from .functions import Array_Function, deriv


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication.
    X = [x_1, x_2, ..., x_n]
    W = [w_1, w_2, ..., w_n]_t
    """

    assert X.shape[1] == W.shape[0], f"""
    For matrix multiplication, the number of columns in the first array
    should match the number of rows in the second; instead the number of columns
    in the first array is {X.shape[1]} and the number of rows in the second
    array is {W.shape[0]}
    """

    N = np.dot(X, W)
    return N


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the backward pass of a matrix multiplication with respect
    to the first argument

    X is row vector
    W is col vector
    dN/dX = d(x1*w1 + ... + xn*wn)/dx1, ..., d(x1*w1 + ... + xn*wn)/dxn
    = [w1, w2, ..., wn]
    """
    dNdX = np.transpose(W, (1, 0))
    return dNdX


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    """
    Computes the forward pass of a function involving matrix multiplication,
    one extra function.
    """
    N = matmul_forward(X, W)
    S = sigma(N)
    return S


def matrix_function_backward_1(
    X: ndarray, W: ndarray, sigma: Array_Function
) -> ndarray:
    """
    Computes the derivative of our matrix function with respect to the first element.
    """
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    # S = sigma(N)

    dSdN = deriv(sigma, N)

    dNdX = np.transpose(W, (1, 0))

    return dSdN * dNdX
