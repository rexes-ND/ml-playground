from typing import Dict
import numpy as np
from numpy import ndarray
from typing import Dict, Tuple

def forward_linear_regression(X_batch: ndarray, y_batch: ndarray, weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:

    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

def loss_gradients(forward_info: Dict[str, ndarray], weights: Dict[str, ndarray], weights)