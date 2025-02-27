import numpy as np
from typing import Callable

# Type alias for activation function
Array_Function = Callable[[np.ndarray], np.ndarray]

def deriv(func: Array_Function, x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the activation function.
    """
    if func == sigmoid:
        return func(x) * (1 - func(x))
    else:
        raise NotImplementedError("Derivative not implemented for this function.")

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def matrix_function_forward_sum(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> float:
    """
    Forward pass: Compute sum of all elements after applying activation.
    """
    return np.sum(sigma(np.dot(X, W)))

def matrix_function_backward_sum_1(X: np.ndarray, W: np.ndarray, sigma: Array_Function) -> np.ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the first matrix input.
    '''
    assert X.shape[1] == W.shape[0]
    
    # Matrix multiplication
    N = np.dot(X, W)
    
    S = sigma(N)
    
    # Summing all elements
    L = np.sum(S)
    
    # Backpropagation steps
    dLdS = np.ones_like(S)  # dL/dS
    dSdN = deriv(sigma, N)  # dS/dN
    dLdN = dLdS * dSdN      # dL/dN
    dNdX = np.transpose(W)  # dN/dX
    dLdX = np.dot(dLdN, dNdX)  # dL/dX

    return dLdX

# Verification
np.random.seed(190204)
X = np.random.randn(4, 4)
W = np.random.randn(4, 2)

print("X:")
print(X)
print("\nL:")
print(round(matrix_function_forward_sum(X, W, sigmoid), 4))

print("\ndLdX:")
print(matrix_function_backward_sum_1(X, W, sigmoid))
