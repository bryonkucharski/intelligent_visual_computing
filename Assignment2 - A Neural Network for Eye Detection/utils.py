import numpy as np

def calculate_cost(Y_true, Y_hat):
    m = Y_true.shape[0]
    cost = (-1 / m) * np.sum((np.multiply(Y_true, np.log(Y_hat)) + (1 - Y_true) * np.log(1 - Y_hat)))
    return cost

def der_cost(Y_true, Y_hat):
    return -(np.divide(Y_true, Y_hat) - np.divide(1 - Y_true, 1 - Y_hat))

def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z, alpha = 0.01):
    return np.where(z > 0, z, z * alpha)

def relu_derivative(z, alpha = 0.01):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz

def calculate_backprop( derZ, A_previous, W, m):

    dW = np.dot(A_previous.T, derZ) / m
    db = (np.sum(derZ, axis=0, keepdims=True) / m)
    dA_prev = np.dot(derZ,W.T)

    assert (dA_prev.shape == A_previous.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db
