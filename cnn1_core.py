import json

import numpy as np

from operations import (
    conv_forward,
    max_pool_forward,
    flatten,
    dense_forward,
    sigmoid,
    relu,
    conv_backward,
    max_pool_backward,
    dense_backward,
    relu_backward
)


# ============================================================
# Dropout
# ============================================================

def dropout_forward(A, keep_prob=1.0, training=True):
    if not training or keep_prob == 1.0:
        return A, None

    D = np.random.rand(*A.shape) < keep_prob
    A_drop = (A * D) / keep_prob

    return A_drop, D


def dropout_backward(dA, D, keep_prob=1.0):
    if D is None or keep_prob == 1.0:
        return dA

    return (dA * D) / keep_prob


# ============================================================
# Parameter Initialization
# ============================================================

def initialize_cnn1_parameters(input_shape=(64, 64, 1), seed=42, hidden_units=64):
    np.random.seed(seed)

    height, width, channels = input_shape

    parameters = {}

    parameters["W1"] = np.random.randn(3, 3, channels, 8) * np.sqrt(
        2 / (3 * 3 * channels)
    )
    parameters["b1"] = np.zeros((1, 1, 1, 8))

    parameters["W2"] = np.random.randn(3, 3, 8, 16) * np.sqrt(
        2 / (3 * 3 * 8)
    )
    parameters["b2"] = np.zeros((1, 1, 1, 16))

    flattened_size = 16 * 16 * 16

    parameters["W3"] = np.random.randn(flattened_size, hidden_units) * np.sqrt(
        2 / flattened_size
    )
    parameters["b3"] = np.zeros((1, hidden_units))

    parameters["W4"] = np.random.randn(hidden_units, 1) * np.sqrt(
        2 / hidden_units
    )
    parameters["b4"] = np.zeros((1, 1))

    return parameters


# ============================================================
# Forward Propagation
# ============================================================

def cnn1_forward(
    X,
    parameters,
    conv_keep_prob=1.0,
    dense_keep_prob=1.0,
    training=True
):
    """
    CNN1 architecture:

    X
    -> Conv1
    -> ReLU
    -> MaxPool
    -> Conv2
    -> ReLU
    -> Dropout
    -> MaxPool
    -> Flatten
    -> Dense 64
    -> ReLU
    -> Dropout
    -> Dense
    -> Sigmoid
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]

    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W3 = parameters["W3"]
    b3 = parameters["b3"]

    W4 = parameters["W4"]
    b4 = parameters["b4"]

    Z1, conv_cache1 = conv_forward(X, W1, b1, stride=1, pad=1)
    A1 = relu(Z1)

    P1, pool_cache1 = max_pool_forward(A1, pool_size=2, stride=2)

    Z2, conv_cache2 = conv_forward(P1, W2, b2, stride=1, pad=1)
    A2 = relu(Z2)

    A2_drop, dropout_cache = dropout_forward(
        A2,
        keep_prob=conv_keep_prob,
        training=training
    )

    P2, pool_cache2 = max_pool_forward(A2_drop, pool_size=2, stride=2)

    F = flatten(P2)

    Z3, dense_cache3 = dense_forward(F, W3, b3)
    A3 = relu(Z3)

    A3_drop, dropout_cache3 = dropout_forward(
        A3,
        keep_prob=dense_keep_prob,
        training=training
    )

    Z4, dense_cache4 = dense_forward(A3_drop, W4, b4)
    A4 = sigmoid(Z4)

    cache = {
        "Z1": Z1,
        "conv_cache1": conv_cache1,
        "pool_cache1": pool_cache1,

        "Z2": Z2,
        "conv_cache2": conv_cache2,
        "dropout_cache": dropout_cache,
        "pool_cache2": pool_cache2,

        "P2_shape": P2.shape,
        "dense_cache3": dense_cache3,
        "Z3": Z3,
        "dropout_cache3": dropout_cache3,
        "dense_cache4": dense_cache4,

        "A4": A4,
        "conv_keep_prob": conv_keep_prob,
        "dense_keep_prob": dense_keep_prob
    }

    return A4, cache


# ============================================================
# Cost Function with L2 Regularization
# ============================================================

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Binary cross-entropy cost.

    If lambd > 0, L2 regularization is added.
    """

    m = Y.shape[0]
    epsilon = 1e-8

    AL = np.clip(AL, epsilon, 1 - epsilon)

    data_cost = -(1 / m) * np.sum(
        Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
    )

    l2_cost = 0

    if parameters is not None and lambd > 0:
        for key in parameters:
            if key.startswith("W"):
                l2_cost += np.sum(np.square(parameters[key]))

        l2_cost = (lambd / (2 * m)) * l2_cost

    return data_cost + l2_cost


# ============================================================
# Backward Propagation
# ============================================================

def cnn1_backward(AL, Y, cache, parameters, lambd=0.0):
    """
    Backpropagation for CNN1.

    Includes:
    - dense backward
    - max pool backward
    - dropout backward
    - ReLU backward
    - convolution backward
    - L2 gradient correction
    """

    m = Y.shape[0]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]

    AL = np.clip(AL, 1e-8, 1 - 1e-8)

    dZ4 = AL - Y

    dA3_drop, dW4, db4 = dense_backward(dZ4, cache["dense_cache4"])

    if lambd > 0:
        dW4 += (lambd / m) * W4

    dA3 = dropout_backward(
        dA3_drop,
        cache["dropout_cache3"],
        cache["dense_keep_prob"]
    )

    dZ3 = relu_backward(dA3, cache["Z3"])

    dF, dW3, db3 = dense_backward(dZ3, cache["dense_cache3"])

    if lambd > 0:
        dW3 += (lambd / m) * W3

    dP2 = dF.reshape(cache["P2_shape"])

    dA2_drop = max_pool_backward(dP2, cache["pool_cache2"])

    dA2 = dropout_backward(
        dA2_drop,
        cache["dropout_cache"],
        cache["conv_keep_prob"]
    )

    dZ2 = relu_backward(dA2, cache["Z2"])

    dP1, dW2, db2 = conv_backward(dZ2, cache["conv_cache2"])

    if lambd > 0:
        dW2 += (lambd / m) * W2

    dA1 = max_pool_backward(dP1, cache["pool_cache1"])

    dZ1 = relu_backward(dA1, cache["Z1"])

    dX, dW1, db1 = conv_backward(dZ1, cache["conv_cache1"])

    if lambd > 0:
        dW1 += (lambd / m) * W1

    grads = {
        "dW1": dW1,
        "db1": db1,

        "dW2": dW2,
        "db2": db2,

        "dW3": dW3,
        "db3": db3,

        "dW4": dW4,
        "db4": db4
    }

    return grads


# ============================================================
# Adam Optimizer Core
# ============================================================

def initialize_adam(parameters):
    v = {}
    s = {}

    for key in parameters:
        v["d" + key] = np.zeros_like(parameters[key])
        s["d" + key] = np.zeros_like(parameters[key])

    return v, s


def adam_update(
    parameters,
    grads,
    v,
    s,
    t,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):
    """
    One Adam update step.

    This does not train the model by itself.
    It only updates parameters once using the current gradients.
    """

    for key in parameters:
        grad_key = "d" + key

        v[grad_key] = beta1 * v[grad_key] + (1 - beta1) * grads[grad_key]
        s[grad_key] = beta2 * s[grad_key] + (1 - beta2) * np.square(
            grads[grad_key]
        )

        v_corrected = v[grad_key] / (1 - beta1 ** t)
        s_corrected = s[grad_key] / (1 - beta2 ** t)

        parameters[key] -= learning_rate * v_corrected / (
            np.sqrt(s_corrected) + epsilon
        )

    return parameters, v, s


# ============================================================
# Prediction Helper
# ============================================================

def predict_cnn1(X, parameters):
    AL, _ = cnn1_forward(
        X,
        parameters,
        conv_keep_prob=1.0,
        dense_keep_prob=1.0,
        training=False
    )

    predictions = (AL >= 0.5).astype(int)

    return predictions, AL


def load_cnn1_parameters_from_json(file_path):
    """
    Loads CNN1 parameters saved by main_cnn1.py.

    The JSON file can contain either:
    - {"parameters": {...}}
    - or a raw parameter dictionary.
    """

    with open(file_path, "r") as file:
        payload = json.load(file)

    parameter_data = payload.get("parameters", payload)

    return {
        key: np.array(value, dtype=np.float64)
        for key, value in parameter_data.items()
    }


# ============================================================
# Quick Core Test
# ============================================================

if __name__ == "__main__":
    X = np.random.randn(4, 64, 64, 1)
    Y = np.array([[1], [0], [1], [0]])

    parameters = initialize_cnn1_parameters(input_shape=(64, 64, 1))

    AL, cache = cnn1_forward(
        X,
        parameters,
        conv_keep_prob=0.85,
        dense_keep_prob=0.75,
        training=True
    )

    cost = compute_cost(
        AL,
        Y,
        parameters=parameters,
        lambd=0.005
    )

    grads = cnn1_backward(
        AL,
        Y,
        cache,
        parameters,
        lambd=0.005
    )

    v, s = initialize_adam(parameters)

    parameters, v, s = adam_update(
        parameters,
        grads,
        v,
        s,
        t=1,
        learning_rate=0.001
    )

    print("AL shape:", AL.shape)
    print("Cost:", cost)

    for key in grads:
        print(key, grads[key].shape)
