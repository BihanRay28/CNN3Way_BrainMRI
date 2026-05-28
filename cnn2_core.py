import json

import numpy as np

from operations import (
    conv_forward,
    flatten,
    dense_forward,
    sigmoid,
    relu,
    conv_backward,
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
# Same-Shape Max Pooling for Inception Pool Branches
# ============================================================

def max_pool_same_forward(X, pool_size=3, stride=1, pad=1):
    """
    Max pooling that keeps the spatial size unchanged for stride=1.

    This is used inside the Inception maxpool -> 1x1 conv branch so that
    all branches can be concatenated along the channel axis.
    """

    m, h_prev, w_prev, c_prev = X.shape
    f = pool_size

    h_new = int((h_prev + 2 * pad - f) / stride) + 1
    w_new = int((w_prev + 2 * pad - f) / stride) + 1

    X_pad = np.pad(
        X,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0
    )

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_prev):
                    x_slice = X_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]
                    A[i, h, w, c] = np.max(x_slice)

    cache = (X, X_pad, pool_size, stride, pad)

    return A, cache


def max_pool_same_backward(dA, cache):
    X, X_pad, pool_size, stride, pad = cache

    m, h_prev, w_prev, c_prev = X.shape
    _, h_new, w_new, _ = dA.shape
    f = pool_size

    dX_pad = np.zeros_like(X_pad)

    for i in range(m):
        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_prev):
                    x_slice = X_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]
                    mask = x_slice == np.max(x_slice)
                    dX_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ] += mask * dA[i, h, w, c]

    if pad == 0:
        return dX_pad

    return dX_pad[:, pad:-pad, pad:-pad, :]


# ============================================================
# Parameter Initialization
# ============================================================

def initialize_cnn2_parameters(
    input_shape=(64, 64, 1),
    seed=42,
    hidden_units=64,
    block1_filters=(4, 4, 4),
    block2_filters=(4, 4, 4)
):
    np.random.seed(seed)

    height, width, channels = input_shape
    b1_1x1, b1_3x3, b1_pool = block1_filters
    b2_1x1, b2_3x3, b2_pool = block2_filters

    parameters = {}

    parameters["W1_1x1"] = he_conv_weight(1, channels, b1_1x1)
    parameters["b1_1x1"] = np.zeros((1, 1, 1, b1_1x1))

    parameters["W1_reduce3"] = he_conv_weight(1, channels, b1_3x3)
    parameters["b1_reduce3"] = np.zeros((1, 1, 1, b1_3x3))

    parameters["W1_3x3"] = he_conv_weight(3, b1_3x3, b1_3x3)
    parameters["b1_3x3"] = np.zeros((1, 1, 1, b1_3x3))

    parameters["W1_poolproj"] = he_conv_weight(1, channels, b1_pool)
    parameters["b1_poolproj"] = np.zeros((1, 1, 1, b1_pool))

    block1_channels = b1_1x1 + b1_3x3 + b1_pool

    parameters["W2_1x1"] = he_conv_weight(1, block1_channels, b2_1x1)
    parameters["b2_1x1"] = np.zeros((1, 1, 1, b2_1x1))

    parameters["W2_reduce3"] = he_conv_weight(1, block1_channels, b2_3x3)
    parameters["b2_reduce3"] = np.zeros((1, 1, 1, b2_3x3))

    parameters["W2_3x3"] = he_conv_weight(3, b2_3x3, b2_3x3)
    parameters["b2_3x3"] = np.zeros((1, 1, 1, b2_3x3))

    parameters["W2_poolproj"] = he_conv_weight(1, block1_channels, b2_pool)
    parameters["b2_poolproj"] = np.zeros((1, 1, 1, b2_pool))

    block2_channels = b2_1x1 + b2_3x3 + b2_pool
    flattened_size = height * width * block2_channels

    parameters["W3"] = np.random.randn(flattened_size, hidden_units) * np.sqrt(
        2 / flattened_size
    )
    parameters["b3"] = np.zeros((1, hidden_units))

    parameters["W4"] = np.random.randn(hidden_units, 1) * np.sqrt(
        2 / hidden_units
    )
    parameters["b4"] = np.zeros((1, 1))

    return parameters


def he_conv_weight(filter_size, in_channels, out_channels):
    return np.random.randn(
        filter_size,
        filter_size,
        in_channels,
        out_channels
    ) * np.sqrt(2 / (filter_size * filter_size * in_channels))


# ============================================================
# Inception Block
# ============================================================

def inception_block_forward(X, parameters, prefix):
    W_1x1 = parameters[f"W{prefix}_1x1"]
    b_1x1 = parameters[f"b{prefix}_1x1"]

    W_reduce3 = parameters[f"W{prefix}_reduce3"]
    b_reduce3 = parameters[f"b{prefix}_reduce3"]

    W_3x3 = parameters[f"W{prefix}_3x3"]
    b_3x3 = parameters[f"b{prefix}_3x3"]

    W_poolproj = parameters[f"W{prefix}_poolproj"]
    b_poolproj = parameters[f"b{prefix}_poolproj"]

    Z_1x1, conv_cache_1x1 = conv_forward(X, W_1x1, b_1x1, stride=1, pad=0)
    A_1x1 = relu(Z_1x1)

    Z_reduce3, conv_cache_reduce3 = conv_forward(
        X,
        W_reduce3,
        b_reduce3,
        stride=1,
        pad=0
    )
    A_reduce3 = relu(Z_reduce3)

    Z_3x3, conv_cache_3x3 = conv_forward(
        A_reduce3,
        W_3x3,
        b_3x3,
        stride=1,
        pad=1
    )
    A_3x3 = relu(Z_3x3)

    P_pool, pool_cache = max_pool_same_forward(
        X,
        pool_size=3,
        stride=1,
        pad=1
    )

    Z_poolproj, conv_cache_poolproj = conv_forward(
        P_pool,
        W_poolproj,
        b_poolproj,
        stride=1,
        pad=0
    )
    A_poolproj = relu(Z_poolproj)

    A = np.concatenate([A_1x1, A_3x3, A_poolproj], axis=3)

    cache = {
        "Z_1x1": Z_1x1,
        "conv_cache_1x1": conv_cache_1x1,
        "Z_reduce3": Z_reduce3,
        "conv_cache_reduce3": conv_cache_reduce3,
        "Z_3x3": Z_3x3,
        "conv_cache_3x3": conv_cache_3x3,
        "pool_cache": pool_cache,
        "Z_poolproj": Z_poolproj,
        "conv_cache_poolproj": conv_cache_poolproj,
        "branch_channels": (
            A_1x1.shape[3],
            A_3x3.shape[3],
            A_poolproj.shape[3]
        )
    }

    return A, cache


def inception_block_backward(dA, cache, parameters, prefix, lambd=0.0):
    m = dA.shape[0]
    c_1x1, c_3x3, c_pool = cache["branch_channels"]

    dA_1x1 = dA[:, :, :, :c_1x1]
    dA_3x3 = dA[:, :, :, c_1x1:c_1x1 + c_3x3]
    dA_poolproj = dA[:, :, :, c_1x1 + c_3x3:c_1x1 + c_3x3 + c_pool]

    dZ_1x1 = relu_backward(dA_1x1, cache["Z_1x1"])
    dX_1x1, dW_1x1, db_1x1 = conv_backward(
        dZ_1x1,
        cache["conv_cache_1x1"]
    )

    dZ_3x3 = relu_backward(dA_3x3, cache["Z_3x3"])
    dA_reduce3, dW_3x3, db_3x3 = conv_backward(
        dZ_3x3,
        cache["conv_cache_3x3"]
    )

    dZ_reduce3 = relu_backward(dA_reduce3, cache["Z_reduce3"])
    dX_reduce3, dW_reduce3, db_reduce3 = conv_backward(
        dZ_reduce3,
        cache["conv_cache_reduce3"]
    )

    dZ_poolproj = relu_backward(dA_poolproj, cache["Z_poolproj"])
    dP_pool, dW_poolproj, db_poolproj = conv_backward(
        dZ_poolproj,
        cache["conv_cache_poolproj"]
    )
    dX_pool = max_pool_same_backward(dP_pool, cache["pool_cache"])

    if lambd > 0:
        dW_1x1 += (lambd / m) * parameters[f"W{prefix}_1x1"]
        dW_reduce3 += (lambd / m) * parameters[f"W{prefix}_reduce3"]
        dW_3x3 += (lambd / m) * parameters[f"W{prefix}_3x3"]
        dW_poolproj += (lambd / m) * parameters[f"W{prefix}_poolproj"]

    dX = dX_1x1 + dX_reduce3 + dX_pool

    grads = {
        f"dW{prefix}_1x1": dW_1x1,
        f"db{prefix}_1x1": db_1x1,
        f"dW{prefix}_reduce3": dW_reduce3,
        f"db{prefix}_reduce3": db_reduce3,
        f"dW{prefix}_3x3": dW_3x3,
        f"db{prefix}_3x3": db_3x3,
        f"dW{prefix}_poolproj": dW_poolproj,
        f"db{prefix}_poolproj": db_poolproj
    }

    return dX, grads


# ============================================================
# Forward Propagation
# ============================================================

def cnn2_forward(
    X,
    parameters,
    conv_keep_prob=1.0,
    dense_keep_prob=1.0,
    training=True
):
    """
    CNN2 Inception architecture:

    X
    -> Inception Block 1
       - 1x1 conv branch
       - 1x1 -> 3x3 conv branch
       - maxpool -> 1x1 conv branch
    -> Concatenate
    -> Dropout
    -> Inception Block 2
    -> Concatenate
    -> Dropout
    -> Flatten
    -> Dense 64
    -> ReLU
    -> Dropout
    -> Dense 1
    -> Sigmoid
    """

    I1, inception_cache1 = inception_block_forward(X, parameters, prefix="1")
    I1_drop, dropout_cache1 = dropout_forward(
        I1,
        keep_prob=conv_keep_prob,
        training=training
    )

    I2, inception_cache2 = inception_block_forward(
        I1_drop,
        parameters,
        prefix="2"
    )
    I2_drop, dropout_cache2 = dropout_forward(
        I2,
        keep_prob=conv_keep_prob,
        training=training
    )

    F = flatten(I2_drop)

    Z3, dense_cache3 = dense_forward(F, parameters["W3"], parameters["b3"])
    A3 = relu(Z3)

    A3_drop, dropout_cache3 = dropout_forward(
        A3,
        keep_prob=dense_keep_prob,
        training=training
    )

    Z4, dense_cache4 = dense_forward(
        A3_drop,
        parameters["W4"],
        parameters["b4"]
    )
    A4 = sigmoid(Z4)

    cache = {
        "inception_cache1": inception_cache1,
        "dropout_cache1": dropout_cache1,
        "inception_cache2": inception_cache2,
        "dropout_cache2": dropout_cache2,
        "I2_drop_shape": I2_drop.shape,
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

def cnn2_backward(AL, Y, cache, parameters, lambd=0.0):
    m = Y.shape[0]

    AL = np.clip(AL, 1e-8, 1 - 1e-8)

    dZ4 = AL - Y

    dA3_drop, dW4, db4 = dense_backward(dZ4, cache["dense_cache4"])

    if lambd > 0:
        dW4 += (lambd / m) * parameters["W4"]

    dA3 = dropout_backward(
        dA3_drop,
        cache["dropout_cache3"],
        cache["dense_keep_prob"]
    )

    dZ3 = relu_backward(dA3, cache["Z3"])

    dF, dW3, db3 = dense_backward(dZ3, cache["dense_cache3"])

    if lambd > 0:
        dW3 += (lambd / m) * parameters["W3"]

    dI2_drop = dF.reshape(cache["I2_drop_shape"])

    dI2 = dropout_backward(
        dI2_drop,
        cache["dropout_cache2"],
        cache["conv_keep_prob"]
    )

    dI1_drop, block2_grads = inception_block_backward(
        dI2,
        cache["inception_cache2"],
        parameters,
        prefix="2",
        lambd=lambd
    )

    dI1 = dropout_backward(
        dI1_drop,
        cache["dropout_cache1"],
        cache["conv_keep_prob"]
    )

    _, block1_grads = inception_block_backward(
        dI1,
        cache["inception_cache1"],
        parameters,
        prefix="1",
        lambd=lambd
    )

    grads = {}
    grads.update(block1_grads)
    grads.update(block2_grads)
    grads.update({
        "dW3": dW3,
        "db3": db3,
        "dW4": dW4,
        "db4": db4
    })

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

def predict_cnn2(X, parameters):
    AL, _ = cnn2_forward(
        X,
        parameters,
        conv_keep_prob=1.0,
        dense_keep_prob=1.0,
        training=False
    )

    predictions = (AL >= 0.5).astype(int)

    return predictions, AL


def load_cnn2_parameters_from_json(file_path):
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
    X = np.random.randn(2, 16, 16, 1)
    Y = np.array([[1], [0]])

    parameters = initialize_cnn2_parameters(
        input_shape=(16, 16, 1),
        block1_filters=(2, 2, 2),
        block2_filters=(2, 2, 2)
    )

    AL, cache = cnn2_forward(
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

    grads = cnn2_backward(
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
