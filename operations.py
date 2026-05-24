
import numpy as np


def zero_pad(X, pad):
    """
    X shape: (m, h, w, c)
    pad: int
    """
    return np.pad(
        X,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0
    )


def relu(Z):
    """
    Z: any shape
    """
    return np.maximum(0, Z)


def sigmoid(Z):
    """
    Z: any shape
    """
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def conv_single_step(a_slice, W, b):
    """
    a_slice shape: (f, f, c_prev)
    W shape: (f, f, c_prev)
    b shape: scalar or (1, 1, 1)
    """
    return np.sum(a_slice * W) + float(b)


def conv_forward(X, W, b, stride=1, pad=0):
    """
    X shape: (m, h_prev, w_prev, c_prev)
    W shape: (f, f, c_prev, c_new)
    b shape: (1, 1, 1, c_new)

    Returns:
    Z shape: (m, h_new, w_new, c_new)
    """

    m, h_prev, w_prev, c_prev = X.shape
    f, _, c_prev_w, c_new = W.shape

    assert c_prev == c_prev_w

    h_new = int((h_prev + 2 * pad - f) / stride) + 1
    w_new = int((w_prev + 2 * pad - f) / stride) + 1

    Z = np.zeros((m, h_new, w_new, c_new))
    X_pad = zero_pad(X, pad)

    for i in range(m):
        x_i = X_pad[i]

        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_new):
                    x_slice = x_i[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(
                        x_slice,
                        W[:, :, :, c],
                        b[:, :, :, c]
                    )

    cache = (X, W, b, stride, pad)

    return Z, cache


def max_pool_forward(X, pool_size=2, stride=2):
    """
    X shape: (m, h_prev, w_prev, c_prev)

    Returns:
    A shape: (m, h_new, w_new, c_prev)
    """

    m, h_prev, w_prev, c_prev = X.shape
    f = pool_size

    h_new = int((h_prev - f) / stride) + 1
    w_new = int((w_prev - f) / stride) + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_prev):
                    x_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    A[i, h, w, c] = np.max(x_slice)

    cache = (X, pool_size, stride)

    return A, cache


def flatten(X):
    """
    X shape: (m, h, w, c)

    Returns:
    X_flat shape: (m, h*w*c)
    """
    return X.reshape(X.shape[0], -1)


def dense_forward(A_prev, W, b):
    """
    A_prev shape: (m, n_prev)
    W shape: (n_prev, n_new)
    b shape: (1, n_new)
    """
    Z = np.dot(A_prev, W) + b
    cache = (A_prev, W, b)

    return Z, cache

def relu_backward(dA, Z):
    """
    dA shape: same as Z
    Z shape: pre-activation values from forward pass

    Returns:
    dZ shape: same as Z
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, A):
    """
    dA shape: same as A
    A shape: sigmoid output from forward pass

    Returns:
    dZ shape: same as A
    """
    dZ = dA * A * (1 - A)

    return dZ


def dense_backward(dZ, cache):
    """
    dZ shape: (m, n_new)

    cache contains:
    A_prev shape: (m, n_prev)
    W shape:      (n_prev, n_new)
    b shape:      (1, n_new)

    Returns:
    dA_prev shape: (m, n_prev)
    dW shape:      (n_prev, n_new)
    db shape:      (1, n_new)
    """

    A_prev, W, b = cache
    m = A_prev.shape[0]

    dW = (1 / m) * np.dot(A_prev.T, dZ)
    db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db


def conv_backward(dZ, cache):
    """
    dZ shape: (m, h_new, w_new, c_new)

    cache contains:
    X shape: (m, h_prev, w_prev, c_prev)
    W shape: (f, f, c_prev, c_new)
    b shape: (1, 1, 1, c_new)
    stride: int
    pad: int

    Returns:
    dX shape: same as X
    dW shape: same as W
    db shape: same as b
    """

    X, W, b, stride, pad = cache

    m, h_prev, w_prev, c_prev = X.shape
    f, _, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape

    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    X_pad = zero_pad(X, pad)

    dX_pad = zero_pad(dX, pad)

    for i in range(m):
        x_pad = X_pad[i]
        dx_pad = dX_pad[i]

        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_new):
                    x_slice = x_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]

                    dx_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += x_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if pad == 0:
            dX[i, :, :, :] = dx_pad
        else:
            dX[i, :, :, :] = dx_pad[pad:-pad, pad:-pad, :]

    return dX, dW, db


def max_pool_backward(dA, cache):
    """
    dA shape: (m, h_new, w_new, c_prev)

    cache contains:
    X shape: (m, h_prev, w_prev, c_prev)
    pool_size: int
    stride: int

    Returns:
    dX shape: same as X
    """

    X, pool_size, stride = cache

    m, h_prev, w_prev, c_prev = X.shape
    _, h_new, w_new, _ = dA.shape

    dX = np.zeros_like(X)
    f = pool_size

    for i in range(m):
        for h in range(h_new):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(w_new):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(c_prev):
                    x_slice = X[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]

                    mask = x_slice == np.max(x_slice)

                    dX[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ] += mask * dA[i, h, w, c]

    return dX