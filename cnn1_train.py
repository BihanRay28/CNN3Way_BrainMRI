import numpy as np

from cnn1_core import (
    initialize_cnn1_parameters,
    cnn1_forward,
    cnn1_backward,
    compute_cost,
    initialize_adam,
    adam_update,
    predict_cnn1
)


# ============================================================
# Mini-Batch Creation
# ============================================================

def create_mini_batches(X, Y, mini_batch_size=32, seed=42):
    np.random.seed(seed)

    m = X.shape[0]
    permutation = np.random.permutation(m)

    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    mini_batches = []
    num_complete_batches = m // mini_batch_size

    for k in range(num_complete_batches):
        mini_batch_X = shuffled_X[
            k * mini_batch_size:(k + 1) * mini_batch_size
        ]

        mini_batch_Y = shuffled_Y[
            k * mini_batch_size:(k + 1) * mini_batch_size
        ]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
            num_complete_batches * mini_batch_size:
        ]

        mini_batch_Y = shuffled_Y[
            num_complete_batches * mini_batch_size:
        ]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


# ============================================================
# Learning Rate Decay
# ============================================================

def learning_rate_decay(initial_lr, epoch, decay_rate=0.02):
    """
    Inverse time decay:

    lr = initial_lr / (1 + decay_rate * epoch)
    """

    return initial_lr / (1 + decay_rate * epoch)


# ============================================================
# Accuracy
# ============================================================

def accuracy(X, Y, parameters):
    predictions, probabilities = predict_cnn1(X, parameters)
    return np.mean(predictions == Y)


# ============================================================
# Training CNN1
# ============================================================

def train_cnn1(
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    input_shape=(64, 64, 1),
    epochs=20,
    mini_batch_size=16,
    initial_learning_rate=0.001,
    decay_rate=0.02,
    conv_keep_prob=0.85,
    dense_keep_prob=0.75,
    lambd=0.005,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    seed=42,
    print_every=1
):
    parameters = initialize_cnn1_parameters(
        input_shape=input_shape,
        seed=seed
    )

    v, s = initialize_adam(parameters)

    history = {
        "train_cost": [],
        "val_cost": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    t = 0

    for epoch in range(epochs):
        epoch_cost = 0

        current_lr = learning_rate_decay(
            initial_learning_rate,
            epoch,
            decay_rate
        )

        mini_batches = create_mini_batches(
            X_train,
            Y_train,
            mini_batch_size=mini_batch_size,
            seed=seed + epoch
        )

        for mini_batch_X, mini_batch_Y in mini_batches:
            # Dropout ON during training
            AL, cache = cnn1_forward(
                mini_batch_X,
                parameters,
                conv_keep_prob=conv_keep_prob,
                dense_keep_prob=dense_keep_prob,
                training=True
            )

            cost = compute_cost(
                AL,
                mini_batch_Y,
                parameters=parameters,
                lambd=lambd
            )

            grads = cnn1_backward(
                AL,
                mini_batch_Y,
                cache,
                parameters,
                lambd=lambd
            )

            t += 1

            parameters, v, s = adam_update(
                parameters,
                grads,
                v,
                s,
                t=t,
                learning_rate=current_lr,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon
            )

            epoch_cost += cost / len(mini_batches)

        # Dropout OFF during full train evaluation
        train_AL, _ = cnn1_forward(
            X_train,
            parameters,
            conv_keep_prob=1.0,
            dense_keep_prob=1.0,
            training=False
        )

        train_cost = compute_cost(
            train_AL,
            Y_train,
            parameters=parameters,
            lambd=lambd
        )

        train_acc = accuracy(X_train, Y_train, parameters)

        history["train_cost"].append(train_cost)
        history["train_accuracy"].append(train_acc)
        history["learning_rate"].append(current_lr)

        if X_val is not None and Y_val is not None:
            # Dropout OFF during validation
            val_AL, _ = cnn1_forward(
                X_val,
                parameters,
                conv_keep_prob=1.0,
                dense_keep_prob=1.0,
                training=False
            )

            val_cost = compute_cost(
                val_AL,
                Y_val,
                parameters=parameters,
                lambd=lambd
            )

            val_acc = accuracy(X_val, Y_val, parameters)

            history["val_cost"].append(val_cost)
            history["val_accuracy"].append(val_acc)

        if print_every is not None and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Learning rate: {current_lr:.8f}")
            print(f"Mini-batch cost: {epoch_cost:.5f}")
            print(f"Train cost: {train_cost:.5f}")
            print(f"Train accuracy: {train_acc:.4f}")

            if X_val is not None and Y_val is not None:
                print(f"Val cost: {val_cost:.5f}")
                print(f"Val accuracy: {val_acc:.4f}")

            print("-" * 50)

    return parameters, history


# ============================================================
# Testing
# ============================================================

def evaluate_cnn1(X_test, Y_test, parameters):
    # Dropout OFF during testing
    AL, _ = cnn1_forward(
        X_test,
        parameters,
        conv_keep_prob=1.0,
        dense_keep_prob=1.0,
        training=False
    )

    test_cost = compute_cost(
        AL,
        Y_test,
        parameters=parameters,
        lambd=0.0
    )

    test_predictions = (AL >= 0.5).astype(int)
    test_accuracy = np.mean(test_predictions == Y_test)

    return test_cost, test_accuracy, test_predictions, AL


# ============================================================
# Temporary Dummy Test
# ============================================================

if __name__ == "__main__":
    X_train = np.random.randn(12, 64, 64, 1)
    Y_train = np.array(
        [[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]]
    )

    X_val = np.random.randn(4, 64, 64, 1)
    Y_val = np.array([[1], [0], [1], [0]])

    parameters, history = train_cnn1(
        X_train,
        Y_train,
        X_val=X_val,
        Y_val=Y_val,
        input_shape=(64, 64, 1),
        epochs=2,
        mini_batch_size=4,
        initial_learning_rate=0.001,
        decay_rate=0.02,
        conv_keep_prob=0.85,
        dense_keep_prob=0.75,
        lambd=0.005,
        print_every=1
    )

    test_cost, test_acc, preds, probs = evaluate_cnn1(
        X_val,
        Y_val,
        parameters
    )

    print("Final test cost:", test_cost)
    print("Final test accuracy:", test_acc)
