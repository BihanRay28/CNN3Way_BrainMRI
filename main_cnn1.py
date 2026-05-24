import os
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
    from PIL import Image, ImageDraw

from data import load_brain_mri_kfold_data
from cnn1_core import (
    initialize_cnn1_parameters,
    cnn1_forward,
    cnn1_backward,
    compute_cost,
    initialize_adam,
    adam_update
)
from cnn1_train import (
    create_mini_batches,
    learning_rate_decay
)


# ============================================================
# Folder Setup
# ============================================================

def make_output_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/cnn1", exist_ok=True)
    os.makedirs("results/cnn1/plots", exist_ok=True)
    os.makedirs("results/cnn1/metrics", exist_ok=True)
    os.makedirs("Output", exist_ok=True)


# ============================================================
# Progress Bar
# ============================================================

def print_progress_bar(
    epoch,
    epochs,
    batch_index,
    total_batches,
    batch_cost,
    learning_rate,
    bar_length=30
):
    progress = batch_index / total_batches
    filled = int(progress * bar_length)

    bar = "#" * filled + "." * (bar_length - filled)

    print(
        f"\rEpoch {epoch}/{epochs} [{bar}] "
        f"{batch_index}/{total_batches} "
        f"| batch cost: {batch_cost:.5f} "
        f"| lr: {learning_rate:.8f}",
        end="",
        flush=True
    )

    if batch_index == total_batches:
        print()


# ============================================================
# Metrics
# ============================================================

def confusion_matrix_binary(Y_true, Y_pred):
    Y_true = Y_true.reshape(-1)
    Y_pred = Y_pred.reshape(-1)

    TP = int(np.sum((Y_true == 1) & (Y_pred == 1)))
    TN = int(np.sum((Y_true == 0) & (Y_pred == 0)))
    FP = int(np.sum((Y_true == 0) & (Y_pred == 1)))
    FN = int(np.sum((Y_true == 1) & (Y_pred == 0)))

    matrix = np.array([
        [TN, FP],
        [FN, TP]
    ])

    return matrix, TP, TN, FP, FN


def classification_metrics(Y_true, Y_pred):
    matrix, TP, TN, FP, FN = confusion_matrix_binary(Y_true, Y_pred)

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / max(total, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "confusion_matrix": matrix
    }


def evaluate_model(X, Y, parameters):
    AL, _ = cnn1_forward(
        X,
        parameters,
        conv_keep_prob=1.0,
        dense_keep_prob=1.0,
        training=False
    )

    cost = compute_cost(
        AL,
        Y,
        parameters=None,
        lambd=0.0
    )

    predictions = (AL >= 0.5).astype(int)
    metrics = classification_metrics(Y, predictions)

    return cost, metrics, predictions, AL


def print_metrics(metrics, title):
    print()
    print(title)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1 score:", round(metrics["f1"], 4))
    print("TP:", metrics["TP"])
    print("TN:", metrics["TN"])
    print("FP:", metrics["FP"])
    print("FN:", metrics["FN"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


# ============================================================
# Plotting
# ============================================================

def copy_parameters(parameters):
    return {
        key: value.copy()
        for key, value in parameters.items()
    }


def is_better_validation_result(current_key, best_key, min_delta=0.001):
    if best_key is None:
        return True

    current_f1, current_accuracy, current_negative_cost = current_key
    best_f1, best_accuracy, best_negative_cost = best_key

    if current_f1 > best_f1 + min_delta:
        return True

    if abs(current_f1 - best_f1) <= min_delta:
        if current_accuracy > best_accuracy + min_delta:
            return True

        if (
            abs(current_accuracy - best_accuracy) <= min_delta
            and current_negative_cost > best_negative_cost + min_delta
        ):
            return True

    return False


def pad_metric_histories(all_histories, metric_name):
    max_length = max(len(history[metric_name]) for history in all_histories)
    padded_histories = []

    for history in all_histories:
        values = list(history[metric_name])

        if len(values) < max_length:
            values.extend([values[-1]] * (max_length - len(values)))

        padded_histories.append(values)

    return np.array(padded_histories)


def save_fallback_line_plot(series_list, labels, title, x_label, y_label, save_paths):
    if isinstance(save_paths, str):
        save_paths = [save_paths]

    width, height = 900, 600
    margin_left = 85
    margin_right = 35
    margin_top = 55
    margin_bottom = 75

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    values = np.array([value for series in series_list for value in series])
    y_min = float(np.min(values))
    y_max = float(np.max(values))

    if abs(y_max - y_min) < 1e-8:
        y_max += 1
        y_min -= 1

    x_count = max(len(series) for series in series_list)
    x_max = max(x_count - 1, 1)

    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline="black",
        width=2
    )

    for step in range(6):
        y_value = y_min + ((y_max - y_min) * step / 5)
        y = plot_bottom - int((plot_bottom - plot_top) * step / 5)
        draw.line([plot_left, y, plot_right, y], fill=(220, 220, 220))
        draw.text((10, y - 8), f"{y_value:.3f}", fill="black")

    colors = [(30, 90, 180), (210, 80, 40), (40, 150, 90), (140, 70, 170)]

    for series_index, series in enumerate(series_list):
        points = []

        for index, value in enumerate(series):
            x = plot_left + int((plot_right - plot_left) * index / x_max)
            y_ratio = (float(value) - y_min) / (y_max - y_min)
            y = plot_bottom - int((plot_bottom - plot_top) * y_ratio)
            points.append((x, y))

        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colors[series_index])
        else:
            draw.line(points, fill=colors[series_index], width=3)

        legend_x = plot_left + 15
        legend_y = plot_top + 15 + (series_index * 22)
        draw.line(
            [legend_x, legend_y + 7, legend_x + 30, legend_y + 7],
            fill=colors[series_index],
            width=3
        )
        draw.text((legend_x + 40, legend_y), labels[series_index], fill="black")

    draw.text((plot_left, 18), title, fill="black")
    draw.text(((width // 2) - 45, height - 35), x_label, fill="black")
    draw.text((10, 18), y_label, fill="black")

    for save_path in save_paths:
        image.save(save_path)


def save_fallback_confusion_matrix(matrix, title, save_paths):
    if isinstance(save_paths, str):
        save_paths = [save_paths]

    width, height = 650, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    matrix = np.array(matrix)
    max_value = max(float(np.max(matrix)), 1.0)

    cell_size = 155
    start_x = 180
    start_y = 130

    draw.text((start_x, 35), title, fill="black")
    draw.text((start_x + 65, height - 45), "Predicted label", fill="black")
    draw.text((25, start_y + 125), "True label", fill="black")

    x_labels = ["No tumor", "Tumor"]
    y_labels = ["No tumor", "Tumor"]

    for index, label in enumerate(x_labels):
        draw.text(
            (start_x + (index * cell_size) + 45, start_y - 30),
            label,
            fill="black"
        )

    for index, label in enumerate(y_labels):
        draw.text(
            (start_x - 95, start_y + (index * cell_size) + 65),
            label,
            fill="black"
        )

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            intensity = int(255 - (180 * (value / max_value)))
            fill = (intensity, intensity + 20, 255)

            x0 = start_x + (col * cell_size)
            y0 = start_y + (row * cell_size)
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], fill=fill, outline="black", width=2)
            draw.text((x0 + 70, y0 + 70), str(value), fill="black")

    for save_path in save_paths:
        image.save(save_path)


def save_matplotlib_figure(save_paths):
    if isinstance(save_paths, str):
        save_paths = [save_paths]

    for save_path in save_paths:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_cost_curve(history, fold_number):
    epochs = np.arange(1, len(history["train_cost"]) + 1)
    save_path = f"results/cnn1/plots/cnn1_fold_{fold_number}_cost_curve.png"

    if plt is None:
        save_fallback_line_plot(
            [history["train_cost"], history["val_cost"]],
            ["Train cost", "Validation cost"],
            f"CNN1 Cost Curve - Fold {fold_number}",
            "Epoch",
            "Cost",
            save_path
        )
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_cost"], label="Train cost")
    plt.plot(epochs, history["val_cost"], label="Validation cost")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title(f"CNN1 Cost Curve - Fold {fold_number}")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy_curve(history, fold_number):
    epochs = np.arange(1, len(history["train_accuracy"]) + 1)
    save_path = f"results/cnn1/plots/cnn1_fold_{fold_number}_accuracy_curve.png"

    if plt is None:
        save_fallback_line_plot(
            [history["train_accuracy"], history["val_accuracy"]],
            ["Train accuracy", "Validation accuracy"],
            f"CNN1 Accuracy Curve - Fold {fold_number}",
            "Epoch",
            "Accuracy",
            save_path
        )
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"CNN1 Accuracy Curve - Fold {fold_number}")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(matrix, fold_number):
    save_path = f"results/cnn1/plots/cnn1_fold_{fold_number}_confusion_matrix.png"

    if plt is None:
        save_fallback_confusion_matrix(
            matrix,
            f"CNN1 Confusion Matrix - Fold {fold_number}",
            save_path
        )
        return

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)

    plt.title(f"CNN1 Confusion Matrix - Fold {fold_number}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks([0, 1], ["No tumor", "Tumor"])
    plt.yticks([0, 1], ["No tumor", "Tumor"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.colorbar()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_total_cost_curve(all_histories):
    train_costs = pad_metric_histories(all_histories, "train_cost")
    val_costs = pad_metric_histories(all_histories, "val_cost")
    epochs = np.arange(1, train_costs.shape[1] + 1)
    save_path = "results/cnn1/plots/cnn1_total_cost_curve.png"
    output_path = "Output/CNN1Cost.jpg"

    if plt is None:
        save_fallback_line_plot(
            [np.mean(train_costs, axis=0), np.mean(val_costs, axis=0)],
            ["Mean train cost", "Mean validation cost"],
            "CNN1 Total Cost Curve Across Folds",
            "Epoch",
            "Cost",
            [save_path, output_path]
        )
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, np.mean(train_costs, axis=0), label="Mean train cost")
    plt.plot(epochs, np.mean(val_costs, axis=0), label="Mean validation cost")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("CNN1 Total Cost Curve Across Folds")
    plt.legend()
    plt.grid(True)

    save_matplotlib_figure([save_path, output_path])
    plt.show()


def plot_total_accuracy_curve(all_histories):
    train_accuracies = pad_metric_histories(all_histories, "train_accuracy")
    val_accuracies = pad_metric_histories(all_histories, "val_accuracy")
    epochs = np.arange(1, train_accuracies.shape[1] + 1)
    save_path = "results/cnn1/plots/cnn1_total_accuracy_curve.png"
    output_path = "Output/CNN1Accuracy.jpg"

    if plt is None:
        save_fallback_line_plot(
            [
                np.mean(train_accuracies, axis=0),
                np.mean(val_accuracies, axis=0)
            ],
            ["Mean train accuracy", "Mean validation accuracy"],
            "CNN1 Total Accuracy Curve Across Folds",
            "Epoch",
            "Accuracy",
            [save_path, output_path]
        )
        return

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        np.mean(train_accuracies, axis=0),
        label="Mean train accuracy"
    )
    plt.plot(
        epochs,
        np.mean(val_accuracies, axis=0),
        label="Mean validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN1 Total Accuracy Curve Across Folds")
    plt.legend()
    plt.grid(True)

    save_matplotlib_figure([save_path, output_path])
    plt.show()


def plot_total_confusion_matrix(matrix):
    save_path = "results/cnn1/plots/cnn1_total_confusion_matrix.png"
    output_path = "Output/CNN1ConfMatrix.jpg"

    if plt is None:
        save_fallback_confusion_matrix(
            matrix,
            "CNN1 Total Confusion Matrix Across Folds",
            [save_path, output_path]
        )
        return

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)

    plt.title("CNN1 Total Confusion Matrix Across Folds")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks([0, 1], ["No tumor", "Tumor"])
    plt.yticks([0, 1], ["No tumor", "Tumor"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.colorbar()

    save_matplotlib_figure([save_path, output_path])
    plt.show()


# ============================================================
# One Fold Training
# ============================================================

def train_one_fold(
    X_train,
    Y_train,
    X_val,
    Y_val,
    fold_number,
    input_shape=(64, 64, 1),
    epochs=20,
    mini_batch_size=8,
    initial_learning_rate=0.001,
    decay_rate=0.02,
    conv_keep_prob=0.85,
    dense_keep_prob=0.75,
    lambd=0.005,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    early_stopping=True,
    min_epochs=5,
    patience=3,
    min_delta=0.001,
    overfit_gap=0.08,
    overfit_patience=2,
    seed=42
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
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rate": [],
        "stopped_epoch": None,
        "best_epoch": None,
        "stop_reason": None
    }

    t = 0
    best_parameters = None
    best_key = None
    best_val_cost = None
    no_improvement_count = 0
    overfit_count = 0

    for epoch in range(1, epochs + 1):
        current_lr = learning_rate_decay(
            initial_learning_rate,
            epoch - 1,
            decay_rate
        )

        mini_batches = create_mini_batches(
            X_train,
            Y_train,
            mini_batch_size=mini_batch_size,
            seed=seed + epoch
        )

        total_batches = len(mini_batches)

        for batch_index, (mini_batch_X, mini_batch_Y) in enumerate(mini_batches, start=1):
            AL, cache = cnn1_forward(
                mini_batch_X,
                parameters,
                conv_keep_prob=conv_keep_prob,
                dense_keep_prob=dense_keep_prob,
                training=True
            )

            batch_cost = compute_cost(
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

            print_progress_bar(
                epoch=epoch,
                epochs=epochs,
                batch_index=batch_index,
                total_batches=total_batches,
                batch_cost=batch_cost,
                learning_rate=current_lr
            )

        train_cost, train_metrics, _, _ = evaluate_model(
            X_train,
            Y_train,
            parameters
        )

        val_cost, val_metrics, _, _ = evaluate_model(
            X_val,
            Y_val,
            parameters
        )

        history["train_cost"].append(train_cost)
        history["val_cost"].append(val_cost)

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])

        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        history["learning_rate"].append(current_lr)

        print(f"Fold {fold_number} | Epoch {epoch}/{epochs} summary")
        print(f"Train cost: {train_cost:.5f} | Val cost: {val_cost:.5f}")
        print(f"Train acc: {train_metrics['accuracy']:.4f} | Val acc: {val_metrics['accuracy']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print("-" * 70)

        current_key = (
            val_metrics["f1"],
            val_metrics["accuracy"],
            -val_cost
        )

        improved = is_better_validation_result(
            current_key=current_key,
            best_key=best_key,
            min_delta=min_delta
        )

        if improved:
            best_key = current_key
            best_val_cost = val_cost
            best_parameters = copy_parameters(parameters)
            history["best_epoch"] = epoch
            no_improvement_count = 0
            overfit_count = 0
        elif early_stopping and epoch >= min_epochs:
            no_improvement_count += 1

            if (
                best_val_cost is not None
                and train_metrics["f1"] - val_metrics["f1"] >= overfit_gap
                and val_cost > best_val_cost + min_delta
            ):
                overfit_count += 1
            else:
                overfit_count = 0

        if early_stopping and epoch >= min_epochs:
            if overfit_count >= overfit_patience:
                history["stopped_epoch"] = epoch
                history["stop_reason"] = (
                    "overfit guard: train F1 is too far above validation F1 "
                    "while validation cost is worse than the best epoch"
                )
                break

            if no_improvement_count >= patience:
                history["stopped_epoch"] = epoch
                history["stop_reason"] = (
                    "early stopping: validation F1/accuracy/cost did not "
                    "improve within patience"
                )
                break

    if best_parameters is not None:
        parameters = best_parameters

    if history["stopped_epoch"] is None:
        history["stopped_epoch"] = len(history["train_cost"])
        history["stop_reason"] = "max epochs reached"

    print()
    print(f"Fold {fold_number} training stopped at epoch {history['stopped_epoch']}/{epochs}")
    print(f"Best epoch restored: {history['best_epoch']}")
    print("Stop reason:", history["stop_reason"])
    print("-" * 70)

    final_val_cost, final_val_metrics, predictions, probabilities = evaluate_model(
        X_val,
        Y_val,
        parameters
    )

    return parameters, history, final_val_metrics, predictions, probabilities


# ============================================================
# Main K-Fold Experiment
# ============================================================

def run_cnn1_experiment(
    data_dir="data",
    image_size=(64, 64),
    grayscale=True,
    k=5,
    epochs=20,
    mini_batch_size=8,
    initial_learning_rate=0.001,
    decay_rate=0.02,
    conv_keep_prob=0.85,
    dense_keep_prob=0.75,
    lambd=0.005,
    early_stopping=True,
    min_epochs=5,
    patience=3,
    min_delta=0.001,
    overfit_gap=0.08,
    overfit_patience=2,
    seed=42
):
    make_output_dirs()

    X, Y, folds = load_brain_mri_kfold_data(
        data_dir=data_dir,
        image_size=image_size,
        grayscale=grayscale,
        k=k,
        seed=seed
    )

    print()
    print("=" * 70)
    print("CNN1 Baseline Experiment Started")
    print("=" * 70)
    print("Full X shape:", X.shape)
    print("Full Y shape:", Y.shape)
    print("Number of folds:", len(folds))
    print("Epochs:", epochs)
    print("Mini-batch size:", mini_batch_size)
    print("Initial learning rate:", initial_learning_rate)
    print("Decay rate:", decay_rate)
    print("Conv keep prob:", conv_keep_prob)
    print("Dense keep prob:", dense_keep_prob)
    print("L2 lambda:", lambd)
    print("Early stopping:", early_stopping)
    print("Minimum epochs:", min_epochs)
    print("Patience:", patience)
    print("Minimum improvement delta:", min_delta)
    print("Overfit F1 gap:", overfit_gap)
    print("Overfit patience:", overfit_patience)
    print("=" * 70)

    experiment_config = {
        "data_dir": data_dir,
        "image_size": image_size,
        "grayscale": grayscale,
        "k": k,
        "epochs": epochs,
        "mini_batch_size": mini_batch_size,
        "initial_learning_rate": initial_learning_rate,
        "decay_rate": decay_rate,
        "conv_keep_prob": conv_keep_prob,
        "dense_keep_prob": dense_keep_prob,
        "lambd": lambd,
        "early_stopping": early_stopping,
        "min_epochs": min_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "overfit_gap": overfit_gap,
        "overfit_patience": overfit_patience,
        "seed": seed
    }

    all_histories = []
    all_metrics = []
    all_parameters = []

    input_shape = (
        image_size[0],
        image_size[1],
        1 if grayscale else 3
    )

    for fold_index, fold in enumerate(folds, start=1):
        X_train, Y_train, X_val, Y_val = fold

        print()
        print("=" * 70)
        print(f"Fold {fold_index}/{k}")
        print("X_train:", X_train.shape)
        print("Y_train:", Y_train.shape)
        print("X_val:", X_val.shape)
        print("Y_val:", Y_val.shape)
        print("=" * 70)

        parameters, history, final_metrics, predictions, probabilities = train_one_fold(
            X_train,
            Y_train,
            X_val,
            Y_val,
            fold_number=fold_index,
            input_shape=input_shape,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            conv_keep_prob=conv_keep_prob,
            dense_keep_prob=dense_keep_prob,
            lambd=lambd,
            early_stopping=early_stopping,
            min_epochs=min_epochs,
            patience=patience,
            min_delta=min_delta,
            overfit_gap=overfit_gap,
            overfit_patience=overfit_patience,
            seed=seed + fold_index
        )

        print_metrics(
            final_metrics,
            title=f"Fold {fold_index} Final Validation Metrics"
        )

        plot_cost_curve(history, fold_index)
        plot_accuracy_curve(history, fold_index)
        plot_confusion_matrix(final_metrics["confusion_matrix"], fold_index)

        all_histories.append(history)
        all_metrics.append(final_metrics)
        all_parameters.append(parameters)

    summary = summarize_kfold_metrics(all_metrics)
    total_metrics = summarize_total_metrics(all_metrics)
    best_fold_index = select_best_fold_index(all_metrics)
    updated_parameters_path = save_updated_parameters(
        parameters=all_parameters[best_fold_index],
        fold_number=best_fold_index + 1,
        fold_metrics=all_metrics[best_fold_index],
        experiment_config=experiment_config
    )

    save_experiment_summary(
        summary,
        all_metrics,
        total_metrics,
        experiment_config
    )

    plot_total_cost_curve(all_histories)
    plot_total_accuracy_curve(all_histories)
    plot_total_confusion_matrix(total_metrics["confusion_matrix"])

    print()
    print("=" * 70)
    print("CNN1 K-Fold Final Summary")
    print("=" * 70)
    print("Mean accuracy:", round(summary["mean_accuracy"], 4))
    print("Mean precision:", round(summary["mean_precision"], 4))
    print("Mean recall:", round(summary["mean_recall"], 4))
    print("Mean F1:", round(summary["mean_f1"], 4))
    print()
    print("Total validation accuracy:", round(total_metrics["accuracy"], 4))
    print("Total validation precision:", round(total_metrics["precision"], 4))
    print("Total validation recall:", round(total_metrics["recall"], 4))
    print("Total validation F1:", round(total_metrics["f1"], 4))
    print("Total confusion matrix:")
    print(total_metrics["confusion_matrix"])
    print()
    print("Best fold saved:", best_fold_index + 1)
    print("Updated parameters JSON:", updated_parameters_path)
    print("=" * 70)

    return all_parameters, all_histories, all_metrics, summary, total_metrics


# ============================================================
# Summary Saving
# ============================================================

def summarize_kfold_metrics(all_metrics):
    accuracies = [m["accuracy"] for m in all_metrics]
    precisions = [m["precision"] for m in all_metrics]
    recalls = [m["recall"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),

        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),

        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),

        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s))
    }


def summarize_total_metrics(all_metrics):
    total_matrix = np.sum(
        [metric["confusion_matrix"] for metric in all_metrics],
        axis=0
    )

    TN = int(total_matrix[0, 0])
    FP = int(total_matrix[0, 1])
    FN = int(total_matrix[1, 0])
    TP = int(total_matrix[1, 1])

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / max(total, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "confusion_matrix": total_matrix
    }


def select_best_fold_index(all_metrics):
    return max(
        range(len(all_metrics)),
        key=lambda index: (
            all_metrics[index]["f1"],
            all_metrics[index]["accuracy"],
            all_metrics[index]["recall"]
        )
    )


def serialize_parameters(parameters):
    return {
        key: value.tolist()
        for key, value in parameters.items()
    }


def serialize_metric(metric):
    metric_copy = metric.copy()

    if "confusion_matrix" in metric_copy:
        metric_copy["confusion_matrix"] = (
            metric_copy["confusion_matrix"].tolist()
        )

    return metric_copy


def save_updated_parameters(
    parameters,
    fold_number,
    fold_metrics,
    experiment_config=None,
    save_path="results/cnn1/metrics/updated_parameters_1.json"
):
    output = {
        "model": "cnn1",
        "architecture": "Conv-ReLU-MaxPool-Conv-ReLU-Dropout-MaxPool-Flatten-Dense64-ReLU-Dropout-Dense1-Sigmoid",
        "selected_fold": fold_number,
        "selection_rule": "highest validation F1, then accuracy, then recall",
        "config": experiment_config,
        "validation_metrics": serialize_metric(fold_metrics),
        "parameters": serialize_parameters(parameters)
    }

    with open(save_path, "w") as file:
        json.dump(output, file)

    return save_path


def save_experiment_summary(
    summary,
    all_metrics,
    total_metrics,
    experiment_config=None
):
    serializable_metrics = []

    for metric in all_metrics:
        serializable_metrics.append(serialize_metric(metric))

    total_metrics_copy = serialize_metric(total_metrics)

    output = {
        "config": experiment_config,
        "summary": summary,
        "fold_metrics": serializable_metrics,
        "total_metrics": total_metrics_copy
    }

    save_path = "results/cnn1/metrics/cnn1_kfold_summary.json"

    with open(save_path, "w") as file:
        json.dump(output, file, indent=4)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_cnn1_experiment(
        data_dir="data",
        image_size=(64, 64),
        grayscale=True,
        k=5,
        epochs=20,
        mini_batch_size=8,
        initial_learning_rate=0.001,
        decay_rate=0.02,
        conv_keep_prob=0.85,
        dense_keep_prob=0.75,
        lambd=0.005,
        early_stopping=True,
        min_epochs=5,
        patience=3,
        min_delta=0.001,
        overfit_gap=0.08,
        overfit_patience=2,
        seed=42
    )
