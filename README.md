# CNN3Way Brain MRI Classification

This repository contains a from-scratch NumPy implementation of a convolutional neural network for binary brain MRI tumor classification.

The long-term goal of the project is to compare three CNN-style architectures on the same or similar MRI dataset:

1. CNN Base
2. InceptionNet
3. ResNet

At the current stage, the CNN Base architecture is implemented and evaluated using stratified 5-fold cross-validation.

## Current Status

Implemented:

- Brain MRI image loading and preprocessing
- Binary tumor/no-tumor classification
- CNN Base implemented from scratch using NumPy
- Manual convolution, pooling, dense layers, activations, and backpropagation
- Adam optimizer from scratch
- Dropout and L2 regularization
- Stratified k-fold validation
- Dynamic early stopping per fold
- Per-fold and total metric reporting
- Accuracy, cost, and confusion matrix plots
- Saved best-fold parameters for future prediction

Planned:

- InceptionNet-style model from scratch
- ResNet-style model from scratch
- Comparison of all three architectures using the same training/evaluation pipeline

## Dataset Layout

The dataset is not committed to this repository. Place the dataset locally using this structure:

```text
data/
  yes/
    image files for tumor cases
  no/
    image files for no-tumor cases
```

Labels:

```text
yes -> 1
no  -> 0
```

Supported image extensions:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

During the last recorded run, the dataset contained:

```text
Tumor images:    155
No-tumor images: 98
Total images:    253
```

Each image is resized to `64 x 64`, converted to grayscale, normalized to `[0, 1]`, and stored with shape:

```text
(64, 64, 1)
```

The final loaded dataset shape is:

```text
X: (253, 64, 64, 1)
Y: (253, 1)
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

```text
numpy
Pillow
matplotlib
```

If `matplotlib` is not available, the training script has a Pillow-based fallback for saving basic plots.

## Running Training

Run the main CNN Base experiment:

```bash
python main_cnn1.py
```

On Windows, if `python` is not recognized, use the Python executable installed on your system.

The script prints live training progress:

```text
Epoch 1/20 [##############################] 26/26 | batch cost: ... | lr: ...
Fold 1 | Epoch 1/20 summary
Train cost: ... | Val cost: ...
Train acc: ... | Val acc: ...
Train F1: ... | Val F1: ...
```

## CNN Base Architecture

The implemented CNN Base architecture is:

```text
Input: 64 x 64 x 1

Conv1 -> ReLU -> MaxPool
Conv2 -> ReLU -> Dropout -> MaxPool
Flatten
Dense 64 -> ReLU -> Dropout
Dense 1 -> Sigmoid
```

Detailed shape flow:

```text
Input image:
64 x 64 x 1

Conv1:
3 x 3 filters, 8 output channels, stride 1, padding 1
Output: 64 x 64 x 8

ReLU:
Keeps positive activations and sets negative activations to zero

MaxPool:
2 x 2 pool, stride 2
Output: 32 x 32 x 8

Conv2:
3 x 3 filters, 16 output channels, stride 1, padding 1
Output: 32 x 32 x 16

ReLU + Dropout:
Conv dropout keep probability = 0.85

MaxPool:
2 x 2 pool, stride 2
Output: 16 x 16 x 16

Flatten:
16 x 16 x 16 = 4096 features

Dense hidden layer:
4096 -> 64

ReLU + Dropout:
Dense dropout keep probability = 0.75

Output layer:
64 -> 1

Sigmoid:
Returns probability of tumor
```

Prediction rule:

```text
probability >= 0.5 -> tumor
probability < 0.5  -> no tumor
```

## Why the Dense Layer Was Updated

The first CNN Base version connected the flattened feature vector directly to a single sigmoid output:

```text
4096 -> 1
```

That made the classifier head too broad and contributed to overfitting. The updated model uses an intermediate hidden layer:

```text
4096 -> 64 -> 1
```

This gives the model a smaller learned representation before the final sigmoid output. A second dropout layer was also added after the 64-node hidden layer to reduce overfitting.

Current dense parameter shapes:

```text
W3: 4096 x 64
b3: 1 x 64
W4: 64 x 1
b4: 1 x 1
```

## Training Configuration

The current CNN Base training configuration is:

```text
image_size = (64, 64)
grayscale = True
k = 5
max epochs = 20
mini_batch_size = 8
initial_learning_rate = 0.001
decay_rate = 0.02
conv_keep_prob = 0.85
dense_keep_prob = 0.75
L2 lambda = 0.005
optimizer = Adam
beta1 = 0.9
beta2 = 0.999
seed = 42
```

Learning rate decay:

```text
lr = initial_lr / (1 + decay_rate * epoch)
```

## Dynamic Early Stopping

Each fold can run up to 20 epochs, but it can stop earlier if validation performance stops improving or overfitting becomes too strong.

Current early stopping settings:

```text
early_stopping = True
min_epochs = 5
patience = 3
min_delta = 0.001
overfit_gap = 0.08
overfit_patience = 2
```

The model tracks validation F1, validation accuracy, and validation cost. It restores the best epoch's parameters before computing final fold metrics.

Overfitting guard:

```text
train F1 - validation F1 >= 0.08
```

and validation cost is worse than the best epoch for 2 consecutive epochs.

This prevents the model from continuing training when training performance keeps improving but validation performance begins to degrade.

## Results From Latest Recorded CNN Base Run

5-fold summary:

```text
Mean accuracy:  0.8225
Mean precision: 0.8189
Mean recall:    0.9161
Mean F1:        0.8632
```

Total validation metrics:

```text
Accuracy:  0.8221
Precision: 0.8161
Recall:    0.9161
F1:        0.8632
```

Total confusion matrix:

```text
[[ 66  32]
 [ 13 142]]
```

Matrix format:

```text
[[TN FP]
 [FN TP]]
```

Interpretation:

- True negatives: 66
- False positives: 32
- False negatives: 13
- True positives: 142

The model has high recall, meaning it catches most tumor cases. The false-positive count is still relatively high, so the model tends to predict tumor more often than no-tumor in uncertain cases.

Best fold:

```text
Fold 5
Accuracy:  0.9200
Precision: 0.9091
Recall:    0.9677
F1:        0.9375
```

The best fold's parameters are saved after training.

## Output Files

Metrics:

```text
results/cnn1/metrics/cnn1_kfold_summary.json
```

This file stores:

- Training configuration
- Mean k-fold metrics
- Per-fold metrics
- Total validation metrics
- Total confusion matrix

Saved best-fold parameters:

```text
results/cnn1/metrics/updated_parameters_1.json
```

This file is ignored by Git because it can become large. It is generated locally after training.

Per-fold plots:

```text
results/cnn1/plots/cnn1_fold_1_accuracy_curve.png
results/cnn1/plots/cnn1_fold_1_cost_curve.png
results/cnn1/plots/cnn1_fold_1_confusion_matrix.png
...
```

Total plots:

```text
results/cnn1/plots/cnn1_total_accuracy_curve.png
results/cnn1/plots/cnn1_total_cost_curve.png
results/cnn1/plots/cnn1_total_confusion_matrix.png
```

Simplified report outputs:

```text
Output/CNN1Accuracy.jpg
Output/CNN1Cost.jpg
Output/CNN1ConfMatrix.jpg
```

## File-by-File Code Overview

### `data.py`

Handles dataset loading and splitting.

Main responsibilities:

- Read image files from `data/yes` and `data/no`
- Resize images to `64 x 64`
- Convert images to grayscale
- Normalize pixel values to `[0, 1]`
- Create labels
- Shuffle the dataset
- Build train/test, train/validation/test, and stratified k-fold splits

Important functions:

```text
load_image_as_array()
get_image_paths()
load_brain_mri_dataset()
train_test_split()
train_val_test_split()
create_stratified_k_folds()
load_brain_mri_kfold_data()
print_class_balance()
```

### `operations.py`

Contains low-level neural network math operations implemented manually with NumPy.

Main responsibilities:

- Zero padding
- Convolution forward pass
- Max pooling forward pass
- Flattening
- Dense layer forward pass
- ReLU and sigmoid activations
- Dense layer backward pass
- Convolution backward pass
- Max pooling backward pass
- ReLU backward pass

Important functions:

```text
zero_pad()
relu()
sigmoid()
conv_single_step()
conv_forward()
max_pool_forward()
flatten()
dense_forward()
relu_backward()
sigmoid_backward()
dense_backward()
conv_backward()
max_pool_backward()
```

### `cnn1_core.py`

Defines the CNN Base model itself.

Main responsibilities:

- Initialize CNN parameters
- Run forward propagation
- Apply dropout
- Compute binary cross-entropy cost with optional L2 regularization
- Run backpropagation
- Initialize Adam optimizer state
- Update parameters using Adam
- Predict labels and probabilities
- Load saved parameters from JSON

Important functions:

```text
dropout_forward()
dropout_backward()
initialize_cnn1_parameters()
cnn1_forward()
compute_cost()
cnn1_backward()
initialize_adam()
adam_update()
predict_cnn1()
load_cnn1_parameters_from_json()
```

### `cnn1_train.py`

Contains reusable training helpers.

This file is useful for smaller experiments or testing the CNN outside the full k-fold experiment.

Main responsibilities:

- Create mini-batches
- Apply learning rate decay
- Compute accuracy
- Train CNN1 on a normal train/validation split
- Evaluate CNN1 on test data

Important functions:

```text
create_mini_batches()
learning_rate_decay()
accuracy()
train_cnn1()
evaluate_cnn1()
```

### `main_cnn1.py`

This is the main experiment runner.

Main responsibilities:

- Create output directories
- Load the dataset
- Create stratified k-fold splits
- Train one fold at a time
- Print live progress bars
- Apply dynamic early stopping
- Restore best epoch parameters
- Calculate metrics
- Plot cost curves, accuracy curves, and confusion matrices
- Save summary JSON
- Save best-fold parameters

Important functions:

```text
make_output_dirs()
print_progress_bar()
confusion_matrix_binary()
classification_metrics()
evaluate_model()
train_one_fold()
run_cnn1_experiment()
summarize_kfold_metrics()
summarize_total_metrics()
save_experiment_summary()
save_updated_parameters()
```

## Prediction With Saved Parameters

After training, load the saved best-fold parameters and run prediction:

```python
from cnn1_core import load_cnn1_parameters_from_json, predict_cnn1

parameters = load_cnn1_parameters_from_json(
    "results/cnn1/metrics/updated_parameters_1.json"
)

predictions, probabilities = predict_cnn1(X_new, parameters)
```

`X_new` must be preprocessed in the same format used during training:

```text
(m, 64, 64, 1)
```

with pixel values normalized to `[0, 1]`.

## GitHub Notes

The repository ignores local dataset files and large generated parameter files.

Ignored:

```text
data/
archive.zip
*.zip
results/cnn1/metrics/updated_parameters_*.json
__pycache__/
```

This keeps the repository focused on source code, results summaries, and plots while avoiding large or dataset-specific files.

## Limitations

- The CNN is implemented from scratch, so training is slow compared to PyTorch or TensorFlow.
- The dataset is small and imbalanced.
- The model has high recall but still produces a noticeable number of false positives.
- Current implementation is only CNN Base; InceptionNet and ResNet are planned but not yet implemented.

## Next Steps

Planned improvements:

- Add InceptionNet from scratch using NumPy
- Add ResNet from scratch using NumPy
- Reuse the same data loading and metric pipeline for all architectures
- Compare CNN Base, InceptionNet, and ResNet using the same folds and hyperparameter policy
- Add a prediction script for single-image inference
- Improve README with dataset source and project report references
