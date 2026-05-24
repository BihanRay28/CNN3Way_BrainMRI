import numpy as np

from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_as_array(image_path, image_size=(64, 64), grayscale=True):
    """
    Loads one image, resizes it, normalizes pixels to [0, 1].

    grayscale=True:
        returns shape (64, 64, 1)

    grayscale=False:
        returns shape (64, 64, 3)
    """

    image = Image.open(image_path)

    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(image_size)

    image_array = np.array(image, dtype=np.float32) / 255.0

    if grayscale:
        image_array = np.expand_dims(image_array, axis=-1)

    return image_array


def get_image_paths(folder_path):
    """
    Gets all image files from a folder.

    This does not depend on numbering.
    Weird names like Y1.jpg, Y10.jpg, no 3.jpg all work.
    """

    folder_path = Path(folder_path)

    image_paths = []

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file_path)

    return image_paths


def load_brain_mri_dataset(
    data_dir="data",
    image_size=(64, 64),
    grayscale=True,
    shuffle=True,
    seed=42
):
    """
    Expected folder structure:

    CNN3Way/
    ├── data.py
    ├── data/
    │   ├── yes/
    │   └── no/

    Labels:
        yes -> 1
        no  -> 0

    Returns:
        X shape: (m, height, width, channels)
        Y shape: (m, 1)
    """

    data_dir = Path(data_dir)

    yes_dir = data_dir / "yes"
    no_dir = data_dir / "no"

    if not yes_dir.exists():
        raise FileNotFoundError(f"Could not find folder: {yes_dir}")

    if not no_dir.exists():
        raise FileNotFoundError(f"Could not find folder: {no_dir}")

    X_list = []
    Y_list = []

    yes_paths = get_image_paths(yes_dir)
    no_paths = get_image_paths(no_dir)

    print(f"Found tumor images: {len(yes_paths)}")
    print(f"Found no-tumor images: {len(no_paths)}")

    for image_path in yes_paths:
        try:
            image_array = load_image_as_array(
                image_path,
                image_size=image_size,
                grayscale=grayscale
            )

            X_list.append(image_array)
            Y_list.append(1)

        except Exception as e:
            print(f"Skipping corrupted/unreadable image: {image_path}")
            print("Reason:", e)

    for image_path in no_paths:
        try:
            image_array = load_image_as_array(
                image_path,
                image_size=image_size,
                grayscale=grayscale
            )

            X_list.append(image_array)
            Y_list.append(0)

        except Exception as e:
            print(f"Skipping corrupted/unreadable image: {image_path}")
            print("Reason:", e)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)

    if shuffle:
        np.random.seed(seed)
        permutation = np.random.permutation(X.shape[0])

        X = X[permutation]
        Y = Y[permutation]

    return X, Y


def train_test_split(
    X,
    Y,
    test_size=0.2,
    seed=42,
    shuffle=True
):
    """
    Creates a normal train/test split.

    Returns:
        X_train, Y_train, X_test, Y_test
    """

    m = X.shape[0]
    indices = np.arange(m)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    test_count = int(m * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    Y_train = Y[train_indices]

    X_test = X[test_indices]
    Y_test = Y[test_indices]

    return X_train, Y_train, X_test, Y_test


def train_val_test_split(
    X,
    Y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    shuffle=True
):
    """
    Creates train/validation/test split.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """

    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    m = X.shape[0]
    indices = np.arange(m)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_end = int(m * train_size)
    val_end = train_end + int(m * val_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train = X[train_indices]
    Y_train = Y[train_indices]

    X_val = X[val_indices]
    Y_val = Y[val_indices]

    X_test = X[test_indices]
    Y_test = Y[test_indices]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_stratified_k_folds(
    X,
    Y,
    k=5,
    seed=42,
    shuffle=True
):
    """
    Creates stratified k-fold cross-validation splits.

    This keeps the class ratio similar in every fold.

    Labels:
        0 -> no tumor
        1 -> tumor

    Returns:
        folds

    Each fold is:
        X_train, Y_train, X_val, Y_val
    """

    if k <= 1:
        raise ValueError("k must be greater than 1")

    Y_flat = Y.reshape(-1)

    class_0_indices = np.where(Y_flat == 0)[0]
    class_1_indices = np.where(Y_flat == 1)[0]

    if k > len(class_0_indices) or k > len(class_1_indices):
        raise ValueError("k cannot be greater than the number of samples in the smallest class")

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

    class_0_folds = np.array_split(class_0_indices, k)
    class_1_folds = np.array_split(class_1_indices, k)

    folds = []

    for fold_index in range(k):
        val_indices = np.concatenate(
            [
                class_0_folds[fold_index],
                class_1_folds[fold_index]
            ]
        )

        train_indices = np.concatenate(
            [
                np.concatenate(
                    [class_0_folds[i] for i in range(k) if i != fold_index]
                ),
                np.concatenate(
                    [class_1_folds[i] for i in range(k) if i != fold_index]
                )
            ]
        )

        if shuffle:
            np.random.seed(seed + fold_index)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        X_val = X[val_indices]
        Y_val = Y[val_indices]

        folds.append((X_train, Y_train, X_val, Y_val))

    return folds


def load_brain_mri_kfold_data(
    data_dir="data",
    image_size=(64, 64),
    grayscale=True,
    k=5,
    seed=42
):
    """
    Loads the Brain MRI dataset and creates stratified k-fold splits.

    Returns:
        X, Y, folds
    """

    X, Y = load_brain_mri_dataset(
        data_dir=data_dir,
        image_size=image_size,
        grayscale=grayscale,
        shuffle=True,
        seed=seed
    )

    folds = create_stratified_k_folds(
        X,
        Y,
        k=k,
        seed=seed,
        shuffle=True
    )

    return X, Y, folds

def print_class_balance(Y, title="Class balance"):
    """
    Prints class counts.

    Label:
        0 -> no tumor
        1 -> tumor
    """

    total = Y.shape[0]
    tumor_count = int(np.sum(Y == 1))
    no_tumor_count = int(np.sum(Y == 0))

    print(title)
    print("Total:", total)
    print("Tumor:", tumor_count)
    print("No tumor:", no_tumor_count)

    if total > 0:
        print("Tumor %:", round((tumor_count / total) * 100, 2))
        print("No tumor %:", round((no_tumor_count / total) * 100, 2))

if __name__ == "__main__":
    X, Y = load_brain_mri_dataset(
        data_dir="data",
        image_size=(64, 64),
        grayscale=True,
        shuffle=True,
        seed=42
    )

    print()
    print("Dataset loaded successfully.")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print_class_balance(Y, title="Full dataset balance")

    print()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
        X,
        Y,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=42,
        shuffle=True
    )

    print("Train/Val/Test split:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape)
    print("Y_val:", Y_val.shape)
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)

    print()
    folds = create_stratified_k_folds(
        X,
        Y,
        k=5,
        seed=42,
        shuffle=True
    )

    print("Stratified K-fold split:")
    print("Number of folds:", len(folds))

    for i, fold in enumerate(folds):
        X_train_fold, Y_train_fold, X_val_fold, Y_val_fold = fold

        print(f"Fold {i + 1}")
        print("X_train:", X_train_fold.shape)
        print("Y_train:", Y_train_fold.shape)
        print("X_val:", X_val_fold.shape)
        print("Y_val:", Y_val_fold.shape)

        print_class_balance(Y_train_fold, title="Train fold balance")
        print_class_balance(Y_val_fold, title="Validation fold balance")

        print("-" * 40)