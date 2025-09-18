import os
import sys
import argparse
from pathlib import Path
from typing import Iterable, List

# Quiet TensorFlow logs a bit
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist


# ------------------------------- Config -------------------------------- #

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL_PATH = "digit_model.keras"


# ------------------------------- Helpers ------------------------------- #

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def preprocess_image(path: Path, target_size=(28, 28)) -> np.ndarray:
    """Load image as grayscale 28x28, scale to [0,1], return (28,28,1) float32."""
    img = image.load_img(str(path), color_mode="grayscale", target_size=target_size)
    arr = image.img_to_array(img).astype("float32") / 255.0  # (28,28,1)
    return arr


def iter_image_paths(root_dir: Path) -> Iterable[Path]:
    """Yield all image file paths recursively under root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        d = Path(dirpath)
        for fname in filenames:
            p = d / fname
            if is_image_file(p):
                yield p


def normalize_images_root(root_dir: Path) -> Path:
    """
    If root has exactly one subfolder and no images, descend into that subfolder.
    This handles archives like: root_dir/digits/<images>.
    """
    root_dir = root_dir.resolve()
    try:
        entries = list(root_dir.iterdir())
    except Exception:
        return root_dir

    has_images_here = any((root_dir / name).is_file() and (root_dir / name).suffix.lower() in IMG_EXTS
                          for name in os.listdir(root_dir))
    subdirs = [x for x in entries if x.is_dir()]
    if not has_images_here and len(subdirs) == 1:
        return subdirs[0]
    return root_dir


# ------------------------- Model build / train ------------------------- #

def build_and_train_model(epochs: int = 3) -> tf.keras.Model:
    """
    Small CNN trained on MNIST (downloads automatically on first run).
    Good baseline for MNIST-like digits.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
    return model


def try_load_model(model_path: Path) -> tf.keras.Model | None:
    """
    Try to load a saved model from either:
      - .keras file
      - SavedModel directory (contains saved_model.pb)
    Return None if not loadable.
    """
    try:
        if model_path.is_file() and model_path.suffix.lower() == ".keras":
            print(f"[INFO] Loading Keras model file: {model_path}")
            return load_model(model_path)
        if model_path.is_dir():
            # Heuristic: SavedModel has a saved_model.pb file inside
            pb = model_path / "saved_model.pb"
            if pb.exists():
                print(f"[INFO] Loading SavedModel directory: {model_path}")
                return load_model(model_path)
        return None
    except Exception as e:
        print(f"[WARN] Could not load model from {model_path}: {e}")
        return None


# ------------------------------ Inference ----------------------------- #

def count_digits_batched(model: tf.keras.Model, images_dir: Path, batch_size: int = 64) -> List[int]:
    """
    Batch predictions for speed. Returns counts[0..9].
    Skips unreadable/corrupt images but continues.
    """
    images_dir = normalize_images_root(images_dir)
    counts = [0] * 10
    batch_arrays: List[np.ndarray] = []
    batch_paths: List[Path] = []
    total = 0

    for p in iter_image_paths(images_dir):
        try:
            arr = preprocess_image(p)  # (28,28,1)
            batch_arrays.append(arr)
            batch_paths.append(p)

            if len(batch_arrays) >= batch_size:
                preds = model.predict(np.stack(batch_arrays, axis=0), verbose=0)  # (B,10)
                labels = np.argmax(preds, axis=1)
                for lab in labels:
                    counts[int(lab)] += 1
                total += len(batch_arrays)
                batch_arrays.clear()
                batch_paths.clear()
        except Exception as e:
            print(f"[WARN] Skipping {p} ({e})")

    # Flush remaining
    if batch_arrays:
        preds = model.predict(np.stack(batch_arrays, axis=0), verbose=0)
        labels = np.argmax(preds, axis=1)
        for lab in labels:
            counts[int(lab)] += 1
        total += len(batch_arrays)

    print(f"[INFO] Processed {total} image(s).")
    return counts


# --------------------------------- CLI -------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Count digits (0–9) in a folder using a CNN classifier."
    )
    parser.add_argument(
        "--images", required=True,
        help="Path to the folder containing digit images (recurses into subfolders)."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to save/load the model. Use '.keras' for a single-file Keras model, "
             "or a directory path for TensorFlow SavedModel. Default: digit_model.keras"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Epochs to train when no saved model is found or when --force-retrain is used."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save the model after training."
    )
    parser.add_argument(
        "--save-format", choices=["keras", "savedmodel"], default="keras",
        help="Save as '.keras' file or TensorFlow SavedModel directory (default: keras)."
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Ignore any saved model and train a fresh one."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size used for prediction (default: 64)."
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[ERROR] Images folder not found or not a directory: {images_dir}")
        sys.exit(1)

    # Make TF play nicer with GPU memory if present
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    model_path = Path(args.model)

    # Load or train
    model = None
    if not args.force_retrain:
        model = try_load_model(model_path)

    if model is None:
        print("[INFO] No usable saved model found or force retrain requested. Training a new model on MNIST...")
        model = build_and_train_model(epochs=args.epochs)

    # Count (we do this BEFORE trying to save, so saving can’t block results)
    counts = count_digits_batched(model, images_dir, batch_size=args.batch_size)

    # Pretty print for you + raw list for code consumption
    pretty = [f"digit_{i}: {c} times" for i, c in enumerate(counts)]
    print("[ " + ", ".join(pretty) + " ]")
    print("Raw counts:", counts)

    # Save (optional)
    if not args.no_save:
        try:
            if args.save_format == "keras":
                # Ensure path ends with .keras
                if model_path.is_dir():
                    # If a directory was passed but format=keras, turn it into a file inside it
                    model_path.mkdir(parents=True, exist_ok=True)
                    model_path = model_path / "digit_model.keras"
                else:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Saving model to {model_path}")
                model.save(model_path)  # native Keras format
            else:
                # SavedModel directory
                if not model_path.exists():
                    model_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Exporting SavedModel to {model_path}")
                model.save(model_path)  # directory export
        except Exception as e:
            print(f"[WARN] Save failed: {e} (continuing)")


if __name__ == "__main__":
    main()
