"""
train.py
--------
Trains a MobileNetV2-based waste classifier on the 6-class dataset.

Workflow:
  1. Load image paths + labels
  2. Stratified 70 / 15 / 15 split
  3. Build tf.data pipelines with augmentation on the training set
  4. Phase A: freeze base, train classification head (up to 15 epochs)
  5. Phase B: unfreeze top 30 layers, fine-tune (up to 10 epochs)
  6. Save best model → models/waste_classifier.keras
  7. Save training curves → outputs/

Run:
    python src/train.py
"""

import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image as PILImage

# ─── Project root (one level above src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "Dataset" / "extracted" / "dataset-resized"
MODEL_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
MODEL_PATH  = MODEL_DIR / "waste_classifier.keras"

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
CLASSES          = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE       = (224, 224)
BATCH_SIZE       = 32
SEED             = 42
HEAD_EPOCHS      = 15       # Phase A: train head only
FINETUNE_EPOCHS  = 10       # Phase B: fine-tune top layers
DROPOUT_RATE     = 0.4
DENSE_UNITS      = 128
FINETUNE_AT      = 100      # unfreeze layers from this index onward in MobileNetV2

# ─── Reproducibility ──────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def setup_tensorflow():
    """Import TensorFlow (stable or nightly) and set seeds."""
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
        print(f"TensorFlow version: {tf.__version__}")
        return tf
    except ImportError as exc:
        print(f"[ERROR] TensorFlow not available: {exc}")
        sys.exit(1)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_image_paths_labels() -> tuple[list[str], list[int]]:
    """Scan DATASET_DIR and return (image_paths, integer_labels)."""
    supported_exts = {".jpg", ".jpeg", ".png"}
    paths, labels = [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"[WARNING] Class folder not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in supported_exts:
                paths.append(str(img_path))
                labels.append(class_idx)

    print(f"Total images found: {len(paths)}")
    return paths, labels


def split_data(paths, labels):
    """Stratified 70 / 15 / 15 split."""
    # First split off test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        paths, labels,
        test_size=0.15,
        random_state=SEED,
        stratify=labels
    )
    # Split remainder into train (70/85 ≈ 82.4%) and val (15/85 ≈ 17.6%)
    val_ratio_of_temp = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_of_temp,
        random_state=SEED,
        stratify=y_temp
    )
    print(f"Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_indices(X_train, X_val, X_test, y_train, y_val, y_test):
    """Persist the split so evaluate.py can reproduce the exact test set."""
    split_data = {
        "train": {"paths": X_train, "labels": y_train},
        "val":   {"paths": X_val,   "labels": y_val},
        "test":  {"paths": X_test,  "labels": y_test},
    }
    split_path = MODEL_DIR / "data_split.json"
    with open(split_path, "w") as f:
        json.dump(split_data, f)
    print(f"Split indices saved -> {split_path}")


# ─── tf.data Pipeline ─────────────────────────────────────────────────────────
def build_dataset(tf, paths, labels, training: bool, batch_size: int):
    """
    Build a tf.data.Dataset from file paths + labels.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        # MobileNetV2 expects pixels in [-1, 1]
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ─── Model Building ───────────────────────────────────────────────────────────
def build_model(tf) -> tuple:
    """
    MobileNetV2 classification head with integrated augmentation layers.
    Base is frozen initially; fine-tuning unlocks top layers in Phase B.
    """
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # Data augmentation (active during training mode only)
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08)(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


# ─── Training Plots ───────────────────────────────────────────────────────────
def save_training_curves(history_a, history_b=None):
    """Save accuracy and loss curves combining Phase A + B histories."""
    acc  = history_a.history["accuracy"]
    val_acc  = history_a.history["val_accuracy"]
    loss = history_a.history["loss"]
    val_loss = history_a.history["val_loss"]

    if history_b:
        acc      += history_b.history["accuracy"]
        val_acc  += history_b.history["val_accuracy"]
        loss     += history_b.history["loss"]
        val_loss += history_b.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    finetune_start = len(history_a.history["accuracy"]) + 1

    # ── Accuracy curve ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, acc,     label="Training Accuracy",   linewidth=2)
    ax.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2, linestyle="--")
    if history_b:
        ax.axvline(x=finetune_start, color="gray", linestyle=":", label="Fine-tuning starts")
    ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = OUTPUT_DIR / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Accuracy curve saved -> {acc_path}")

    # ── Loss curve ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, loss,     label="Training Loss",   linewidth=2)
    ax.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2, linestyle="--")
    if history_b:
        ax.axvline(x=finetune_start, color="gray", linestyle=":", label="Fine-tuning starts")
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = OUTPUT_DIR / "loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved -> {loss_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("   SMART WASTE CLASSIFICATION — TRAINING")
    print("=" * 60)

    # Validate dataset
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import TF
    tf = setup_tensorflow()

    # Load data
    print("\n[1/7] Loading image paths...")
    paths, labels = load_image_paths_labels()
    if len(paths) == 0:
        print("[ERROR] No images found. Check dataset path.")
        sys.exit(1)

    # Split
    print("\n[2/7] Splitting data (70/15/15 stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(paths, labels)
    save_split_indices(X_train, X_val, X_test, y_train, y_val, y_test)

    # Class weights (handles imbalance — trash only 137 samples)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print(f"\nClass weights: { {CLASSES[k]: round(v, 3) for k, v in class_weight_dict.items()} }")

    # Build tf.data datasets
    print("\n[3/7] Building tf.data pipelines...")
    train_ds = build_dataset(tf, X_train, y_train, training=True,  batch_size=BATCH_SIZE)
    val_ds   = build_dataset(tf, X_val,   y_val,   training=False, batch_size=BATCH_SIZE)
    test_ds  = build_dataset(tf, X_test,  y_test,  training=False, batch_size=BATCH_SIZE)

    # Build model
    print("\n[4/7] Building MobileNetV2 model...")
    model, base_model = build_model(tf)
    model.summary(print_fn=lambda x: print("  " + x))

    # ── Phase A: Train classification head ─────────────────────────────────
    print(f"\n[5/7] Phase A — Training head ({HEAD_EPOCHS} epochs max)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_a = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks_a,
        class_weight=class_weight_dict
    )

    # ── Phase B: Fine-tune upper layers ────────────────────────────────────
    print(f"\n[6/7] Phase B — Fine-tuning top layers ({FINETUNE_EPOCHS} epochs max)...")
    base_model.trainable = True
    # Freeze all layers except those after FINETUNE_AT
    for layer in base_model.layers[:FINETUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_b = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks_b,
        class_weight=class_weight_dict
    )

    # Save curves
    print("\n[7/7] Saving training curves...")
    save_training_curves(history_a, history_b)

    # Final report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    best_val_acc = max(
        max(history_a.history["val_accuracy"]),
        max(history_b.history["val_accuracy"])
    )
    print(f"  Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved              : {MODEL_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
