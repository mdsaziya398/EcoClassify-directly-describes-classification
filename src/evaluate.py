"""
evaluate.py
-----------
Evaluates the saved waste classifier on the held-out test set.

Generates:
  - Test accuracy, precision, recall, F1
  - outputs/classification_report.txt
  - outputs/confusion_matrix.png

Run:
    python src/evaluate.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_PATH     = PROJECT_ROOT / "models" / "waste_classifier.keras"
SPLIT_PATH     = PROJECT_ROOT / "models" / "data_split.json"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"
REPORT_PATH    = OUTPUT_DIR / "classification_report.txt"
CM_PATH        = OUTPUT_DIR / "confusion_matrix.png"

CLASSES        = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE     = (224, 224)
BATCH_SIZE     = 32


def setup_tensorflow():
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        return tf
    except ImportError as exc:
        print(f"[ERROR] TensorFlow not available: {exc}")
        sys.exit(1)


def load_split():
    """Load the test paths + labels saved by train.py."""
    if not SPLIT_PATH.exists():
        print(f"[ERROR] Split file not found: {SPLIT_PATH}")
        print("  Please run src/train.py first.")
        sys.exit(1)
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
    return split["test"]["paths"], split["test"]["labels"]


def build_test_dataset(tf, paths, labels):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = (
        ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return ds


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Waste Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {CM_PATH}")


def main():
    print("\n" + "=" * 60)
    print("   SMART WASTE CLASSIFICATION — EVALUATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("  Run src/train.py first.")
        sys.exit(1)

    tf = setup_tensorflow()
    print(f"\nLoading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. [OK]")

    # Load test split
    print("\nLoading test split ...")
    test_paths, test_labels = load_split()
    print(f"Test set size: {len(test_paths)} images")

    # Build dataset
    test_ds = build_test_dataset(tf, test_paths, test_labels)

    # Predict
    print("\nRunning predictions ...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(test_labels)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'='*60}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print("\nClassification Report:\n")
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved -> {REPORT_PATH}")

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred)

    print("\nEvaluation complete. [OK]\n")


if __name__ == "__main__":
    main()
