"""
predict.py
----------
Runs inference on a single image and prints the predicted class + confidence.

Usage:
    python src/predict.py path/to/image.jpg

Output:
    Predicted class : Plastic
    Confidence      : 94.32%
"""

import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "waste_classifier.keras"

CLASSES      = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMAGE_SIZE   = (224, 224)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def setup_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        print(f"[ERROR] TensorFlow not available: {exc}")
        sys.exit(1)


def load_and_preprocess(tf, image_path: Path):
    """Load an image, resize, and preprocess for MobileNetV2."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)   # add batch dimension
        return arr
    except Exception as exc:
        print(f"[ERROR] Could not read image: {exc}")
        sys.exit(1)


def predict(image_path_str: str):
    image_path = Path(image_path_str)

    # ── Validate path ────────────────────────────────────────────────────
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)
    if not image_path.is_file():
        print(f"[ERROR] Path is not a file: {image_path}")
        sys.exit(1)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"[ERROR] Unsupported format '{image_path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Please run 'python src/train.py' first.")
        sys.exit(1)

    tf = setup_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH)

    # ── Preprocess + Predict ─────────────────────────────────────────────
    input_arr = load_and_preprocess(tf, image_path)
    predictions = model.predict(input_arr, verbose=0)[0]

    predicted_idx = int(predictions.argmax())
    predicted_class = CLASSES[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    # ── Output ───────────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"  Image           : {image_path.name}")
    print(f"  Predicted class : {predicted_class}")
    print(f"  Confidence      : {confidence:.2f}%")
    print("=" * 40)
    print("\nAll class probabilities:")
    for cls, prob in zip(CLASSES, predictions):
        bar = "#" * int(prob * 30)
        print(f"  {cls:<12} {prob * 100:6.2f}%  {bar}")
    print()

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        print("Example: python src/predict.py test_image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
