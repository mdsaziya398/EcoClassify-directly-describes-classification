"""
dataset_analysis.py
-------------------
Scans Dataset/extracted/dataset-resized/ and reports:
  - Which class folders exist
  - How many images per class (only .jpg / .jpeg / .png)
  - Corrupted / unreadable images
  - Overall summary

Run:
    python src/dataset_analysis.py
"""

import sys
from pathlib import Path
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR = Path("Dataset") / "extracted" / "dataset-resized"
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_image(path: Path) -> bool:
    """Return True if the file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_corrupted(path: Path) -> bool:
    """Try to open the image; return True if it cannot be read."""
    try:
        with Image.open(path) as img:
            img.verify()          # lightweight check — does not decode full image
        return False
    except Exception:
        return True


def analyse_class(class_dir: Path) -> dict:
    """Return a dict with image count, format breakdown, and corrupted list."""
    all_files = list(class_dir.iterdir())
    image_files = [f for f in all_files if is_image(f)]
    non_image_files = [f for f in all_files if f.is_file() and not is_image(f)]
    corrupted = [f for f in image_files if is_corrupted(f)]

    format_counts: dict[str, int] = {}
    for f in image_files:
        ext = f.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1

    return {
        "total_files": len(all_files),
        "image_count": len(image_files),
        "non_image_count": len(non_image_files),
        "format_counts": format_counts,
        "corrupted": corrupted,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 55)
    print("   SMART WASTE CLASSIFICATION — DATASET ANALYSIS")
    print("=" * 55)

    # 1. Verify dataset root exists
    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found:\n  {DATASET_DIR.resolve()}")
        print("Please ensure the dataset is extracted to Dataset/extracted/dataset-resized/")
        sys.exit(1)

    print(f"\nDataset path : {DATASET_DIR.resolve()}")

    # 2. Detect class folders
    found_classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    print(f"Folders found: {found_classes}")

    # 3. Check for expected classes
    missing = [c for c in EXPECTED_CLASSES if c not in found_classes]
    extra = [c for c in found_classes if c not in EXPECTED_CLASSES]
    if missing:
        print(f"\n[WARNING] Missing expected class folders: {missing}")
    if extra:
        print(f"[INFO]    Extra folders (ignored): {extra}")

    # 4. Per-class analysis
    print("\n" + "-" * 55)
    print(f"{'Class':<14} {'Images':>7}  {'Formats':<22}  {'Corrupted':>9}")
    print("-" * 55)

    total_images = 0
    total_corrupted = 0
    class_image_counts: dict[str, int] = {}

    for cls in EXPECTED_CLASSES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            print(f"{cls:<14} {'MISSING':>7}")
            class_image_counts[cls] = 0
            continue

        info = analyse_class(cls_dir)
        fmt_str = ", ".join(f"{k}:{v}" for k, v in sorted(info["format_counts"].items()))
        corrupted_count = len(info["corrupted"])

        print(
            f"{cls:<14} {info['image_count']:>7}  {fmt_str:<22}  {corrupted_count:>9}"
        )
        total_images += info["image_count"]
        total_corrupted += corrupted_count
        class_image_counts[cls] = info["image_count"]

        if info["corrupted"]:
            for bad_file in info["corrupted"]:
                print(f"              [CORRUPTED] {bad_file.name}")

    # 5. Summary
    print("-" * 55)
    print(f"{'TOTAL':<14} {total_images:>7}")
    print(f"\nCorrupted images found: {total_corrupted}")

    # 6. Class distribution
    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION")
    print("=" * 55)
    for cls, count in class_image_counts.items():
        pct = (count / total_images * 100) if total_images else 0
        bar = "#" * int(pct / 2)
        print(f"  {cls:<14} {count:>4}  ({pct:5.1f}%)  {bar}")

    # 7. Imbalance warning
    counts = list(class_image_counts.values())
    max_c, min_c = max(counts), min(counts)
    ratio = max_c / min_c if min_c > 0 else float("inf")
    if ratio > 2:
        print(
            f"\n[WARNING] Class imbalance detected - ratio {ratio:.1f}x "
            f"(max={max_c}, min={min_c})."
        )
        print("  Stratified splitting and class weighting are recommended.")

    # 8. Recommended split sizes
    print("\n" + "=" * 55)
    print("RECOMMENDED SPLIT SIZES  (70 / 15 / 15)")
    print("=" * 55)
    train_n  = int(total_images * 0.70)
    val_n    = int(total_images * 0.15)
    test_n   = total_images - train_n - val_n
    print(f"  Training   : ~{train_n}")
    print(f"  Validation : ~{val_n}")
    print(f"  Test       : ~{test_n}")
    print()

    print("Dataset analysis complete. [OK]\n")


if __name__ == "__main__":
    main()

