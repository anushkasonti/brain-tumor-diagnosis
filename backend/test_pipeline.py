import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from backend.pipeline import full_pipeline


DATA_DIR = "data_samples"
VALID_EXTS = {".png", ".jpg", ".jpeg"}


def process_one_image(image_path: str):
    print("\n======================================")
    print(f"Processing: {image_path}")
    print("======================================")

    result = full_pipeline(image_path)

    print("=== PIPELINE RESULT ===")
    print(f"Has tumor       : {result['has_tumor']}")
    print(f"Detection prob  : {result['detection_prob']:.4f}")
    print(f"Predicted label : {result['predicted_label']}")
    if result["class_probs"] is None:
        print("Class probabilities: None (no tumor detected)")
    else:
        print("Class probabilities:")
        for cls, p in result["class_probs"].items():
            print(f"  {cls:10s}: {p:.4f}")

    mask = result["segmentation_mask"]
    overlay = result["overlay_image"]

    if mask is None:
        print("Segmentation mask: None (no tumor detected)")
    else:
        print(f"Segmentation mask shape: {mask.shape}")
        print(f"Mask unique values: {set(mask.flatten())}")

    print(f"Overlay image shape: {overlay.shape}")

    # ---- Visualization (same as before) ----
    orig_img = Image.open(image_path).convert("L")

    h, w, _ = overlay.shape
    orig_img = orig_img.resize((w, h))
    orig_np = np.array(orig_img).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Predicted Tumor Mask (overlay)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()  # Close the window to move to next image


def main():
    files = [
        f
        for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]
    files.sort()

    if not files:
        print(f"No images found in {DATA_DIR}")
        return

    for fname in files:
        image_path = os.path.join(DATA_DIR, fname)
        process_one_image(image_path)


if __name__ == "__main__":
    main()
