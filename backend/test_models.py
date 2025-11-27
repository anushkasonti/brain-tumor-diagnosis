import os

from utils.preprocessing import (
    load_image_from_path,
    prepare_for_detection,
    prepare_for_classification,
)
from backend.detection_inference import run_detection
from backend.classification_inference import run_classification


def main():
    # Adjust name here if your sample is jpg instead of png
    image_path = os.path.join("data_samples", "sample1.png")

    print(f"[INFO] Loading image from: {image_path}")
    image = load_image_from_path(image_path)

    # Detection preprocessing
    det_img = prepare_for_detection(image)
    # Classification preprocessing
    cls_img = prepare_for_classification(image)

    # Run detection
    print("[INFO] Running detection model...")
    prob_tumor = run_detection(det_img)
    print(f"Detection – tumor probability: {prob_tumor:.4f}")

    # Run classification
    print("[INFO] Running classification model...")
    pred_label, probs = run_classification(cls_img)
    print(f"Classification – predicted class: {pred_label}")
    print("Class probabilities:")
    for cls, p in probs.items():
        print(f"  {cls:10s}: {p:.4f}")


if __name__ == "__main__":
    main()
