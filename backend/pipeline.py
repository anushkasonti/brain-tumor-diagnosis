import os
import numpy as np

from utils.preprocessing import (
    load_image_from_path,
    prepare_for_detection,
    # prepare_for_classification,  # not needed anymore
)
from utils.visualization import overlay_mask_on_image
from backend.classification_inference import run_classification
from backend.detection_inference import run_detection
from backend.segmentation_inference import run_segmentation

# You can tune this later based on detection model performance
TUMOR_THRESHOLD = 0.5


def _run_pipeline_core(img_rgb: np.ndarray) -> dict:
    """
    Core pipeline logic operating on an in-memory RGB image.

    Steps:
    1. Run detection:
       - If no tumor: return early, no classification/segmentation.
       - If tumor: continue.
    2. Run classification to get tumor type.
    3. Run segmentation to get binary mask.
    4. Create overlay image (original + green tumor region).
    """

    # 1. Detection
    det_input = prepare_for_detection(img_rgb)
    prob_tumor = run_detection(det_input)

    has_tumor = float(prob_tumor) >= TUMOR_THRESHOLD

    # If no tumor: skip classification and segmentation
    if not has_tumor:
        return {
            "has_tumor": False,
            "detection_prob": float(prob_tumor),
            "predicted_label": None,
            "class_probs": None,
            "segmentation_mask": None,
            # just return original image as overlay
            "overlay_image": img_rgb,
        }

    # 2. Tumor present -> classification
    # run_classification expects an unbatched image (H, W, 3) or (H, W)
    pred_label, probs = run_classification(img_rgb)

    # 3. Segmentation
    mask = run_segmentation(img_rgb)  # (H, W) binary {0,1}

    # 4. Overlay
    overlay = overlay_mask_on_image(img_rgb, mask)

    return {
        "has_tumor": True,
        "detection_prob": float(prob_tumor),
        "predicted_label": pred_label,
        "class_probs": probs,
        "segmentation_mask": mask,
        "overlay_image": overlay,
    }


def full_pipeline(image_path: str) -> dict:
    """
    Pipeline entry point when you have an image path on disk.
    """
    img_rgb = load_image_from_path(image_path)
    return _run_pipeline_core(img_rgb)


def full_pipeline_from_array(img_rgb: np.ndarray) -> dict:
    """
    Pipeline entry point when you already have an RGB numpy image
    (e.g. from Streamlit file uploader). Shape (H, W, 3), dtype uint8.
    """
    return _run_pipeline_core(img_rgb)
