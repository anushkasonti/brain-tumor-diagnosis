import os
import numpy as np
import tensorflow as tf

# Adjusted to use your actual file name
# MODEL_PATH = os.path.join("../models", "detection", "final_model.keras")
# detection_inference.py
MODEL_PATH = os.path.join("models", "detection", "final_model.keras")


# Load the Keras detection model once at import time
_det_model = tf.keras.models.load_model(MODEL_PATH)


def run_detection(image: np.ndarray) -> float:
    """
    Run tumor detection on a single image.

    Parameters
    ----------
    image : np.ndarray
        Expected shape (H, W) or (H, W, 1), grayscale, values in [0, 255] or [0, 1].
        Our preprocessing will ensure it is 224x224 and grayscale.

    Returns
    -------
    prob_tumor : float
        Probability that tumor is present (0.0 – 1.0).
    """
    x = image.astype("float32")

    # If pixel values are 0–255, scale to 0–1
    if x.max() > 1.0:
        x = x / 255.0

    # We want final shape: (1, 224, 224, 1)  [NHWC]

    if x.ndim == 2:
        # (H, W) -> (H, W, 1) -> (1, H, W, 1)
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)

    elif x.ndim == 3:
        # Could be (H, W, 1)
        if x.shape[-1] == 1:
            # Add batch dimension -> (1, H, W, 1)
            x = np.expand_dims(x, axis=0)
        else:
            raise ValueError(
                f"run_detection expected grayscale (H,W) or (H,W,1), "
                f"got shape {x.shape}"
            )

    elif x.ndim == 4:
        # Assume already batched, e.g. output of prepare_for_detection: (1, H, W, 1)
        # Do nothing
        pass

    else:
        raise ValueError(f"Unexpected input shape for detection: {x.shape}")

    # Forward pass
    preds = _det_model.predict(x)

    # Common case: model outputs shape (1, 1) with sigmoid
    prob_tumor = float(preds[0][0])
    return prob_tumor
