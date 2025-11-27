import numpy as np


def overlay_mask_on_image(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    color=(144, 238, 144),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Create a color overlay on the image wherever mask == 1.

    Parameters
    ----------
    img_rgb : np.ndarray
        Original RGB image, shape (H, W, 3), uint8.
    mask : np.ndarray
        Binary mask, shape (H, W), values {0, 1}.
    color : tuple
        Overlay color (R, G, B) in 0–255.
    alpha : float
        Blending factor: 0 = original, 1 = full overlay color.

    Returns
    -------
    blended : np.ndarray
        RGB image with overlay, shape (H, W, 3), uint8.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must have shape (H, W, 3)")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")

    h, w = img_rgb.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {(h, w)}"
        )

    # Ensure mask is boolean
    mask_bool = mask.astype(bool)

    blended = img_rgb.astype("float32").copy()
    overlay_color = np.array(color, dtype="float32")

    # Apply blending where mask is True
    blended[mask_bool] = (
        (1.0 - alpha) * blended[mask_bool] + alpha * overlay_color
    )

    return blended.astype("uint8")
