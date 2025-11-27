import numpy as np
from typing import Union
from pathlib import Path

import cv2
from PIL import Image


# ----------------------------------------------------------------------
# Common image helpers
# ----------------------------------------------------------------------

DET_IMG_SIZE = 224
CLS_IMG_SIZE = 224
SEG_IMG_SIZE = 224


def _open_image(path_or_file: Union[str, Path, "IO"]) -> Image.Image:
    """
    Open an image from a filesystem path or a file-like object
    and return a PIL Image in RGB mode.
    """
    img = Image.open(path_or_file).convert("RGB")
    return img


# ----------------------------------------------------------------------
# Public API used by backend.pipeline and (previously) Streamlit
# ----------------------------------------------------------------------

def load_image_from_path(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk and return an RGB numpy array.

    Returns
    -------
    img_rgb : np.ndarray
        Shape (H, W, 3), dtype uint8.
    """
    img = _open_image(path)
    return np.array(img)


def load_image_from_streamlit(file) -> np.ndarray:
    """
    Load an uploaded image from Streamlit and return an RGB numpy array.

    Parameters
    ----------
    file : UploadedFile
        Object returned by st.file_uploader.

    Returns
    -------
    img_rgb : np.ndarray
        Shape (H, W, 3), dtype uint8.
    """
    img = _open_image(file)
    return np.array(img)


def prepare_for_detection(img_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB image for the detection model.

    Steps:
    - Convert to grayscale
    - Resize to 224×224
    - Scale to [0, 1]
    - Return with shape (1, 224, 224, 1)  [NHWC]
    """
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    gray = cv2.resize(gray, (DET_IMG_SIZE, DET_IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = gray.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0

    x = x[..., np.newaxis]            # (H, W, 1)
    x = np.expand_dims(x, axis=0)     # (1, H, W, 1)
    return x


def prepare_for_classification(img_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB image for the classification model.

    Steps:
    - Resize to 224×224
    - Scale to [0, 1] if needed
    - Convert to (1, 3, 224, 224)  [NCHW]

    NOTE:
    Your classification_inference.run_classification currently performs
    its own normalization from (H,W,3) -> (1,3,H,W), so in the pipeline
    we usually pass the raw RGB image. This function exists for backward
    compatibility (e.g. test scripts) and returns a sensible tensor.
    """
    img = cv2.resize(img_rgb, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = img.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0

    x = np.transpose(x, (2, 0, 1))    # (3, H, W)
    x = np.expand_dims(x, axis=0)     # (1, 3, H, W)
    return x


def prepare_for_segmentation_input(img_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB image for the segmentation model.

    Steps:
    - Convert to grayscale
    - Resize to 224×224
    - Scale to [0, 1]
    - Return with shape (1, 1, 224, 224)  [NCHW]
    """
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    gray = cv2.resize(gray, (SEG_IMG_SIZE, SEG_IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = gray.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0

    x = np.expand_dims(x, axis=0)     # (1, H, W)
    x = np.expand_dims(x, axis=0)     # (1, 1, H, W)
    return x
