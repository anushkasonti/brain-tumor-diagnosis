import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from backend.segmentation_model import UNet

# --- CONFIGURATION (MUST MATCH TRAINING / predict.py) ---

# MODEL_PATH = os.path.join("../models", "segmentation", "segmentation_model.pth")
MODEL_PATH = os.path.join("models", "segmentation", "segmentation_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224  # same as in your original predict.py


# Same transforms as in predict.py: resize + ToTensor ONLY
_seg_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # (1, H, W) in [0,1]
    ]
)

# --- MODEL LOADING ---

# IMPORTANT: n_channels=1 because the model was trained on grayscale images
_unet_model = UNet(n_channels=1, n_classes=1)
_unet_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
)
_unet_model.to(DEVICE)
_unet_model.eval()


def run_segmentation(rgb_image: np.ndarray) -> np.ndarray:
    """
    Run UNet segmentation on an in-memory RGB image.

    Parameters
    ----------
    rgb_image : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB.

    Returns
    -------
    mask_resized : np.ndarray
        Binary mask of shape (H, W), dtype uint8, values {0, 1},
        resized back to the original image size.
    """

    # Remember original size (for upsampling the mask later)
    orig_h, orig_w = rgb_image.shape[:2]

    # Convert numpy RGB -> PIL Image -> grayscale "L"
    pil_image = Image.fromarray(rgb_image).convert("L")

    # Apply the SAME transforms as in predict.py
    input_tensor = _seg_transform(pil_image)  # shape: (1, H, W)

    # Add batch dimension and move to device: (1, 1, H, W)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output_logits = _unet_model(input_batch)  # (1, 1, H, W)
        output_probs = torch.sigmoid(output_logits)

    # Threshold at 0.5 to get binary mask
    pred_mask = (output_probs > 0.5).float().cpu().squeeze(0).squeeze(0)  # (H, W)

    # Convert to numpy uint8
    mask_np = pred_mask.numpy().astype(np.uint8)  # 0 or 1, size IMAGE_SIZE x IMAGE_SIZE

    # Resize mask back to original image size using nearest neighbor
    mask_pil = Image.fromarray(mask_np * 255)
    mask_resized_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
    mask_resized = (np.array(mask_resized_pil) > 0).astype(np.uint8)  # 0 or 1

    return mask_resized
