import os
from typing import Dict, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

# ------------- Model definition (must match training script) -------------


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.mean((2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return torch.relu(out)


class SmallResNetSE(nn.Module):
    def __init__(self, num_classes: int = 3):
        # 3 classes: glioma, meningioma, pituitary
        super().__init__()
        # NOTE: in_channels=1 to match the trained checkpoint
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_c: int, out_c: int, blocks: int, stride: int):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# ------------------------- Inference utilities ---------------------------

# Order MUST match the training label order
CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

# # Path to the trained classification model
# MODEL_PATH = os.path.join("../models", "classification", "best_model_unified.pth")
MODEL_PATH = os.path.join("models", "classification", "best_model_unified.pth")


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = SmallResNetSE(num_classes=len(CLASS_NAMES))

# Load weights
# state = torch.load(MODEL_PATH, map_location=_device)
state = torch.load(str(MODEL_PATH), map_location=_device)
if isinstance(state, dict) and "state_dict" in state:
    _model.load_state_dict(state["state_dict"])
else:
    _model.load_state_dict(state)

_model.to(_device)
_model.eval()


def run_classification(image: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Run classification on a single image.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB or (H, W) grayscale.
        If values are 0–255 they are scaled to [0, 1].

    Returns
    -------
    pred_label : str
        Predicted tumor type.
    probs_dict : dict
        Mapping from class name to probability.
    """
    # Convert to grayscale because the model is 1-channel
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    x = gray.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0

    # (H, W) -> (1, 1, H, W)
    x = np.expand_dims(x, axis=0)   # (1, H, W)
    x = np.expand_dims(x, axis=0)   # (1, 1, H, W)

    tensor = torch.from_numpy(x).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    probs_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

    return pred_label, probs_dict
