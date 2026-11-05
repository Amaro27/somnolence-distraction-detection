"""
Image transformation and preprocessing components for driver gaze detection.

Provides CLAHE contrast equalization and PyTorch tensor pipelines for both
112x112 LightGazeCNN and 224x224 GazeResNetLSTM inputs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    from .configuration import CLSTMConfig, LightGazeConfig
except ImportError:
    from configuration import CLSTMConfig, LightGazeConfig


class CLAHETransform:
    """Applies Contrast Limited Adaptive Histogram Equalization in luminance space."""

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img.convert("L"))
        enhanced = self.clahe.apply(np_img)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_enhanced)


class FramePreprocessor:
    """Prepares raw OpenCV BGR frames into normalized PyTorch model input tensors."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, transform: transforms.Compose, device: torch.device):
        self.transform = transform
        self.device = device

    @classmethod
    def for_light_cnn(
        cls,
        config: Optional[LightGazeConfig] = None,
        device: Optional[torch.device] = None,
    ) -> FramePreprocessor:
        cfg = config or LightGazeConfig()
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
        ])
        return cls(transform=pipeline, device=dev)

    @classmethod
    def for_clstm(
        cls,
        config: Optional[CLSTMConfig] = None,
        device: Optional[torch.device] = None,
    ) -> FramePreprocessor:
        cfg = config or CLSTMConfig()
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clahe = CLAHETransform(
            clip_limit=cfg.clahe_clip_limit,
            tile_grid_size=cfg.clahe_tile_grid_size,
        )
        pipeline = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            clahe,
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
        ])
        return cls(transform=pipeline, device=dev)

    def preprocess_bgr(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Converts BGR frame to RGB PIL Image and applies PyTorch normalization."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        tensor: torch.Tensor = self.transform(pil_img)
        return tensor.unsqueeze(0).to(self.device)

    def preprocess_pil(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocesses an already loaded PIL RGB image."""
        tensor: torch.Tensor = self.transform(pil_img.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)
