"""
Neural network architectures and robust checkpoint loaders for driver gaze detection.

Contains LightGazeCNN (lightweight feedforward CNN) and GazeResNetLSTM (ResNet18 + LSTM),
with automated weight discovery and fallback loading.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .configuration import CLSTMConfig, LightGazeConfig
except ImportError:
    from configuration import CLSTMConfig, LightGazeConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. LightGazeCNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class LightGazeCNN(nn.Module):
    """Lightweight 4-stage convolutional neural network for driver gaze classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GazeResNetLSTM Architecture
# ─────────────────────────────────────────────────────────────────────────────

class GazeResNetLSTM(nn.Module):
    """Hybrid spatial-temporal model combining ResNet-18 spatial features with LSTM."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes),
        )

    def extract_single_frame_feature(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim embedding from a single frame (1, C, H, W) -> (1, 512)."""
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor)
        return feat.view(1, self.feature_dim)

    def forward_sequence_features(self, feat_sequence: torch.Tensor) -> torch.Tensor:
        """Propagates (1, Seq_Len, 512) feature sequence through LSTM to produce logits."""
        lstm_out, _ = self.lstm(feat_sequence)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)

    def forward(self, img_sequence: torch.Tensor) -> torch.Tensor:
        """Full end-to-end forward pass for (Batch, Seq, C, H, W)."""
        batch_size, seq_len, c, h, w = img_sequence.shape
        flat_imgs = img_sequence.view(batch_size * seq_len, c, h, w)
        feats = self.feature_extractor(flat_imgs).view(batch_size, seq_len, self.feature_dim)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Loaders & Checkpoint Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_light_cnn(
    config: Optional[LightGazeConfig] = None,
    device: Optional[torch.device] = None,
) -> LightGazeCNN:
    """Instantiates LightGazeCNN and loads trained weights."""
    cfg = config or LightGazeConfig()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightGazeCNN(num_classes=cfg.num_classes).to(dev)
    path = cfg.resolve_weights_path()

    if os.path.isfile(path):
        try:
            ckpt = torch.load(path, map_location=dev, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt)
            print(f"[INFO] Loaded LightGazeCNN weights from: {path}")
        except Exception as err:
            print(f"[WARNING] Could not load checkpoint from {path}: {err}. Using uninitialized weights.")
    else:
        print(f"[WARNING] LightGazeCNN weights file not found at '{path}'. Running uninitialized.")

    model.eval()
    return model


def load_clstm(
    config: Optional[CLSTMConfig] = None,
    device: Optional[torch.device] = None,
) -> GazeResNetLSTM:
    """Instantiates GazeResNetLSTM and loads unified or split ResNet18/LSTM weights."""
    cfg = config or CLSTMConfig()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GazeResNetLSTM(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
    ).to(dev)

    path = cfg.resolve_weights_path()

    if os.path.isfile(path):
        try:
            ckpt = torch.load(path, map_location=dev, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

            # Unified checkpoint with feature_extractor
            if any(k.startswith("feature_extractor.") for k in state_dict.keys()):
                model.load_state_dict(state_dict, strict=True)
                print(f"[INFO] Loaded unified GazeResNetLSTM weights from: {path}")
            else:
                # Checkpoint has only LSTM + Classifier
                model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded LSTM & Classifier from: {path}")

                # Load pretrained backbone
                bb_path = cfg.backbone_fallback_path
                if os.path.isfile(bb_path):
                    bb_state = torch.load(bb_path, map_location=dev, weights_only=False)
                    unified_bb = {}
                    for k, v in bb_state.items():
                        if not k.startswith("fc."):
                            if k.startswith("conv1."):
                                new_k = k.replace("conv1.", "feature_extractor.0.")
                            elif k.startswith("bn1."):
                                new_k = k.replace("bn1.", "feature_extractor.1.")
                            elif k.startswith("layer1."):
                                new_k = k.replace("layer1.", "feature_extractor.4.")
                            elif k.startswith("layer2."):
                                new_k = k.replace("layer2.", "feature_extractor.5.")
                            elif k.startswith("layer3."):
                                new_k = k.replace("layer3.", "feature_extractor.6.")
                            elif k.startswith("layer4."):
                                new_k = k.replace("layer4.", "feature_extractor.7.")
                            else:
                                new_k = f"feature_extractor.{k}"
                            unified_bb[new_k] = v
                    model.load_state_dict(unified_bb, strict=False)
                    print(f"[INFO] Loaded ResNet18 backbone weights from: {bb_path}")
        except Exception as err:
            print(f"[WARNING] Could not load GazeResNetLSTM weights: {err}")
    else:
        print(f"[WARNING] GazeResNetLSTM weights file not found at '{path}'. Running uninitialized.")

    model.eval()
    return model
