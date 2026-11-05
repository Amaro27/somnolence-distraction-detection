"""
Task 2: Driver Gaze Distraction Inference (LightGazeCNN & C-LSTM).

Provides:
- LightGazeCNN: Lightweight 4-stage convolutional gaze classification with mode temporal smoothing.
- GazeResNetLSTM (C-LSTM): ResNet18 spatial feature extraction with recurrent LSTM sequence modeling.
"""

from __future__ import annotations

from collections import Counter, deque
import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from .configuration import (
        BackendType,
        DistractionConfig,
        DistractionOutput,
        GazeStatus,
    )
except ImportError:
    from configuration import (
        BackendType,
        DistractionConfig,
        DistractionOutput,
        GazeStatus,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing Transforms
# ─────────────────────────────────────────────────────────────────────────────

class CLAHETransform:
    """Applies Contrast Limited Adaptive Histogram Equalization in luminance space."""

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img.convert("L"))
        enhanced = self.clahe.apply(np_img)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_enhanced)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Architectures
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
        return self.classifier(self.features(x))


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
# 3. Model Weight Checkpoint Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_light_cnn(
    config: Optional[DistractionConfig] = None,
    device: Optional[torch.device] = None,
) -> LightGazeCNN:
    """Instantiates LightGazeCNN and loads trained weights."""
    cfg = config or DistractionConfig.for_light_cnn()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGazeCNN(num_classes=cfg.num_classes).to(dev)
    path = cfg.resolve_weights_path()

    if os.path.isfile(path):
        try:
            ckpt = torch.load(path, map_location=dev, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            print(f"[INFO] Loaded LightGazeCNN weights from: {path}")
        except Exception as e:
            print(f"[WARNING] Error loading LightGazeCNN weights: {e}")
    else:
        print(f"[WARNING] LightGazeCNN weights not found at: {path}")

    model.eval()
    return model


def load_clstm(
    config: Optional[DistractionConfig] = None,
    device: Optional[torch.device] = None,
) -> GazeResNetLSTM:
    """Instantiates GazeResNetLSTM and loads trained weights."""
    cfg = config or DistractionConfig.for_clstm()
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

            if any(k.startswith("feature_extractor.") for k in state_dict.keys()):
                model.load_state_dict(state_dict, strict=True)
                print(f"[INFO] Loaded unified GazeResNetLSTM weights from: {path}")
            else:
                model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded LSTM & Classifier from: {path}")

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
        print(f"[WARNING] GazeResNetLSTM weights file not found at: {path}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified Distraction Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class DistractionInferenceEngine:
    """Manages preprocessing, neural forward inference, and temporal smoothing for both backends."""

    def __init__(
        self,
        config: Optional[DistractionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or DistractionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = self.config.backend

        if self.backend == BackendType.CLSTM:
            img_size = self.config.img_size if self.config.img_size != 112 else 224
            clahe = CLAHETransform(
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=self.config.clahe_tile_grid_size,
            )
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                clahe,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model: nn.Module = load_clstm(self.config, self.device)
            self._feature_buffer: deque[torch.Tensor] = deque(maxlen=self.config.seq_len)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = load_light_cnn(self.config, self.device)
            self._pred_buffer: deque[int] = deque(maxlen=self.config.window_size)
            self._conf_buffer: deque[float] = deque(maxlen=self.config.window_size)

    def reset(self) -> None:
        """Clears temporal state buffers."""
        if self.backend == BackendType.CLSTM:
            self._feature_buffer.clear()
        else:
            self._pred_buffer.clear()
            self._conf_buffer.clear()

    def process_frame(self, frame_bgr: np.ndarray) -> DistractionOutput:
        """Runs end-to-end preprocessing, neural inference, and temporal modeling."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if self.backend == BackendType.CLSTM:
            # 1. Spatial 512-d feature extraction
            frame_feat = self.model.extract_single_frame_feature(tensor)
            self._feature_buffer.append(frame_feat)

            # 2. Recurrent LSTM sequence evaluation
            seq_tensor = torch.cat(list(self._feature_buffer), dim=0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model.forward_sequence_features(seq_tensor)
                probs = torch.softmax(logits, dim=1)
                conf_t, pred_t = torch.max(probs, 1)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            raw_pred = int(pred_t.item())
            confidence = float(conf_t.item() * 100.0)
            is_distracted = (raw_pred == 0)
            status = GazeStatus.NOT_LOOKING_ROAD if is_distracted else GazeStatus.LOOKING_ROAD

            return DistractionOutput(
                status=status,
                confidence=confidence,
                is_distracted=is_distracted,
                latency_ms=lat_ms,
                buffer_len=len(self._feature_buffer),
                buffer_cap=self.config.seq_len,
                backend=BackendType.CLSTM,
            )

        else:
            # LightGazeCNN Feedforward
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf_t, pred_t = torch.max(probs, 1)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            raw_pred = int(pred_t.item())
            raw_conf = float(conf_t.item() * 100.0)

            self._pred_buffer.append(raw_pred)
            self._conf_buffer.append(raw_conf)

            most_common = Counter(self._pred_buffer).most_common(1)[0][0]
            smoothed_conf = float(np.mean(self._conf_buffer))

            is_distracted = (most_common == 0)
            status = GazeStatus.NOT_LOOKING_ROAD if is_distracted else GazeStatus.LOOKING_ROAD

            return DistractionOutput(
                status=status,
                confidence=smoothed_conf,
                is_distracted=is_distracted,
                latency_ms=lat_ms,
                buffer_len=len(self._pred_buffer),
                buffer_cap=self.config.window_size,
                backend=BackendType.LIGHT_CNN,
            )
