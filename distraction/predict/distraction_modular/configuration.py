"""
Configuration, enums, and data transfer schemas for distraction and gaze detection.

Contains domain settings, model configs, thresholds, and immutable DTOs
used across real-time detection and offline evaluation components without
heavy I/O dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Tuple

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Enums
# ─────────────────────────────────────────────────────────────────────────────

class GazeClass(IntEnum):
    """Underlying model classification output indices."""
    NOT_LOOKING_ROAD = 0
    LOOKING_ROAD = 1


class DistractionStatus(StrEnum):
    """Categorical alertness state based on gaze and attention direction."""
    SAFE = "SAFE"
    DISTRACTED = "DISTRACTED"


class BackendType(StrEnum):
    """Supported detection inference model backends."""
    LIGHT_CNN = "light_cnn"
    CLSTM = "clstm"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model & Pipeline Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LightGazeConfig:
    """Settings for LightGazeCNN model and mode temporal smoothing."""
    img_size: int = 112
    window_size: int = 15
    weights_path: str = "light_cnn_session_full_model.pth"
    num_classes: int = 2
    confidence_threshold: float = 0.5

    def resolve_weights_path(self) -> str:
        """Finds weights file checking relative paths, current dir, and parent dirs."""
        candidates = [
            self.weights_path,
            os.path.join(os.path.dirname(__file__), "..", "..", self.weights_path),
            os.path.join("CNN_Gaze", self.weights_path),
            f"{self.weights_path}.pth",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        return self.weights_path


@dataclass(frozen=True)
class CLSTMConfig:
    """Settings for GazeResNetLSTM architecture and sequential feature modeling."""
    img_size: int = 224
    seq_len: int = 20
    feature_dim: int = 512
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 2
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    weights_path: str = "resnet18_lstm_unified.pth"
    backbone_fallback_path: str = "resnet18_preprocess_clahe_full_model.pth"

    def resolve_weights_path(self) -> str:
        candidates = [
            self.weights_path,
            os.path.join(os.path.dirname(__file__), "..", "..", self.weights_path),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        return self.weights_path


@dataclass(frozen=True)
class VisualizerConfig:
    """Styling, color schemes, and HUD positioning for distraction warnings."""
    status_colors: Dict[DistractionStatus, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            DistractionStatus.SAFE: (0, 230, 0),       # Vibrant Green (BGR)
            DistractionStatus.DISTRACTED: (0, 0, 255),  # Red (BGR)
        }
    )
    hud_bg_color: Tuple[int, int, int] = (25, 25, 25)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (255, 255, 0)  # Cyan/Yellow
    font_scale_main: float = 0.85
    font_scale_sub: float = 0.65
    thickness: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Transfer Objects (Immutable Typed Results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RawInferenceResult:
    """Instantaneous per-frame neural network forward pass output."""
    pred_idx: int
    raw_label: str
    confidence: float
    probabilities: Tuple[float, ...]
    latency_ms: float


@dataclass(frozen=True)
class DistractionAssessment:
    """Temporal evaluated distraction state presented to UI and control systems."""
    status: DistractionStatus
    smoothed_label: str
    raw_label: str
    confidence: float
    is_distracted: bool
    buffer_length: int
    buffer_capacity: int
    inference_latency_ms: float


@dataclass(frozen=True)
class BenchmarkStats:
    """Descriptive statistics for hardware latency and theoretical throughput."""
    sample_count: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    fps_cap: float
