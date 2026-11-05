"""
Unified Configuration, Enums, and Data Transfer Schemas for Complete ADAS Monitoring.

Consolidates all domain settings, indices, thresholds, and immutable DTOs for:
- Task 1: Head Pose Estimation & Pitch Droop
- Task 2: Driver Gaze Distraction (LightGazeCNN)
- Task 3: Facial Dynamics via EAR & MAR (Micro-sleep, 30s PERCLOS, Yawning)
- Task 4: Multimodal Alert Fusion Engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import os
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Enums
# ─────────────────────────────────────────────────────────────────────────────

class DriverRiskLevel(StrEnum):
    """Holistic categorical risk assessment for the driver."""
    SAFE = "SAFE"
    DISTRACTED = "DISTRACTED"
    DROWSY = "DROWSY"
    CRITICAL_RISK = "CRITICAL_RISK"


class CalibrationPhase(StrEnum):
    """Guided sequential calibration sequence."""
    IDLE = "IDLE"
    CALIB_HEAD_POSE = "CALIB_HEAD_POSE"    # Step 1: Neutral head pose (~2s)
    CALIB_OPEN_EYES = "CALIB_OPEN_EYES"    # Step 2: Open-eye EAR baseline (~3s)
    TRANSITION_EYES = "TRANSITION_EYES"    # Prompt eye closure (~1.5s)
    CALIB_CLOSED_EYES = "CALIB_CLOSED_EYES"# Step 3: Closed-eye EAR floor (~3s)
    ACTIVE = "ACTIVE"                      # Live multi-task monitoring


class PitchLevel(StrEnum):
    NORMAL = "NORMAL"
    MILD_DROOP = "MILD_DROOP"
    FATIGUE_WARNING = "FATIGUE_WARNING"


class GazeStatus(StrEnum):
    LOOKING_ROAD = "LOOKING_ROAD"
    NOT_LOOKING_ROAD = "NOT_LOOKING_ROAD"


class BackendType(StrEnum):
    """Supported driver gaze distraction inference model backends."""
    LIGHT_CNN = "light_cnn"
    CLSTM = "clstm"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Subsystem Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeadPoseConfig:
    kp_pitch: float = 0.18
    pitch_thresh_mild: float = -8.0
    pitch_thresh_warning: float = -15.0
    roll_thresh_warning: float = 18.0
    sustained_frames: int = 20
    required_calib_frames: int = 60


@dataclass(frozen=True)
class DistractionConfig:
    backend: BackendType = BackendType.LIGHT_CNN
    img_size: int = 112
    window_size: int = 15
    seq_len: int = 20
    weights_path: str = ""
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    hidden_dim: int = 128
    num_layers: int = 2
    feature_dim: int = 512
    num_classes: int = 2
    backbone_fallback_path: str = "resnet18_preprocess_clahe_full_model.pth"

    def resolve_weights_path(self) -> str:
        target = self.weights_path
        if not target:
            target = (
                "light_cnn_session_full_model.pth"
                if self.backend == BackendType.LIGHT_CNN
                else "resnet18_lstm_unified.pth"
            )

        candidates = [
            target,
            os.path.join(os.path.dirname(__file__), "..", target),
            os.path.join(os.path.dirname(__file__), "..", "..", target),
            os.path.join("CNN_Gaze", target),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return target

    @classmethod
    def for_light_cnn(
        cls,
        weights_path: str = "light_cnn_session_full_model.pth",
        window_size: int = 15,
        img_size: int = 112,
    ) -> DistractionConfig:
        return cls(
            backend=BackendType.LIGHT_CNN,
            img_size=img_size,
            window_size=window_size,
            weights_path=weights_path,
        )

    @classmethod
    def for_clstm(
        cls,
        weights_path: str = "resnet18_lstm_unified.pth",
        seq_len: int = 20,
        img_size: int = 224,
    ) -> DistractionConfig:
        return cls(
            backend=BackendType.CLSTM,
            img_size=img_size,
            seq_len=seq_len,
            weights_path=weights_path,
        )


@dataclass(frozen=True)
class EarMarConfig:
    factor_descenso: float = 0.70
    sensibilidad_cierre: float = 0.20
    min_dynamic_range: float = 0.05
    median_window: int = 5
    mov_avg_window: int = 40
    min_samples_for_update: int = 15
    blink_max_normal_sec: float = 0.50   # >= 500ms triggers micro-sleep
    perclos_window_sec: float = 30.0
    perclos_threshold_pct: float = 15.0  # >= 15% triggers somnolence warning
    mar_threshold: float = 0.55
    yawn_sustained_sec: float = 4.0      # >= 4s triggers yawn alert
    calib_step_sec: float = 3.0


@dataclass(frozen=True)
class VisualizerConfig:
    color_safe: Tuple[int, int, int] = (0, 220, 0)         # Green (BGR)
    color_distracted: Tuple[int, int, int] = (0, 140, 255) # Orange (BGR)
    color_drowsy: Tuple[int, int, int] = (0, 69, 255)      # Red-Orange (BGR)
    color_critical: Tuple[int, int, int] = (0, 0, 255)     # Vibrant Red (BGR)
    bg_container: Tuple[int, int, int] = (25, 25, 25)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (255, 255, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Transfer Objects (Immutable Typed Results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaselineValues:
    rho: float
    roll: float
    ear_open: float
    ear_closed: float
    thresh_descent: float
    thresh_closure: float


@dataclass(frozen=True)
class PoseOutput:
    pitch_deg: float
    roll_deg: float
    pitch_level: PitchLevel
    is_drooping: bool


@dataclass(frozen=True)
class DistractionOutput:
    status: GazeStatus
    confidence: float
    is_distracted: bool
    latency_ms: float
    buffer_len: int
    buffer_cap: int
    backend: BackendType = BackendType.LIGHT_CNN


@dataclass(frozen=True)
class EarMarOutput:
    ear_raw: float
    ear_filtered: float
    mar_raw: float
    thresh_descent: float
    thresh_closure: float
    last_blink_ms: float
    is_microsleep: bool
    perclos_pct: float
    is_perclos_alarm: bool
    is_perclos_warmed_up: bool
    is_yawning: bool
    is_yawn_alarm: bool


@dataclass(frozen=True)
class UnifiedDriverAssessment:
    risk_level: DriverRiskLevel
    status_banner: str
    primary_color: Tuple[int, int, int]
    phase: CalibrationPhase
    guidance_prompt: str
    pose: PoseOutput
    distraction: DistractionOutput
    ear_mar: EarMarOutput
    fps: float
    landmarks: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    stage_latencies_ms: Optional[Dict[str, float]] = None
