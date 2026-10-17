"""
Configuration, enums, and data transfer schemas for head pose estimation.

Contains all domain settings, landmark indices, thresholds, and immutable DTOs
used across pipeline components without external I/O or rendering dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np

# StrEnum support (Python 3.11+ native, fallback for 3.10)
try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Enums & Landmark Indices
# ─────────────────────────────────────────────────────────────────────────────

class PitchLevel(StrEnum):
    """Categorical alertness state based on pitch droop."""
    NORMAL = "NORMAL"
    MILD_DROOP = "MILD_DROOP"
    FATIGUE_WARNING = "FATIGUE_WARNING"


class RollLevel(StrEnum):
    """Categorical head tilt state based on absolute roll angle."""
    NORMAL = "NORMAL"
    MILD_TILT = "MILD_TILT"
    SIGNIFICANT_TILT = "SIGNIFICANT_TILT"


@dataclass(frozen=True)
class LandmarkIndices:
    """dlib 68-landmark indices used for face frame and pose projection."""
    right_eye_outer: int = 36  # Outer corner of right eye (image-left)
    left_eye_outer: int = 45   # Outer corner of left eye (image-right)
    nose_tip: int = 33         # Subnasale / tip of nose
    chin: int = 8              # Gnathion / lowest point of chin

    @property
    def key_indices(self) -> Tuple[int, ...]:
        """Subset of landmarks drawn on preview HUD."""
        return (self.right_eye_outer, self.left_eye_outer, self.nose_tip, self.chin)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration Dataclasses (Domain-Scoped)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoseEstimatorConfig:
    """Anthropometric and numerical stability parameters for pose geometry."""
    kp_pitch: float = 0.18        # Anthropometric sensitivity constant for pitch
    epsilon: float = 1e-6         # Division-by-zero guard for vector norms & height
    landmarks: LandmarkIndices = field(default_factory=LandmarkIndices)


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for neutral baseline accumulation."""
    required_frames: int = 60     # Frames needed to establish neutral baseline (~2s at 30 FPS)


@dataclass(frozen=True)
class ClassifierConfig:
    """Thresholds and temporal hysteresis limits for fatigue detection."""
    # Pitch thresholds (degrees): negative values represent drooping downward
    pitch_thresh_mild: float = -8.0
    pitch_thresh_warning: float = -15.0

    # Roll thresholds (degrees): absolute lateral tilt deviation from baseline
    roll_thresh_mild: float = 8.0
    roll_thresh_warning: float = 18.0

    # Temporal persistence filter
    sustained_frames: int = 20    # Frames a condition must persist before escalating


@dataclass(frozen=True)
class VisualizerConfig:
    """Visual style, color mappings, and HUD layout settings."""
    level_colors: dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        PitchLevel.NORMAL: (0, 255, 0),              # Green
        PitchLevel.MILD_DROOP: (0, 200, 255),        # Yellow / Orange
        PitchLevel.FATIGUE_WARNING: (0, 0, 255),     # Red
        RollLevel.NORMAL: (0, 255, 0),               # Green
        RollLevel.MILD_TILT: (0, 200, 255),          # Yellow / Orange
        RollLevel.SIGNIFICANT_TILT: (0, 0, 255),     # Red
    })
    landmark_color: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    axis_x_color: Tuple[int, int, int] = (255, 255, 0)    # Cyan / Yellow
    progress_bar_color: Tuple[int, int, int] = (0, 230, 0)
    warning_banner_bg: Tuple[int, int, int] = (0, 0, 255)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pipeline Data Transfer Objects (Immutable Typed Results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaselinePose:
    """Calibrated neutral pose baseline values."""
    rho: float
    roll: float


@dataclass(frozen=True)
class PoseMetrics:
    """Computed 2D facial frame geometry and head orientation metrics."""
    pitch_deg: float
    roll_deg: float
    delta_roll: float
    rho: float
    e_mid: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    w_eyes: float


@dataclass(frozen=True)
class FatigueAssessment:
    """Classification outcome for pitch droop and roll tilt."""
    pitch_level: PitchLevel
    roll_level: RollLevel
    pitch_counter: int
    roll_counter: int

    @property
    def is_warning(self) -> bool:
        return self.pitch_level == PitchLevel.FATIGUE_WARNING
