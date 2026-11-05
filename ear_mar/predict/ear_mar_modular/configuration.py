"""
Configuration, enums, and data transfer schemas for EAR and MAR somnolence detection.

Contains domain settings, threshold ratios, landmark ranges, and immutable DTOs
used across calibration, filtering, fatigue analyzers, and HUD presentation.
Zero OpenCV/dlib dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Enums & Indices
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationPhase(StrEnum):
    """Lifecycle phases for two-stage physiological eye calibration."""
    IDLE = "IDLE"                      # Awaiting user activation ('c' key)
    CALIB_OPEN = "CALIB_OPEN"          # Measuring open-eye baseline (~4s)
    TRANSITION = "TRANSITION"          # User prompt to close eyes (~2s)
    CALIB_CLOSED = "CALIB_CLOSED"      # Measuring closed-eye floor (~4s)
    ACTIVE = "ACTIVE"                  # Calibration complete, active monitoring


class AlertSeverity(StrEnum):
    """Categorical severity of driver fatigue events."""
    NORMAL = "NORMAL"
    YAWN_ALERT = "YAWN_ALERT"
    PERCLOS_WARNING = "PERCLOS_WARNING"
    MICROSLEEP_CRITICAL = "MICROSLEEP_CRITICAL"


@dataclass(frozen=True)
class EarMarLandmarkIndices:
    """dlib 68-point index slices for eyes and mouth."""
    right_eye_start: int = 36
    right_eye_end: int = 42
    left_eye_start: int = 42
    left_eye_end: int = 48
    mouth_start: int = 48
    mouth_end: int = 68


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters for two-phase baseline establishment."""
    phase_duration_sec: float = 4.0      # Duration to sample open/closed states
    transition_duration_sec: float = 2.0 # Pause duration prompting eye closure
    factor_descenso: float = 0.70        # 70% relative aperture to start blink cycle
    sensibilidad_cierre: float = 0.20    # 20% remaining aperture to confirm closure
    min_dynamic_range: float = 0.05      # Safety guard for eye aperture difference


@dataclass(frozen=True)
class AdaptiveFilterConfig:
    """Parameters for cascaded median filter and moving average baseline tracking."""
    median_window: int = 5               # Median filter length for raw EAR smoothing
    mov_avg_window: int = 40             # Moving average window length for dynamic drift
    min_samples_for_update: int = 15     # Minimum samples in moving average before adaptation


@dataclass(frozen=True)
class BlinkConfig:
    """Parameters for blink duration tracking and micro-sleep detection."""
    max_normal_duration_sec: float = 0.50 # Events >= 500ms are classified as prolonged / micro-sleep
    alarm_cooldown_sec: float = 1.2       # Duration to hold micro-sleep visual alarm on screen


@dataclass(frozen=True)
class PerclosConfig:
    """Parameters for cumulative percentage of eye closure tracking."""
    window_sec: float = 30.0             # Time horizon window in seconds
    threshold_fatigue_pct: float = 15.0  # 15.0% closure over 30s triggers somnolence warning


@dataclass(frozen=True)
class YawnConfig:
    """Parameters for sustained mouth opening classification."""
    mar_threshold: float = 0.55          # Mouth Aspect Ratio threshold
    sustained_duration_sec: float = 4.0  # Must be sustained >= 4.0s to confirm yawn
    alarm_cooldown_sec: float = 1.5      # Duration to hold yawn visual alarm on screen


@dataclass(frozen=True)
class VisualizerConfig:
    """Styling, color schemes, and HUD positioning for telemetry."""
    color_normal: Tuple[int, int, int] = (0, 230, 0)       # Green (BGR)
    color_warning: Tuple[int, int, int] = (0, 140, 255)    # Orange (BGR)
    color_critical: Tuple[int, int, int] = (0, 0, 255)     # Red (BGR)
    color_text: Tuple[int, int, int] = (255, 255, 255)     # White
    color_accent: Tuple[int, int, int] = (0, 255, 255)     # Yellow
    hud_bg_color: Tuple[int, int, int] = (25, 25, 25)      # Dark Gray


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Transfer Objects (Immutable Typed Results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibratedThresholds:
    """Physiological baseline and derived trigger thresholds."""
    ear_base_open: float
    ear_base_closed: float
    dynamic_range: float
    threshold_descent: float
    threshold_closure: float


@dataclass(frozen=True)
class EarMarMetrics:
    """Instantaneous facial aspect ratio metrics for current frame."""
    ear_left: float
    ear_right: float
    ear_raw: float
    ear_median: float
    ear_mov_avg: float
    mar_raw: float
    threshold_descent: float
    threshold_closure: float


@dataclass(frozen=True)
class BlinkAssessment:
    """Live state of eye blink cycle and micro-sleep validation."""
    in_blink_cycle: bool
    reached_effective_closure: bool
    last_blink_duration_ms: float
    is_microsleep_alarm: bool


@dataclass(frozen=True)
class PerclosAssessment:
    """Live state of 30-second rolling window eye closure accumulation."""
    perclos_pct: float
    is_warmed_up: bool
    warmup_remaining_sec: float
    is_perclos_alarm: bool


@dataclass(frozen=True)
class YawnAssessment:
    """Live state of sustained mouth aspect ratio opening."""
    is_yawning: bool
    current_yawn_duration_sec: float
    is_yawn_alarm: bool


@dataclass(frozen=True)
class EarMarStateAssessment:
    """Consolidated multi-metric fatigue assessment for a single frame."""
    phase: CalibrationPhase
    metrics: EarMarMetrics
    blink: BlinkAssessment
    perclos: PerclosAssessment
    yawn: YawnAssessment
    severity: AlertSeverity
    calib_progress: float
    status_summary: str
