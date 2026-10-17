"""
Fatigue and distraction classification component for head pose estimation.

Performs threshold comparison and temporal hysteresis (debouncing) over
pitch droop (somnolence) and lateral roll tilt.
Extensible for PERCLOS blending, yaw distraction, or multi-stage alertness models.
"""

from __future__ import annotations

from typing import Optional

try:
    from .configuration import (
        BaselinePose,
        ClassifierConfig,
        FatigueAssessment,
        PitchLevel,
        PoseMetrics,
        RollLevel,
    )
except ImportError:
    from configuration import (
        BaselinePose,
        ClassifierConfig,
        FatigueAssessment,
        PitchLevel,
        PoseMetrics,
        RollLevel,
    )


class FatigueClassifier:
    """Classifies pitch droop (somnolence) and roll tilt with temporal debounce."""

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.reset()

    def reset(self) -> None:
        """Resets persistence counters for pitch droop and roll tilt."""
        self.pitch_counter: int = 0
        self.roll_counter: int = 0

    def classify(self, metrics: PoseMetrics, baseline: BaselinePose) -> FatigueAssessment:
        """Evaluates live pose metrics against calibrated baseline and alert thresholds."""
        cfg = self.config
        delta_roll = metrics.roll_deg - baseline.roll

        # Pitch droop classification (negative pitch = looking downward)
        if metrics.pitch_deg > cfg.pitch_thresh_mild:
            self.pitch_counter = 0
            pitch_level = PitchLevel.NORMAL
        elif metrics.pitch_deg > cfg.pitch_thresh_warning:
            self.pitch_counter += 1
            pitch_level = (
                PitchLevel.MILD_DROOP
                if self.pitch_counter > cfg.sustained_frames
                else PitchLevel.NORMAL
            )
        else:
            self.pitch_counter += 1
            pitch_level = (
                PitchLevel.FATIGUE_WARNING
                if self.pitch_counter > cfg.sustained_frames
                else PitchLevel.MILD_DROOP
            )

        # Roll tilt classification
        abs_droll = abs(delta_roll)
        if abs_droll < cfg.roll_thresh_mild:
            self.roll_counter = 0
            roll_level = RollLevel.NORMAL
        elif abs_droll < cfg.roll_thresh_warning:
            self.roll_counter += 1
            roll_level = (
                RollLevel.MILD_TILT
                if self.roll_counter > cfg.sustained_frames
                else RollLevel.NORMAL
            )
        else:
            self.roll_counter += 1
            roll_level = (
                RollLevel.SIGNIFICANT_TILT
                if self.roll_counter > cfg.sustained_frames
                else RollLevel.MILD_TILT
            )

        return FatigueAssessment(
            pitch_level=pitch_level,
            roll_level=roll_level,
            pitch_counter=self.pitch_counter,
            roll_counter=self.roll_counter,
        )
