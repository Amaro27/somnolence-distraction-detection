"""
Neutral baseline calibration component for head pose estimation.

Accumulates neutral pose measurements over a configurable frame buffer
to establish individual baselines for vertical ratio (rho) and roll.
Extensible for custom outlier filtering, confidence weighting, or dynamic recalibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from .configuration import (
        BaselinePose,
        CalibrationConfig,
        PoseMetrics,
    )
except ImportError:
    from configuration import (
        BaselinePose,
        CalibrationConfig,
        PoseMetrics,
    )


class PoseCalibrator:
    """Accumulates neutral pose measurements over N frames to determine baseline."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.reset()

    def reset(self) -> None:
        """Clears accumulated samples and invalidates current baseline."""
        self._rho_buffer: list[float] = []
        self._roll_buffer: list[float] = []
        self._baseline: Optional[BaselinePose] = None

    @property
    def is_calibrated(self) -> bool:
        """Indicates whether a baseline has been successfully computed."""
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[BaselinePose]:
        """The computed baseline or None if still accumulating."""
        return self._baseline

    @property
    def progress(self) -> float:
        """Completion ratio in [0.0, 1.0] towards baseline establishment."""
        return min(len(self._rho_buffer) / self.config.required_frames, 1.0)

    def accumulate(self, metrics: PoseMetrics) -> Optional[BaselinePose]:
        """Accumulates neutral sample and computes median baseline once full."""
        self._rho_buffer.append(metrics.rho)
        self._roll_buffer.append(metrics.roll_deg)

        if len(self._rho_buffer) >= self.config.required_frames:
            self._baseline = BaselinePose(
                rho=float(np.median(self._rho_buffer)),
                roll=float(np.median(self._roll_buffer)),
            )
            return self._baseline
        return None
