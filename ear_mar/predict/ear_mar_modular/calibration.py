"""
Physiological two-phase eye calibration component.

Executes a guided state machine to measure user open-eye baseline and closed-eye
floor, deriving individualized descent and deep closure thresholds.
"""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

try:
    from .configuration import (
        CalibrationConfig,
        CalibrationPhase,
        CalibratedThresholds,
    )
except ImportError:
    from configuration import (
        CalibrationConfig,
        CalibrationPhase,
        CalibratedThresholds,
    )


class TwoPhaseEarCalibrator:
    """Manages the two-stage calibration state machine and threshold calculations."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.reset()

    def reset(self) -> None:
        """Resets calibration state machine and buffers."""
        self.phase: CalibrationPhase = CalibrationPhase.IDLE
        self._phase_start_time: float = 0.0
        self._open_samples: List[float] = []
        self._closed_samples: List[float] = []
        self._thresholds: Optional[CalibratedThresholds] = None

    @property
    def is_calibrated(self) -> bool:
        return self._thresholds is not None

    @property
    def thresholds(self) -> Optional[CalibratedThresholds]:
        return self._thresholds

    def start_calibration(self, current_time: Optional[float] = None) -> None:
        """Triggers the start of Phase 1 (measuring open eyes)."""
        t = current_time if current_time is not None else time.time()
        self.phase = CalibrationPhase.CALIB_OPEN
        self._phase_start_time = t
        self._open_samples.clear()
        self._closed_samples.clear()
        self._thresholds = None
        print("[INFO] Started Phase 1: Calibrating Open Eyes. Keep eyes open looking straight.")

    def update(self, ear_val: float, current_time: Optional[float] = None) -> CalibrationPhase:
        """
        Advances calibration state machine with current EAR sample.

        Returns
        -------
        CalibrationPhase
            The active state phase.
        """
        if self.phase == CalibrationPhase.IDLE or self.phase == CalibrationPhase.ACTIVE:
            return self.phase

        t = current_time if current_time is not None else time.time()
        elapsed = t - self._phase_start_time

        if self.phase == CalibrationPhase.CALIB_OPEN:
            if ear_val > 0.0:
                self._open_samples.append(ear_val)

            if elapsed >= self.config.phase_duration_sec:
                self.phase = CalibrationPhase.TRANSITION
                self._phase_start_time = t
                print("[INFO] Phase 1 completed. Prepare to close eyes...")

        elif self.phase == CalibrationPhase.TRANSITION:
            if elapsed >= self.config.transition_duration_sec:
                self.phase = CalibrationPhase.CALIB_CLOSED
                self._phase_start_time = t
                print("[INFO] Started Phase 2: Calibrating Closed Eyes. Keep eyes gently closed.")

        elif self.phase == CalibrationPhase.CALIB_CLOSED:
            if ear_val > 0.0:
                self._closed_samples.append(ear_val)

            if elapsed >= self.config.phase_duration_sec:
                self._thresholds = self._compute_thresholds()
                self.phase = CalibrationPhase.ACTIVE
                print(
                    f"[INFO] Calibration Complete! Open: {self._thresholds.ear_base_open:.3f} | "
                    f"Closed: {self._thresholds.ear_base_closed:.3f} | "
                    f"Descent Thresh: {self._thresholds.threshold_descent:.3f} | "
                    f"Closure Thresh: {self._thresholds.threshold_closure:.3f}"
                )

        return self.phase

    def _compute_thresholds(self) -> CalibratedThresholds:
        ear_open = float(np.mean(self._open_samples)) if self._open_samples else 0.30
        ear_closed = float(np.mean(self._closed_samples)) if self._closed_samples else 0.15

        dynamic_range = max(self.config.min_dynamic_range, ear_open - ear_closed)

        # Descent threshold (70% relative aperture to start blink cycle)
        thresh_descent = ear_closed + (self.config.factor_descenso * dynamic_range)

        # Deep closure threshold (20% remaining aperture to confirm closure)
        thresh_closure = ear_closed + (self.config.sensibilidad_cierre * dynamic_range)

        return CalibratedThresholds(
            ear_base_open=ear_open,
            ear_base_closed=ear_closed,
            dynamic_range=dynamic_range,
            threshold_descent=thresh_descent,
            threshold_closure=thresh_closure,
        )

    def get_phase_progress(self, current_time: Optional[float] = None) -> float:
        """Returns progress in [0.0, 1.0] for the active calibration step."""
        if self.phase == CalibrationPhase.IDLE:
            return 0.0
        if self.phase == CalibrationPhase.ACTIVE:
            return 1.0

        t = current_time if current_time is not None else time.time()
        elapsed = t - self._phase_start_time

        if self.phase in (CalibrationPhase.CALIB_OPEN, CalibrationPhase.CALIB_CLOSED):
            return min(elapsed / self.config.phase_duration_sec, 1.0)
        elif self.phase == CalibrationPhase.TRANSITION:
            return min(elapsed / self.config.transition_duration_sec, 1.0)
        return 0.0

    def get_guidance_prompt(self, current_time: Optional[float] = None) -> str:
        """Returns human-readable instructions for the UI overlay."""
        t = current_time if current_time is not None else time.time()
        elapsed = t - self._phase_start_time

        if self.phase == CalibrationPhase.IDLE:
            return "Press 'c' to start 2-phase eye calibration"
        elif self.phase == CalibrationPhase.CALIB_OPEN:
            rem = max(0, int(self.config.phase_duration_sec - elapsed) + 1)
            return f"PHASE 1: Keep eyes OPEN looking forward ({rem}s)"
        elif self.phase == CalibrationPhase.TRANSITION:
            return "CLOSE EYES NOW AND HOLD..."
        elif self.phase == CalibrationPhase.CALIB_CLOSED:
            rem = max(0, int(self.config.phase_duration_sec - elapsed) + 1)
            return f"PHASE 2: Keep eyes GENTLY CLOSED ({rem}s)"
        else:
            return "Active Monitoring"
