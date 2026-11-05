"""
Cascaded adaptive filtering component for Eye Aspect Ratio (EAR).

Smooths instantaneous signal noise using a Median Filter and tracks dynamic
baseline drift via a Moving Average, with safe guards preventing blink
artifacts from distorting the physiological thresholds.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple
import numpy as np

try:
    from .configuration import (
        AdaptiveFilterConfig,
        CalibrationConfig,
        CalibratedThresholds,
    )
except ImportError:
    from configuration import (
        AdaptiveFilterConfig,
        CalibrationConfig,
        CalibratedThresholds,
    )


class CascadedAdaptiveFilter:
    """Combines a fast Median Filter with a slow Moving Average baseline updater."""

    def __init__(
        self,
        filter_config: Optional[AdaptiveFilterConfig] = None,
        calib_config: Optional[CalibrationConfig] = None,
    ):
        self.cfg_filter = filter_config or AdaptiveFilterConfig()
        self.cfg_calib = calib_config or CalibrationConfig()

        self._buffer_raw: deque[float] = deque(maxlen=self.cfg_filter.median_window)
        self._buffer_median: deque[float] = deque(maxlen=self.cfg_filter.mov_avg_window)

        self._current_ear_open: float = 0.30
        self._current_ear_closed: float = 0.15
        self._thresh_descent: float = 0.25
        self._thresh_closure: float = 0.18
        self._floor_closure: float = 0.18

    def initialize_from_calibration(self, thresholds: CalibratedThresholds) -> None:
        """Pre-fills filter buffers with calibrated baseline values to avoid cold-start lag."""
        self._current_ear_open = thresholds.ear_base_open
        self._current_ear_closed = thresholds.ear_base_closed
        self._thresh_descent = thresholds.threshold_descent
        self._thresh_closure = thresholds.threshold_closure
        self._floor_closure = thresholds.threshold_closure

        self._buffer_raw.clear()
        self._buffer_median.clear()

        for _ in range(self._buffer_raw.maxlen):
            self._buffer_raw.append(thresholds.ear_base_open)
        for _ in range(self._buffer_median.maxlen):
            self._buffer_median.append(thresholds.ear_base_open)

    def filter_and_adapt(
        self,
        ear_raw: float,
        in_blink_cycle: bool,
    ) -> Tuple[float, float, float, float]:
        """
        Processes single-frame EAR, updates median & moving average, and adapts thresholds.

        Parameters
        ----------
        ear_raw : float
            Instantaneous raw EAR from 6 landmarks.
        in_blink_cycle : bool
            Guard flag; if True or if ear_raw is below descent threshold, dynamic updates freeze.

        Returns
        -------
        Tuple[float, float, float, float]
            (ear_median, ear_mov_avg, adaptive_threshold_descent, adaptive_threshold_closure)
        """
        # Guard: Only adapt if eyes are demonstrably open and not entering a blink event
        if not in_blink_cycle and ear_raw > self._thresh_descent:
            self._buffer_raw.append(ear_raw)

            if len(self._buffer_raw) == self._buffer_raw.maxlen:
                ear_med = float(np.median(self._buffer_raw))
                self._buffer_median.append(ear_med)

                if len(self._buffer_median) >= self.cfg_filter.min_samples_for_update:
                    self._current_ear_open = float(np.mean(self._buffer_median))
                    dynamic_range = max(
                        self.cfg_calib.min_dynamic_range,
                        self._current_ear_open - self._current_ear_closed,
                    )

                    # Update thresholds relative to adapted open baseline
                    self._thresh_closure = self._current_ear_closed + (
                        self.cfg_calib.sensibilidad_cierre * dynamic_range
                    )
                    self._thresh_descent = self._current_ear_closed + (
                        self.cfg_calib.factor_descenso * dynamic_range
                    )

        # Safety floor: Adaptive closure threshold cannot fall below the calibrated static floor
        if self._thresh_closure < self._floor_closure:
            self._thresh_closure = self._floor_closure

        val_median = float(np.median(self._buffer_raw)) if self._buffer_raw else ear_raw
        val_mov_avg = float(np.mean(self._buffer_median)) if self._buffer_median else ear_raw

        return val_median, val_mov_avg, self._thresh_descent, self._thresh_closure
