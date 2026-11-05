"""
Task 3: Facial Dynamics Estimation via EAR & MAR.

Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) with cascaded
adaptive filtering, micro-sleep detection (>500ms), 30s PERCLOS, and yawning.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Optional, Tuple
import numpy as np

try:
    from .configuration import EarMarConfig, EarMarOutput
except ImportError:
    from configuration import EarMarConfig, EarMarOutput


def calc_ear(pts_eye: np.ndarray) -> float:
    """Computes Eye Aspect Ratio from 6 eye landmarks."""
    a = float(np.linalg.norm(pts_eye[1] - pts_eye[5]))
    b = float(np.linalg.norm(pts_eye[2] - pts_eye[4]))
    c = float(np.linalg.norm(pts_eye[0] - pts_eye[3]))
    return (a + b) / (2.0 * c) if c > 1e-6 else 0.0


def calc_mar(pts_mouth: np.ndarray) -> float:
    """Computes Mouth Aspect Ratio from inner lips (indices 12:20 within slice 48:68)."""
    a = float(np.linalg.norm(pts_mouth[13] - pts_mouth[19]))
    b = float(np.linalg.norm(pts_mouth[14] - pts_mouth[18]))
    c = float(np.linalg.norm(pts_mouth[15] - pts_mouth[17]))
    d = float(np.linalg.norm(pts_mouth[12] - pts_mouth[16]))
    return (a + b + c) / (2.0 * d) if d > 1e-6 else 0.0


class EarMarSubsystem:
    """Manages adaptive filtering, blink duration, PERCLOS, and yawning tracking."""

    def __init__(self, config: Optional[EarMarConfig] = None):
        self.config = config or EarMarConfig()

        # Cascaded Adaptive Filter Buffers
        self._buf_raw: deque[float] = deque(maxlen=self.config.median_window)
        self._buf_median: deque[float] = deque(maxlen=self.config.mov_avg_window)

        # Baseline & Dynamic Thresholds
        self._ear_open: float = 0.30
        self._ear_closed: float = 0.15
        self._thresh_descent: float = 0.25
        self._thresh_closure: float = 0.18
        self._floor_closure: float = 0.18

        # Blink & Micro-sleep state
        self.in_blink_cycle: bool = False
        self.reached_effective_closure: bool = False
        self._t_start_closure: float = 0.0
        self._t_last_closed: float = 0.0
        self._t_last_microsleep: float = 0.0
        self.last_blink_ms: float = 0.0
        self.is_microsleep: bool = False

        # PERCLOS state
        self._perclos_buffer: deque[Tuple[float, int]] = deque()
        self._monitoring_start_time: float = 0.0
        self.perclos_pct: float = 0.0
        self.is_perclos_alarm: bool = False

        # Yawn state
        self.in_yawn: bool = False
        self._t_start_yawn: float = 0.0
        self._t_last_yawn: float = 0.0
        self.is_yawn_alarm: bool = False

    def initialize_baselines(
        self,
        ear_open: float,
        ear_closed: float,
        thresh_descent: float,
        thresh_closure: float,
    ) -> None:
        """Initializes thresholds from the calibration sequence."""
        self._ear_open = ear_open
        self._ear_closed = ear_closed
        self._thresh_descent = thresh_descent
        self._thresh_closure = thresh_closure
        self._floor_closure = thresh_closure

        self._buf_raw.clear()
        self._buf_median.clear()
        for _ in range(self._buf_raw.maxlen):
            self._buf_raw.append(ear_open)
        for _ in range(self._buf_median.maxlen):
            self._buf_median.append(ear_open)

    def start_monitoring(self, t: float) -> None:
        self._monitoring_start_time = t
        self._perclos_buffer.clear()

    def process(
        self,
        pts: np.ndarray,
        is_active: bool,
        current_time: Optional[float] = None,
    ) -> EarMarOutput:
        t = current_time if current_time is not None else time.time()

        eye_r = pts[36:42]
        eye_l = pts[42:48]
        mouth = pts[48:68]

        ear_raw = (calc_ear(eye_r) + calc_ear(eye_l)) / 2.0
        mar_raw = calc_mar(mouth)

        if not is_active:
            return EarMarOutput(
                ear_raw=ear_raw,
                ear_filtered=ear_raw,
                mar_raw=mar_raw,
                thresh_descent=self._thresh_descent,
                thresh_closure=self._thresh_closure,
                last_blink_ms=0.0,
                is_microsleep=False,
                perclos_pct=0.0,
                is_perclos_alarm=False,
                is_perclos_warmed_up=False,
                is_yawning=False,
                is_yawn_alarm=False,
            )

        # 1. Cascaded Adaptive Filter
        if not self.in_blink_cycle and ear_raw > self._thresh_descent:
            self._buf_raw.append(ear_raw)
            if len(self._buf_raw) == self._buf_raw.maxlen:
                med = float(np.median(self._buf_raw))
                self._buf_median.append(med)
                if len(self._buf_median) >= self.config.min_samples_for_update:
                    self._ear_open = float(np.mean(self._buf_median))
                    dyn_range = max(self.config.min_dynamic_range, self._ear_open - self._ear_closed)
                    self._thresh_closure = self._ear_closed + (self.config.sensibilidad_cierre * dyn_range)
                    self._thresh_descent = self._ear_closed + (self.config.factor_descenso * dyn_range)

        if self._thresh_closure < self._floor_closure:
            self._thresh_closure = self._floor_closure

        ear_med = float(np.median(self._buf_raw)) if self._buf_raw else ear_raw

        # 2. Blink Duration & Micro-sleep
        if ear_raw < self._thresh_descent:
            if not self.in_blink_cycle:
                self.in_blink_cycle = True
                self.reached_effective_closure = False
                self._t_start_closure = 0.0

            if self.in_blink_cycle:
                if ear_raw <= self._thresh_closure:
                    if not self.reached_effective_closure:
                        self.reached_effective_closure = True
                        self._t_start_closure = t

                    self._t_last_closed = t
                    if (t - self._t_start_closure) >= self.config.blink_max_normal_sec:
                        self.is_microsleep = True
                        self._t_last_microsleep = t
        else:
            if self.in_blink_cycle:
                if self.reached_effective_closure:
                    dur_s = self._t_last_closed - self._t_start_closure
                    self.last_blink_ms = max(dur_s * 1000.0, 0.0)
                    if dur_s >= self.config.blink_max_normal_sec:
                        self.is_microsleep = True
                        self._t_last_microsleep = t

                self.in_blink_cycle = False
                self.reached_effective_closure = False

            if (t - self._t_last_microsleep) > 1.2:
                self.is_microsleep = False

        # 3. PERCLOS (30s window)
        is_closed = 1 if ear_raw <= self._thresh_closure else 0
        self._perclos_buffer.append((t, is_closed))
        cutoff = t - self.config.perclos_window_sec
        while self._perclos_buffer and self._perclos_buffer[0][0] < cutoff:
            self._perclos_buffer.popleft()

        elapsed = t - self._monitoring_start_time
        is_warmed_up = elapsed >= self.config.perclos_window_sec

        if is_warmed_up and len(self._perclos_buffer) > 0:
            tot_c = sum(s[1] for s in self._perclos_buffer)
            self.perclos_pct = (tot_c / len(self._perclos_buffer)) * 100.0
            self.is_perclos_alarm = self.perclos_pct >= self.config.perclos_threshold_pct
        else:
            self.perclos_pct = 0.0
            self.is_perclos_alarm = False

        # 4. Yawning
        if mar_raw > self.config.mar_threshold:
            if not self.in_yawn:
                self.in_yawn = True
                self._t_start_yawn = t

            if (t - self._t_start_yawn) >= self.config.yawn_sustained_sec:
                self.is_yawn_alarm = True
                self._t_last_yawn = t
        else:
            if self.in_yawn:
                if (t - self._t_start_yawn) >= self.config.yawn_sustained_sec:
                    self.is_yawn_alarm = True
                    self._t_last_yawn = t
                self.in_yawn = False

            if (t - self._t_last_yawn) > 1.5:
                self.is_yawn_alarm = False

        return EarMarOutput(
            ear_raw=ear_raw,
            ear_filtered=ear_med,
            mar_raw=mar_raw,
            thresh_descent=self._thresh_descent,
            thresh_closure=self._thresh_closure,
            last_blink_ms=self.last_blink_ms,
            is_microsleep=self.is_microsleep,
            perclos_pct=self.perclos_pct,
            is_perclos_alarm=self.is_perclos_alarm,
            is_perclos_warmed_up=is_warmed_up,
            is_yawning=self.in_yawn,
            is_yawn_alarm=self.is_yawn_alarm,
        )
