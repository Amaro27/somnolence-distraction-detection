"""
Unified Calibration Component for Multimodal Driver Monitoring.

Executes a guided, sequential calibration protocol:
1. Neutral Head Pose: Driver looks straight forward at the road (~2s)
2. Open Eyes: Measures relaxed open-eye EAR baseline (~3s)
3. Closed Eyes: Measures closed-eye EAR floor (~3s)
4. Exports personalized BaselineValues to all analytical subsystems.
"""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

try:
    from .configuration import (
        BaselineValues,
        CalibrationPhase,
        EarMarConfig,
        HeadPoseConfig,
    )
except ImportError:
    from configuration import (
        BaselineValues,
        CalibrationPhase,
        EarMarConfig,
        HeadPoseConfig,
    )


class UnifiedSystemCalibrator:
    """Coordinates multi-phase baseline calibration for Head Pose and EAR."""

    def __init__(
        self,
        pose_cfg: Optional[HeadPoseConfig] = None,
        ear_cfg: Optional[EarMarConfig] = None,
    ):
        self.pose_cfg = pose_cfg or HeadPoseConfig()
        self.ear_cfg = ear_cfg or EarMarConfig()
        self.reset()

    def reset(self) -> None:
        self.phase: CalibrationPhase = CalibrationPhase.IDLE
        self._phase_start_time: float = 0.0

        self._rho_buffer: List[float] = []
        self._roll_buffer: List[float] = []
        self._ear_open_buffer: List[float] = []
        self._ear_closed_buffer: List[float] = []

        self.baseline: Optional[BaselineValues] = None

    @property
    def is_calibrated(self) -> bool:
        return self.baseline is not None

    def start_calibration(self, t: Optional[float] = None) -> None:
        """Begins Step 1 of the calibration sequence."""
        now = t if t is not None else time.time()
        self.phase = CalibrationPhase.CALIB_HEAD_POSE
        self._phase_start_time = now
        self._rho_buffer.clear()
        self._roll_buffer.clear()
        self._ear_open_buffer.clear()
        self._ear_closed_buffer.clear()
        self.baseline = None
        print("[INFO] Starting Calibration Step 1: Look straight ahead (Neutral Pose).")

    def update(
        self,
        rho: float,
        roll: float,
        ear_raw: float,
        t: Optional[float] = None,
    ) -> CalibrationPhase:
        if self.phase in (CalibrationPhase.IDLE, CalibrationPhase.ACTIVE):
            return self.phase

        now = t if t is not None else time.time()
        elapsed = now - self._phase_start_time

        # Step 1: Neutral Head Pose
        if self.phase == CalibrationPhase.CALIB_HEAD_POSE:
            self._rho_buffer.append(rho)
            self._roll_buffer.append(roll)
            if len(self._rho_buffer) >= self.pose_cfg.required_calib_frames:
                self.phase = CalibrationPhase.CALIB_OPEN_EYES
                self._phase_start_time = now
                print("[INFO] Step 2: Keep eyes relaxed and OPEN looking at the camera.")

        # Step 2: Open Eyes
        elif self.phase == CalibrationPhase.CALIB_OPEN_EYES:
            if ear_raw > 0.0:
                self._ear_open_buffer.append(ear_raw)
            if elapsed >= self.ear_cfg.calib_step_sec:
                self.phase = CalibrationPhase.TRANSITION_EYES
                self._phase_start_time = now
                print("[INFO] Prepare to close eyes...")

        # Transition prompt
        elif self.phase == CalibrationPhase.TRANSITION_EYES:
            if elapsed >= 1.5:
                self.phase = CalibrationPhase.CALIB_CLOSED_EYES
                self._phase_start_time = now
                print("[INFO] Step 3: Gently CLOSE eyes and hold.")

        # Step 3: Closed Eyes
        elif self.phase == CalibrationPhase.CALIB_CLOSED_EYES:
            if ear_raw > 0.0:
                self._ear_closed_buffer.append(ear_raw)
            if elapsed >= self.ear_cfg.calib_step_sec:
                self.baseline = self._derive_baseline()
                self.phase = CalibrationPhase.ACTIVE
                print(
                    f"[INFO] Full Calibration Complete!\n"
                    f"       Pose Baseline: rho={self.baseline.rho:.3f}, roll={self.baseline.roll:.1f} deg\n"
                    f"       Eye Baselines: open={self.baseline.ear_open:.3f}, closed={self.baseline.ear_closed:.3f}\n"
                    f"       Thresholds:    descent={self.baseline.thresh_descent:.3f}, closure={self.baseline.thresh_closure:.3f}"
                )

        return self.phase

    def _derive_baseline(self) -> BaselineValues:
        rho_med = float(np.median(self._rho_buffer)) if self._rho_buffer else 0.40
        roll_med = float(np.median(self._roll_buffer)) if self._roll_buffer else 0.0

        ear_open = float(np.mean(self._ear_open_buffer)) if self._ear_open_buffer else 0.30
        ear_closed = float(np.mean(self._ear_closed_buffer)) if self._ear_closed_buffer else 0.15

        dyn_range = max(self.ear_cfg.min_dynamic_range, ear_open - ear_closed)
        t_descent = ear_closed + (self.ear_cfg.factor_descenso * dyn_range)
        t_closure = ear_closed + (self.ear_cfg.sensibilidad_cierre * dyn_range)

        return BaselineValues(
            rho=rho_med,
            roll=roll_med,
            ear_open=ear_open,
            ear_closed=ear_closed,
            thresh_descent=t_descent,
            thresh_closure=t_closure,
        )

    def get_guidance_text(self, t: Optional[float] = None) -> str:
        now = t if t is not None else time.time()
        elapsed = now - self._phase_start_time

        if self.phase == CalibrationPhase.IDLE:
            return "Press 'c' to initiate 3-step ADAS calibration"
        elif self.phase == CalibrationPhase.CALIB_HEAD_POSE:
            pct = int(min(len(self._rho_buffer) / max(self.pose_cfg.required_calib_frames, 1), 1.0) * 100)
            return f"STEP 1: Keep head steady looking at road ({pct}%)"
        elif self.phase == CalibrationPhase.CALIB_OPEN_EYES:
            rem = max(0, int(self.ear_cfg.calib_step_sec - elapsed) + 1)
            return f"STEP 2: Keep eyes OPEN looking forward ({rem}s)"
        elif self.phase == CalibrationPhase.TRANSITION_EYES:
            return "PREPARE: Close eyes now..."
        elif self.phase == CalibrationPhase.CALIB_CLOSED_EYES:
            rem = max(0, int(self.ear_cfg.calib_step_sec - elapsed) + 1)
            return f"STEP 3: Keep eyes GENTLY CLOSED ({rem}s)"
        return "Active Monitoring"
