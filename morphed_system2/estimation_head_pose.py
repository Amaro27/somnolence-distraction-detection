"""
Task 1: Head Pose Estimation & Pitch Droop Analysis.

Computes 2D facial coordinate frame and scale-, roll-, and yaw-invariant
vertical ratio (rho) to classify pitch droop and lateral tilt.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    from .configuration import HeadPoseConfig, PitchLevel, PoseOutput
except ImportError:
    from configuration import HeadPoseConfig, PitchLevel, PoseOutput


class HeadPoseEstimator:
    """Calculates instantaneous pitch and roll angles from 2D landmarks."""

    def __init__(self, config: Optional[HeadPoseConfig] = None):
        self.config = config or HeadPoseConfig()
        self.pitch_counter: int = 0

    def reset(self) -> None:
        self.pitch_counter = 0

    def build_face_frame(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        # Indices: 36 (right eye outer), 45 (left eye outer)
        p_right = pts[36].astype(np.float64)
        p_left = pts[45].astype(np.float64)

        e_mid = (p_right + p_left) / 2.0
        v_eyes = p_left - p_right
        w_eyes = float(np.linalg.norm(v_eyes))

        if w_eyes < 1e-6:
            u_x = np.array([1.0, 0.0])
            u_y = np.array([0.0, 1.0])
            roll_deg = 0.0
        else:
            u_x = v_eyes / w_eyes
            u_y = np.array([-u_x[1], u_x[0]])  # Perpendicular pointing downwards
            roll_deg = float(-np.degrees(np.arctan2(v_eyes[1], v_eyes[0])))

        return e_mid, u_x, u_y, w_eyes, roll_deg

    def estimate_rho_and_roll(self, pts: np.ndarray) -> Tuple[float, float]:
        """Calculates instantaneous rho vertical ratio and roll angle."""
        e_mid, _, u_y, _, roll_deg = self.build_face_frame(pts)

        p_nose = pts[33].astype(np.float64)
        p_chin = pts[8].astype(np.float64)

        d_top = float(np.dot(p_nose - e_mid, u_y))
        d_bottom = float(np.dot(p_chin - p_nose, u_y))
        h_face = max(d_top + d_bottom, 1e-6)

        rho = d_top / h_face
        return rho, roll_deg

    def evaluate_pose(
        self,
        pts: np.ndarray,
        baseline_rho: float,
        baseline_roll: float,
    ) -> PoseOutput:
        """Evaluates pitch droop relative to calibrated baseline with temporal debounce."""
        rho, roll_deg = self.estimate_rho_and_roll(pts)

        delta_rho = rho - baseline_rho
        pitch_deg = float(np.degrees(np.arctan2(-delta_rho, self.config.kp_pitch)))

        # Debounced Pitch droop classification
        cfg = self.config
        if pitch_deg > cfg.pitch_thresh_mild:
            self.pitch_counter = 0
            pitch_level = PitchLevel.NORMAL
        elif pitch_deg > cfg.pitch_thresh_warning:
            self.pitch_counter += 1
            pitch_level = PitchLevel.MILD_DROOP if self.pitch_counter > cfg.sustained_frames else PitchLevel.NORMAL
        else:
            self.pitch_counter += 1
            pitch_level = PitchLevel.FATIGUE_WARNING if self.pitch_counter > cfg.sustained_frames else PitchLevel.MILD_DROOP

        is_drooping = (pitch_level == PitchLevel.FATIGUE_WARNING)
        return PoseOutput(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            pitch_level=pitch_level,
            is_drooping=is_drooping,
        )
