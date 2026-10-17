"""
Presentation and HUD rendering component for head pose estimation.

Draws facial landmark points, 2D facial axes, calibration progress bars,
live telemetry overlays, and fatigue warning banners using OpenCV.
Completely isolated from pose estimation math and classification logic.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from .configuration import (
        FatigueAssessment,
        LandmarkIndices,
        PitchLevel,
        PoseMetrics,
        VisualizerConfig,
    )
except ImportError:
    from configuration import (
        FatigueAssessment,
        LandmarkIndices,
        PitchLevel,
        PoseMetrics,
        VisualizerConfig,
    )


class PoseVisualizer:
    """Renders overlays, coordinate axes, HUD metrics, and warning banners."""

    def __init__(
        self,
        config: Optional[VisualizerConfig] = None,
        landmark_indices: Optional[LandmarkIndices] = None,
    ):
        self.config = config or VisualizerConfig()
        self.landmarks = landmark_indices or LandmarkIndices()

    def draw_landmarks(self, frame: np.ndarray, points: np.ndarray) -> None:
        """Draws key landmark dots on the face."""
        for idx in self.landmarks.key_indices:
            cv2.circle(frame, tuple(points[idx]), 3, self.config.landmark_color, -1)

    def draw_face_axes(
        self, frame: np.ndarray, metrics: PoseMetrics, pitch_level: PitchLevel
    ) -> None:
        """Draws perpendicular facial coordinate axes with dynamic color coding."""
        e_mid = metrics.e_mid.astype(int)
        u_x = metrics.u_x
        u_y = metrics.u_y
        w = int(metrics.w_eyes * 0.5)

        # Horizontal eye line axis (cyan/yellow)
        pt_x = (int(e_mid[0] + u_x[0] * w), int(e_mid[1] + u_x[1] * w))
        cv2.arrowedLine(frame, tuple(e_mid), pt_x, self.config.axis_x_color, 2, tipLength=0.2)

        # Vertical midline axis (color-coded by pitch severity)
        vert_color = self.config.level_colors.get(pitch_level, (0, 255, 0))
        pt_y = (int(e_mid[0] + u_y[0] * w * 1.5), int(e_mid[1] + u_y[1] * w * 1.5))
        cv2.arrowedLine(frame, tuple(e_mid), pt_y, vert_color, 2, tipLength=0.2)

    def draw_calibration_overlay(self, frame: np.ndarray, progress: float) -> None:
        """Renders calibration progress bar."""
        bar_x, bar_y, bar_w, bar_h = 20, 25, 180, 14
        cv2.putText(
            frame,
            f"Calibrating: {int(progress * 100)}%",
            (bar_x, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(
            frame,
            (bar_x + 1, bar_y + 1),
            (bar_x + fill_w, bar_y + bar_h - 1),
            self.config.progress_bar_color,
            -1,
        )

    def draw_hud(
        self, frame: np.ndarray, metrics: PoseMetrics, assessment: FatigueAssessment
    ) -> None:
        """Renders live telemetry readout and alert banner when fatigue is detected."""
        h, w = frame.shape[:2]
        pitch_color = self.config.level_colors.get(assessment.pitch_level, (255, 255, 255))
        roll_color = self.config.level_colors.get(assessment.roll_level, (255, 255, 255))

        cv2.putText(
            frame,
            f"Pitch: {metrics.pitch_deg:+5.1f} deg  [{assessment.pitch_level}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pitch_color,
            2,
        )
        cv2.putText(
            frame,
            f"Roll : {metrics.roll_deg:+5.1f} deg  [{assessment.roll_level}]",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            roll_color,
            2,
        )

        if assessment.is_warning:
            cv2.rectangle(frame, (0, 0), (w, 36), self.config.warning_banner_bg, -1)
            cv2.putText(
                frame,
                "FATIGUE DETECTED: HEAD DROOPING!",
                (w // 2 - 190, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
            )
