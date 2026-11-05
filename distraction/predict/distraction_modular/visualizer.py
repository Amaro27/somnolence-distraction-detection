"""
Visualization and presentation component for driver distraction detection.

Renders high-contrast HUD overlays, alert banners, latency metrics, and
temporal buffer progress without coupling to inference models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from .configuration import (
        DistractionAssessment,
        DistractionStatus,
        VisualizerConfig,
    )
except ImportError:
    from configuration import (
        DistractionAssessment,
        DistractionStatus,
        VisualizerConfig,
    )


class DistractionVisualizer:
    """Decoupled OpenCV presentation renderer for driver distraction HUD."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def draw_hud(
        self,
        frame: np.ndarray,
        assessment: DistractionAssessment,
        fps: float,
        model_name: str = "Model",
    ) -> np.ndarray:
        """Overlays telemetry, status banner, latency, and temporal smoothing stats."""
        cfg = self.config
        color = cfg.status_colors.get(assessment.status, (0, 255, 0))

        # Top banner background box
        cv2.rectangle(frame, (10, 10), (520, 150), cfg.hud_bg_color, -1)
        cv2.rectangle(frame, (10, 10), (520, 150), color, 2)

        # 1. Performance telemetry (FPS & Latency)
        telemetry_text = f"{model_name} | FPS: {fps:.1f} | Latency: {assessment.inference_latency_ms:.1f} ms"
        cv2.putText(
            frame,
            telemetry_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cfg.accent_color,
            1,
            cv2.LINE_AA,
        )

        # 2. Main Alert / Status banner
        status_text = f"STATE: {assessment.status.value}"
        cv2.putText(
            frame,
            status_text,
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.font_scale_main,
            color,
            cfg.thickness,
            cv2.LINE_AA,
        )

        # 3. Temporal Buffer & Gaze Direction
        buffer_info = f"Buffer ({assessment.buffer_length}/{assessment.buffer_capacity}f) | {assessment.smoothed_label.upper()} ({assessment.confidence:.1f}%)"
        cv2.putText(
            frame,
            buffer_info,
            (20, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            cfg.text_color,
            1,
            cv2.LINE_AA,
        )

        # 4. Instantaneous raw prediction if available
        raw_info = f"Raw Frame: {assessment.raw_label} | Distracted: {assessment.is_distracted}"
        cv2.putText(
            frame,
            raw_info,
            (20, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return frame
