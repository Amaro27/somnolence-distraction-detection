"""
Visualization and presentation component for EAR/MAR somnolence detection.

Renders high-contrast HUD overlays, facial bounding box, landmark contours,
calibration prompts, and categorized alert banners (Micro-sleep, PERCLOS, Yawning).
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    from .configuration import (
        AlertSeverity,
        CalibrationPhase,
        EarMarStateAssessment,
        VisualizerConfig,
    )
except ImportError:
    from configuration import (
        AlertSeverity,
        CalibrationPhase,
        EarMarStateAssessment,
        VisualizerConfig,
    )


class EarMarVisualizer:
    """Decoupled OpenCV presentation renderer for fatigue and drowsiness HUD."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def draw_hud(
        self,
        frame: np.ndarray,
        assessment: EarMarStateAssessment,
        guidance_text: str,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cfg = self.config
        h, w = frame.shape[:2]

        # 1. Calibration Guidance Banner
        if assessment.phase != CalibrationPhase.ACTIVE:
            self._draw_calibration_hud(frame, assessment, guidance_text)
            return frame

        # 2. Face Bounding Box & Landmark highlights
        color_box = cfg.color_normal
        if assessment.severity == AlertSeverity.MICROSLEEP_CRITICAL:
            color_box = cfg.color_critical
        elif assessment.severity in (AlertSeverity.PERCLOS_WARNING, AlertSeverity.YAWN_ALERT):
            color_box = cfg.color_warning

        if face_bbox is not None:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color_box, 2)

            # Draw alert labels above face
            y_offset = y - 10
            if assessment.blink.is_microsleep_alarm:
                cv2.putText(frame, "ALARMA: OJOS CERRADOS (>500ms)", (x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, cfg.color_critical, 2, cv2.LINE_AA)
                y_offset -= 26

            if assessment.yawn.is_yawn_alarm:
                cv2.putText(frame, "ALERTA: BOSTEZO PROLONGADO (>4s)", (x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, cfg.color_warning, 2, cv2.LINE_AA)
                y_offset -= 26

            if assessment.perclos.is_perclos_alarm:
                cv2.putText(frame, "ALERTA: SOMNOLENCIA (PERCLOS >= 15%)", (x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, cfg.color_critical, 3, cv2.LINE_AA)

        if landmarks is not None:
            # Draw eye contour points in cyan (36:48) and mouth points in yellow (48:68)
            for idx in range(36, 48):
                cv2.circle(frame, tuple(landmarks[idx]), 2, (255, 255, 0), -1)
            for idx in range(48, 68):
                cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 255), -1)

        # 3. Top Telemetry Dashboard Container
        cv2.rectangle(frame, (10, 10), (560, 130), cfg.hud_bg_color, -1)
        cv2.rectangle(frame, (10, 10), (560, 130), color_box, 2)

        # Line 1: EAR Metrics & Adaptive Thresholds
        m = assessment.metrics
        ear_text = f"EAR: {m.ear_raw:.2f} | T_Desc Adapt: {m.threshold_descent:.2f} | T_Cierre: {m.threshold_closure:.2f}"
        cv2.putText(frame, ear_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cfg.color_text, 1, cv2.LINE_AA)

        # Line 2: MAR & Blink Duration
        blink_txt = f"MAR: {m.mar_raw:.2f} | Ultimo Parpadeo: {assessment.blink.last_blink_duration_ms:.0f} ms"
        cv2.putText(frame, blink_txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cfg.color_accent, 1, cv2.LINE_AA)

        # Line 3: PERCLOS Accumulation
        p = assessment.perclos
        if not p.is_warmed_up:
            rem_s = int(p.warmup_remaining_sec)
            perclos_text = f"PERCLOS (30s): 0.0% [Calentamiento: {rem_s}s]"
            color_p = (180, 180, 180)
        else:
            color_p = cfg.color_critical if p.is_perclos_alarm else (255, 100, 255)
            perclos_text = f"PERCLOS (30s): {p.perclos_pct:.1f}% {'(ALERTA FATIGA)' if p.is_perclos_alarm else ''}"

        cv2.putText(frame, perclos_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_p, 2, cv2.LINE_AA)

        return frame

    def _draw_calibration_hud(
        self,
        frame: np.ndarray,
        assessment: EarMarStateAssessment,
        guidance_text: str,
    ) -> None:
        cfg = self.config
        cv2.rectangle(frame, (10, 10), (600, 90), cfg.hud_bg_color, -1)
        cv2.rectangle(frame, (10, 10), (600, 90), (0, 200, 255), 2)

        title = "CALIBRACION FISIOLOGICA EN 2 FASES"
        cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, cfg.color_accent, 2, cv2.LINE_AA)
        cv2.putText(frame, guidance_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cfg.color_text, 2, cv2.LINE_AA)
