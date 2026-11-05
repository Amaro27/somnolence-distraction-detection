"""
Unified Presentation and HUD Component for ADAS Dashboard.

Draws consolidated multi-metric telemetry, risk state banners, landmark contours,
face bounding box, and calibration guidance prompts.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    from .configuration import (
        CalibrationPhase,
        UnifiedDriverAssessment,
        VisualizerConfig,
    )
except ImportError:
    from configuration import (
        CalibrationPhase,
        UnifiedDriverAssessment,
        VisualizerConfig,
    )


class UnifiedADASVisualizer:
    """Decoupled OpenCV presentation renderer for the morphed driver monitor."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def render(self, frame: np.ndarray, state: UnifiedDriverAssessment) -> np.ndarray:
        cfg = self.config

        # 1. Calibration Phase HUD
        if state.phase != CalibrationPhase.ACTIVE:
            cv2.rectangle(frame, (10, 10), (620, 95), cfg.bg_container, -1)
            cv2.rectangle(frame, (10, 10), (620, 95), cfg.accent_color, 2)
            cv2.putText(frame, "ADAS SYSTEM CALIBRATION (Step 1-3)", (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, cfg.accent_color, 2, cv2.LINE_AA)
            cv2.putText(frame, state.guidance_prompt, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cfg.text_color, 2, cv2.LINE_AA)
            return frame

        color = state.primary_color

        # 2. Face Bounding Box & Facial Landmarks
        if state.face_bbox is not None:
            x, y, bw, bh = state.face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # Warning label above bounding box
            if state.risk_level != "SAFE":
                cv2.putText(frame, state.status_banner, (x, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if state.landmarks is not None:
            pts = state.landmarks
            # Draw eyes in cyan (36:48)
            for i in range(36, 48):
                cv2.circle(frame, tuple(pts[i]), 2, (255, 255, 0), -1)
            # Draw mouth in yellow (48:68)
            for i in range(48, 68):
                cv2.circle(frame, tuple(pts[i]), 2, (0, 255, 255), -1)
            # Draw nose and chin points
            for i in (33, 8):
                cv2.circle(frame, tuple(pts[i]), 3, (0, 200, 255), -1)

        # 3. Consolidated Top Dashboard Container
        cv2.rectangle(frame, (10, 10), (620, 165), cfg.bg_container, -1)
        cv2.rectangle(frame, (10, 10), (620, 165), color, 2)

        # Line 1: Telemetry
        backend_name = "C-LSTM" if getattr(state.distraction, "backend", "light_cnn") == "clstm" else "LightCNN"
        telemetry = f"ADAS System ({backend_name}) | FPS: {state.fps:.1f} | Gaze Latency: {state.distraction.latency_ms:.1f}ms"
        cv2.putText(frame, telemetry, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, cfg.accent_color, 1, cv2.LINE_AA)

        # Line 2: Primary Risk State Banner
        risk_text = f"STATE: {state.risk_level.value}"
        cv2.putText(frame, risk_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Line 3: Head Pose & Gaze Telemetry
        p = state.pose
        d = state.distraction
        pose_gaze_txt = f"Pose: Pitch {p.pitch_deg:+.1f} deg | Roll {p.roll_deg:+.1f} deg  ||  Gaze [{backend_name}]: {d.status.value} ({d.confidence:.1f}%)"
        cv2.putText(frame, pose_gaze_txt, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cfg.text_color, 1, cv2.LINE_AA)

        # Line 4: EAR / MAR / PERCLOS / Blink
        em = state.ear_mar
        perclos_str = f"PERCLOS: {em.perclos_pct:.1f}%" if em.is_perclos_warmed_up else "PERCLOS: [Warmup 30s]"
        ear_mar_txt = f"EAR: {em.ear_raw:.2f} | MAR: {em.mar_raw:.2f} | Blink: {em.last_blink_ms:.0f}ms | {perclos_str}"
        cv2.putText(frame, ear_mar_txt, (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

        return frame
