"""
Task 4: Multimodal Alert Fusion Engine.

Combines Head Pose droop, Gaze Distraction, Blink Micro-sleep, PERCLOS, and
Yawning into a cohesive, prioritized driver risk assessment.
Eliminates cross-talk and resolves alert hierarchies.
"""

from __future__ import annotations

from typing import Tuple

try:
    from .configuration import (
        DistractionOutput,
        DriverRiskLevel,
        EarMarOutput,
        GazeStatus,
        PitchLevel,
        PoseOutput,
        VisualizerConfig,
    )
except ImportError:
    from configuration import (
        DistractionOutput,
        DriverRiskLevel,
        EarMarOutput,
        GazeStatus,
        PitchLevel,
        PoseOutput,
        VisualizerConfig,
    )


class MultimodalAlertEngine:
    """Arbitrates and prioritizes multi-stream telemetry signals."""

    def __init__(self, visualizer_config: VisualizerConfig = VisualizerConfig()):
        self.cfg = visualizer_config

    def evaluate(
        self,
        pose: PoseOutput,
        distraction: DistractionOutput,
        ear_mar: EarMarOutput,
    ) -> Tuple[DriverRiskLevel, str, Tuple[int, int, int]]:
        # 1. TIER 1: CRITICAL RISK (Emergency Intervention)
        # Condition A: Active micro-sleep (eyes held shut >= 500ms)
        if ear_mar.is_microsleep:
            return (
                DriverRiskLevel.CRITICAL_RISK,
                "CRITICAL: Micro-sleep detected! Eyes closed >500ms!",
                self.cfg.color_critical,
            )

        # Condition B: Simultaneous nodding droop and off-road gaze
        if pose.is_drooping and distraction.is_distracted:
            return (
                DriverRiskLevel.CRITICAL_RISK,
                "CRITICAL: Driver nodding & looking away from road!",
                self.cfg.color_critical,
            )

        # Condition C: Simultaneous nodding droop and elevated PERCLOS
        if pose.is_drooping and ear_mar.is_perclos_alarm:
            return (
                DriverRiskLevel.CRITICAL_RISK,
                "CRITICAL: Severe somnolence! Droop + high PERCLOS!",
                self.cfg.color_critical,
            )

        # 2. TIER 2: DROWSINESS WARNING
        # Sustained PERCLOS accumulation >= 15%
        if ear_mar.is_perclos_alarm:
            return (
                DriverRiskLevel.DROWSY,
                f"WARNING: High PERCLOS ({ear_mar.perclos_pct:.1f}% >= 15%) - Drowsiness!",
                self.cfg.color_drowsy,
            )

        # Pitch head droop (unless user is actively scanning mirrors in distraction)
        if pose.is_drooping:
            return (
                DriverRiskLevel.DROWSY,
                f"WARNING: Head droop detected (Pitch {pose.pitch_deg:.1f} deg)!",
                self.cfg.color_drowsy,
            )

        # Prolonged yawning
        if ear_mar.is_yawn_alarm:
            return (
                DriverRiskLevel.DROWSY,
                "ALERT: Prolonged yawning detected (>4s) - Fatigue warning!",
                self.cfg.color_drowsy,
            )

        # 3. TIER 3: DISTRACTION ATTENTION
        if distraction.is_distracted:
            return (
                DriverRiskLevel.DISTRACTED,
                f"ATTENTION: Gaze off road ({distraction.confidence:.1f}%)!",
                self.cfg.color_distracted,
            )

        # 4. TIER 4: SAFE / NORMAL
        return (
            DriverRiskLevel.SAFE,
            "Driver attentive and looking at road",
            self.cfg.color_safe,
        )
