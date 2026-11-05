"""
Physiological fatigue analyzers for Eye and Mouth Aspect Ratios.

Includes:
- BlinkDurationTracker: Depth-validated blink tracking and micro-sleep detection (>500ms).
- RollingPerclosTracker: 30-second rolling window cumulative eye closure calculation (>=15%).
- YawnDurationTracker: Sustained mouth opening detection (>4s).
- EarMarAnalyzer: Composite orchestrator producing unified frame assessment.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Optional, Tuple

try:
    from .configuration import (
        AlertSeverity,
        BlinkAssessment,
        BlinkConfig,
        CalibrationPhase,
        EarMarMetrics,
        EarMarStateAssessment,
        PerclosAssessment,
        PerclosConfig,
        YawnAssessment,
        YawnConfig,
    )
except ImportError:
    from configuration import (
        AlertSeverity,
        BlinkAssessment,
        BlinkConfig,
        CalibrationPhase,
        EarMarMetrics,
        EarMarStateAssessment,
        PerclosAssessment,
        PerclosConfig,
        YawnAssessment,
        YawnConfig,
    )


class BlinkDurationTracker:
    """Measures precise blink duration with depth-validation and catches micro-sleeps."""

    def __init__(self, config: Optional[BlinkConfig] = None):
        self.config = config or BlinkConfig()
        self.reset()

    def reset(self) -> None:
        self.in_cycle: bool = False
        self.reached_effective_closure: bool = False
        self._t_start_closure: float = 0.0
        self._t_last_closed: float = 0.0
        self._t_last_prolonged_event: float = 0.0
        self.last_blink_duration_ms: float = 0.0
        self.is_alarm: bool = False

    def update(
        self,
        ear_raw: float,
        thresh_descent: float,
        thresh_closure: float,
        current_time: Optional[float] = None,
    ) -> BlinkAssessment:
        t = current_time if current_time is not None else time.time()

        if ear_raw < thresh_descent:
            # 1. Start blink window upon descending past 70% threshold
            if not self.in_cycle:
                self.in_cycle = True
                self.reached_effective_closure = False
                self._t_start_closure = 0.0
                self._t_last_closed = 0.0

            if self.in_cycle:
                # 2. Trigger exact duration timer when entering deep closure (<= 20% floor)
                if ear_raw <= thresh_closure:
                    if not self.reached_effective_closure:
                        self.reached_effective_closure = True
                        self._t_start_closure = t

                    self._t_last_closed = t

                    # Instant micro-sleep alarm if eyes remain held shut >= 500ms
                    held_duration = t - self._t_start_closure
                    if held_duration >= self.config.max_normal_duration_sec:
                        self.is_alarm = True
                        self._t_last_prolonged_event = t
        else:
            # 3. Eye aperture recovered above descent threshold; finalize event
            if self.in_cycle:
                if self.reached_effective_closure:
                    dur_sec = self._t_last_closed - self._t_start_closure
                    self.last_blink_duration_ms = max(dur_sec * 1000.0, 0.0)

                    if dur_sec >= self.config.max_normal_duration_sec:
                        self.is_alarm = True
                        self._t_last_prolonged_event = t

                self.in_cycle = False
                self.reached_effective_closure = False

            # Clear micro-sleep alarm banner after cooldown
            if (t - self._t_last_prolonged_event) > self.config.alarm_cooldown_sec:
                self.is_alarm = False

        return BlinkAssessment(
            in_blink_cycle=self.in_cycle,
            reached_effective_closure=self.reached_effective_closure,
            last_blink_duration_ms=self.last_blink_duration_ms,
            is_microsleep_alarm=self.is_alarm,
        )


class RollingPerclosTracker:
    """Calculates true time-based PERCLOS over a sliding 30-second window."""

    def __init__(self, config: Optional[PerclosConfig] = None):
        self.config = config or PerclosConfig()
        self.reset()

    def reset(self) -> None:
        self._time_buffer: deque[Tuple[float, int]] = deque()
        self._monitoring_start_time: float = 0.0
        self.perclos_pct: float = 0.0
        self.is_alarm: bool = False

    def start_monitoring(self, start_time: Optional[float] = None) -> None:
        self._monitoring_start_time = start_time if start_time is not None else time.time()
        self._time_buffer.clear()
        self.perclos_pct = 0.0
        self.is_alarm = False

    def update(
        self,
        ear_raw: float,
        thresh_closure: float,
        current_time: Optional[float] = None,
    ) -> PerclosAssessment:
        t = current_time if current_time is not None else time.time()
        if self._monitoring_start_time == 0.0:
            self._monitoring_start_time = t

        is_closed_binary = 1 if ear_raw <= thresh_closure else 0
        self._time_buffer.append((t, is_closed_binary))

        # Evict samples older than 30s
        cutoff = t - self.config.window_sec
        while self._time_buffer and self._time_buffer[0][0] < cutoff:
            self._time_buffer.popleft()

        elapsed_since_start = t - self._monitoring_start_time
        is_warmed_up = elapsed_since_start >= self.config.window_sec
        warmup_remaining = max(0.0, self.config.window_sec - elapsed_since_start)

        if is_warmed_up and len(self._time_buffer) > 0:
            total_closed = sum(sample[1] for sample in self._time_buffer)
            self.perclos_pct = (total_closed / len(self._time_buffer)) * 100.0
            self.is_alarm = self.perclos_pct >= self.config.threshold_fatigue_pct
        else:
            self.perclos_pct = 0.0
            self.is_alarm = False

        return PerclosAssessment(
            perclos_pct=self.perclos_pct,
            is_warmed_up=is_warmed_up,
            warmup_remaining_sec=warmup_remaining,
            is_perclos_alarm=self.is_alarm,
        )


class YawnDurationTracker:
    """Tracks sustained mouth aspect ratio to detect yawning events (>4s)."""

    def __init__(self, config: Optional[YawnConfig] = None):
        self.config = config or YawnConfig()
        self.reset()

    def reset(self) -> None:
        self.in_yawn: bool = False
        self._t_start_yawn: float = 0.0
        self._t_last_confirmed_yawn: float = 0.0
        self.is_alarm: bool = False

    def update(self, mar_raw: float, current_time: Optional[float] = None) -> YawnAssessment:
        t = current_time if current_time is not None else time.time()
        yawn_duration = 0.0

        if mar_raw > self.config.mar_threshold:
            if not self.in_yawn:
                self.in_yawn = True
                self._t_start_yawn = t

            yawn_duration = t - self._t_start_yawn
            if yawn_duration >= self.config.sustained_duration_sec:
                self.is_alarm = True
                self._t_last_confirmed_yawn = t
        else:
            if self.in_yawn:
                yawn_duration = t - self._t_start_yawn
                if yawn_duration >= self.config.sustained_duration_sec:
                    self.is_alarm = True
                    self._t_last_confirmed_yawn = t
                self.in_yawn = False

            if (t - self._t_last_confirmed_yawn) > self.config.alarm_cooldown_sec:
                self.is_alarm = False

        return YawnAssessment(
            is_yawning=self.in_yawn,
            current_yawn_duration_sec=yawn_duration if self.in_yawn else 0.0,
            is_yawn_alarm=self.is_alarm,
        )


class EarMarAnalyzer:
    """Composite coordinator combining Blink, PERCLOS, and Yawn state evaluation."""

    def __init__(
        self,
        blink_config: Optional[BlinkConfig] = None,
        perclos_config: Optional[PerclosConfig] = None,
        yawn_config: Optional[YawnConfig] = None,
    ):
        self.blink_tracker = BlinkDurationTracker(blink_config)
        self.perclos_tracker = RollingPerclosTracker(perclos_config)
        self.yawn_tracker = YawnDurationTracker(yawn_config)

    def reset(self) -> None:
        self.blink_tracker.reset()
        self.perclos_tracker.reset()
        self.yawn_tracker.reset()

    def start_monitoring(self, start_time: Optional[float] = None) -> None:
        self.perclos_tracker.start_monitoring(start_time)

    def analyze_frame(
        self,
        phase: CalibrationPhase,
        metrics: EarMarMetrics,
        current_time: Optional[float] = None,
        calib_progress: float = 1.0,
    ) -> EarMarStateAssessment:
        t = current_time if current_time is not None else time.time()

        if phase != CalibrationPhase.ACTIVE:
            return EarMarStateAssessment(
                phase=phase,
                metrics=metrics,
                blink=BlinkAssessment(False, False, 0.0, False),
                perclos=PerclosAssessment(0.0, False, 30.0, False),
                yawn=YawnAssessment(False, 0.0, False),
                severity=AlertSeverity.NORMAL,
                calib_progress=calib_progress,
                status_summary="Calibrating...",
            )

        blink_ass = self.blink_tracker.update(
            ear_raw=metrics.ear_raw,
            thresh_descent=metrics.threshold_descent,
            thresh_closure=metrics.threshold_closure,
            current_time=t,
        )

        perclos_ass = self.perclos_tracker.update(
            ear_raw=metrics.ear_raw,
            thresh_closure=metrics.threshold_closure,
            current_time=t,
        )

        yawn_ass = self.yawn_tracker.update(
            mar_raw=metrics.mar_raw,
            current_time=t,
        )

        # Severity Hierarchy: Micro-sleep > High PERCLOS > Yawn
        if blink_ass.is_microsleep_alarm:
            severity = AlertSeverity.MICROSLEEP_CRITICAL
            summary = "CRITICAL: Prolonged eye closure (>500ms)!"
        elif perclos_ass.is_perclos_alarm:
            severity = AlertSeverity.PERCLOS_WARNING
            summary = f"WARNING: Somnolence detected (PERCLOS {perclos_ass.perclos_pct:.1f}% >= 15%)"
        elif yawn_ass.is_yawn_alarm:
            severity = AlertSeverity.YAWN_ALERT
            summary = "ALERT: Prolonged yawn detected (>4s)"
        else:
            severity = AlertSeverity.NORMAL
            summary = "Driver attentive / Normal"

        return EarMarStateAssessment(
            phase=phase,
            metrics=metrics,
            blink=blink_ass,
            perclos=perclos_ass,
            yawn=yawn_ass,
            severity=severity,
            calib_progress=1.0,
            status_summary=summary,
        )
