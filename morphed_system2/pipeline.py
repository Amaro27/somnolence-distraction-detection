"""
Single-Pass Runtime Orchestrator for Complete Morphed ADAS System.

Coordinates camera capture and executes landmark extraction EXACTLY ONCE per frame,
distributing features to Head Pose, EAR/MAR, and LightGazeCNN without redundant computation.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple, Union
import cv2
import dlib
import numpy as np
import torch

try:
    from .calibration import UnifiedSystemCalibrator
    from .configuration import (
        BaselineValues,
        CalibrationPhase,
        DistractionConfig,
        DistractionOutput,
        DriverRiskLevel,
        EarMarConfig,
        EarMarOutput,
        GazeStatus,
        HeadPoseConfig,
        PitchLevel,
        PoseOutput,
        UnifiedDriverAssessment,
        VisualizerConfig,
    )
    from .estimation_ear_mar import EarMarSubsystem
    from .estimation_head_pose import HeadPoseEstimator
    from .fusion import MultimodalAlertEngine
    from .models_distraction import DistractionInferenceEngine
    from .realtime_benchmark import LiveSessionBenchmarkTracker
    from .visualizer import UnifiedADASVisualizer
except ImportError:
    from calibration import UnifiedSystemCalibrator
    from configuration import (
        BaselineValues,
        CalibrationPhase,
        DistractionConfig,
        DistractionOutput,
        DriverRiskLevel,
        EarMarConfig,
        EarMarOutput,
        GazeStatus,
        HeadPoseConfig,
        PitchLevel,
        PoseOutput,
        UnifiedDriverAssessment,
        VisualizerConfig,
    )
    from estimation_ear_mar import EarMarSubsystem
    from estimation_head_pose import HeadPoseEstimator
    from fusion import MultimodalAlertEngine
    from models_distraction import DistractionInferenceEngine
    from realtime_benchmark import LiveSessionBenchmarkTracker
    from visualizer import UnifiedADASVisualizer


class UnifiedADASPipeline:
    """Consolidated ADAS pipeline combining Head Pose, Distraction, and EAR/MAR."""

    DEFAULT_PREDICTOR_PATHS = (
        "shape_predictor_68_face_landmarks.dat",
        os.path.join(os.path.dirname(__file__), "..", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("somnolence-distraction-detection-ear_mar_adaptive_filter", "somnolence-distraction-detection-ear_mar_adaptive_filter", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("somnolence-distraction-detection-head-pose-estimation", "somnolence-distraction-detection-head-pose-estimation", "head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
        os.path.join("shape_predictor_68_face_landmarks", "shape_predictor_68_face_landmarks.dat"),
    )

    def __init__(
        self,
        predictor_path: Optional[str] = None,
        pose_config: Optional[HeadPoseConfig] = None,
        distraction_config: Optional[DistractionConfig] = None,
        ear_mar_config: Optional[EarMarConfig] = None,
        visualizer_config: Optional[VisualizerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        model_path = predictor_path or self.find_predictor_file()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Could not locate 68-face landmarks model at: {model_path}")

        print("=" * 65)
        print("[*] Initializing Unified ADAS Morphed Pipeline")
        print(f"[*] Shape Predictor: {model_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        self.pose_estimator = HeadPoseEstimator(pose_config)
        self.gaze_engine = DistractionInferenceEngine(distraction_config, device=device)
        self.ear_mar = EarMarSubsystem(ear_mar_config)
        self.calibrator = UnifiedSystemCalibrator(pose_config, ear_mar_config)
        self.fusion = MultimodalAlertEngine(visualizer_config or VisualizerConfig())
        self.visualizer = UnifiedADASVisualizer(visualizer_config)

    @classmethod
    def find_predictor_file(cls) -> str:
        for p in cls.DEFAULT_PREDICTOR_PATHS:
            if os.path.isfile(p):
                return p
        return cls.DEFAULT_PREDICTOR_PATHS[0]

    def reset_calibration(self) -> None:
        self.calibrator.reset()
        self.pose_estimator.reset()
        self.gaze_engine.reset()
        print("[INFO] Full system calibration reset. Press 'c' to recalibrate.")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        fps: float = 0.0,
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[UnifiedDriverAssessment]]:
        t_frame_start = time.perf_counter()
        t = current_time if current_time is not None else time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Single-pass face detection
        t_face0 = time.perf_counter()
        faces = self.detector(gray, 0)
        face_detected = len(faces) > 0
        lat_face_detect_ms = (time.perf_counter() - t_face0) * 1000.0

        if not face_detected:
            # ── NO FACE PATH ──────────────────────────────────────────────────
            # Task 2 (gaze/distraction) always runs: LightGazeCNN/C-LSTM operates on
            # the full BGR frame and does NOT require facial landmarks.
            t_dist0 = time.perf_counter()
            distraction_out = self.gaze_engine.process_frame(frame_bgr)
            lat_dist_ms = (time.perf_counter() - t_dist0) * 1000.0

            # Frozen neutral values for landmark-dependent tasks
            pose_out = PoseOutput(
                pitch_deg=0.0,
                roll_deg=0.0,
                pitch_level=PitchLevel.NORMAL,
                is_drooping=False,
            )
            is_warmed_up = (
                (t - self.ear_mar._monitoring_start_time >= self.ear_mar.config.perclos_window_sec)
                if self.ear_mar._monitoring_start_time > 0 else False
            )
            ear_mar_out = EarMarOutput(
                ear_raw=0.0,
                ear_filtered=0.0,
                mar_raw=0.0,
                thresh_descent=self.ear_mar._thresh_descent,
                thresh_closure=self.ear_mar._thresh_closure,
                last_blink_ms=self.ear_mar.last_blink_ms,
                is_microsleep=self.ear_mar.is_microsleep,
                perclos_pct=self.ear_mar.perclos_pct,
                is_perclos_alarm=self.ear_mar.is_perclos_alarm,
                is_perclos_warmed_up=is_warmed_up,
                is_yawning=False,
                is_yawn_alarm=self.ear_mar.is_yawn_alarm,
            )

            t_fuse0 = time.perf_counter()
            if self.calibrator.phase == CalibrationPhase.ACTIVE:
                # Calibration complete — fuse normally (distraction can fire)
                risk_level, banner, color = self.fusion.evaluate(
                    pose_out, distraction_out, ear_mar_out
                )
                guidance = "No face — Gaze monitoring active"
            else:
                # Still calibrating — keep distraction live but pause calib
                risk_level = DriverRiskLevel.SAFE
                banner = "Calibrating... (no face)"
                color = (255, 255, 0)
                guidance = "No face detected — waiting to calibrate"
            lat_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0

            t_rend0 = time.perf_counter()
            assessment = UnifiedDriverAssessment(
                risk_level=risk_level,
                status_banner=banner,
                primary_color=color,
                phase=self.calibrator.phase,
                guidance_prompt=guidance,
                pose=pose_out,
                distraction=distraction_out,
                ear_mar=ear_mar_out,
                fps=fps,
                landmarks=None,
                face_bbox=None,
                stage_latencies_ms={
                    "face_detect": lat_face_detect_ms,
                    "landmarks": 0.0,
                    "face_total": lat_face_detect_ms,
                    "distraction": lat_dist_ms,
                    "pose": 0.0,
                    "ear_mar": 0.0,
                    "fusion": lat_fuse_ms,
                    "render": 0.0,
                    "total": (time.perf_counter() - t_frame_start) * 1000.0,
                },
            )
            annotated = self.visualizer.render(frame_bgr, assessment)
            lat_rend_ms = (time.perf_counter() - t_rend0) * 1000.0
            if assessment.stage_latencies_ms is not None:
                assessment.stage_latencies_ms["render"] = lat_rend_ms
                assessment.stage_latencies_ms["total"] = (time.perf_counter() - t_frame_start) * 1000.0

            return annotated, assessment

        # ── FACE DETECTED PATH ────────────────────────────────────────────────
        # Primary face shape
        t_shape0 = time.perf_counter()
        face = faces[0]
        shape = self.predictor(gray, face)
        pts = np.zeros((68, 2), dtype=int)
        for i in range(68):
            pts[i] = (shape.part(i).x, shape.part(i).y)
        lat_landmarks_ms = (time.perf_counter() - t_shape0) * 1000.0

        face_bbox = (face.left(), face.top(), face.width(), face.height())

        # Task 1: Head pose extraction (vertical ratio rho and roll)
        t_pose0 = time.perf_counter()
        rho_val, roll_val = self.pose_estimator.estimate_rho_and_roll(pts)
        lat_pose_ms = (time.perf_counter() - t_pose0) * 1000.0

        # Task 3: EAR & MAR extraction
        eye_r = pts[36:42]
        eye_l = pts[42:48]
        from .estimation_ear_mar import calc_ear
        ear_raw = (calc_ear(eye_r) + calc_ear(eye_l)) / 2.0

        # Calibration State Machine
        phase = self.calibrator.phase
        if phase != CalibrationPhase.ACTIVE:
            prev_phase = phase
            phase = self.calibrator.update(rho_val, roll_val, ear_raw, t=t)

            if prev_phase != CalibrationPhase.ACTIVE and phase == CalibrationPhase.ACTIVE:
                base = self.calibrator.baseline
                if base is not None:
                    self.ear_mar.initialize_baselines(
                        base.ear_open, base.ear_closed, base.thresh_descent, base.thresh_closure
                    )
                    self.ear_mar.start_monitoring(t)

            t_dist0 = time.perf_counter()
            distraction_out = self.gaze_engine.process_frame(frame_bgr)
            lat_dist_ms = (time.perf_counter() - t_dist0) * 1000.0

            t_ear0 = time.perf_counter()
            ear_mar_out = self.ear_mar.process(pts, is_active=False, current_time=t)
            lat_ear_ms = (time.perf_counter() - t_ear0) * 1000.0

            prompt = self.calibrator.get_guidance_text(t)
            t_rend0 = time.perf_counter()
            assessment = UnifiedDriverAssessment(
                risk_level=DriverRiskLevel.SAFE,
                status_banner="Calibrating...",
                primary_color=(255, 255, 0),
                phase=phase,
                guidance_prompt=prompt,
                pose=PoseOutput(0.0, roll_val, PitchLevel.NORMAL, False),
                distraction=distraction_out,
                ear_mar=ear_mar_out,
                fps=fps,
                landmarks=pts,
                face_bbox=face_bbox,
                stage_latencies_ms={
                    "face_detect": lat_face_detect_ms,
                    "landmarks": lat_landmarks_ms,
                    "face_total": lat_face_detect_ms + lat_landmarks_ms,
                    "distraction": lat_dist_ms,
                    "pose": lat_pose_ms,
                    "ear_mar": lat_ear_ms,
                    "fusion": 0.0,
                    "render": 0.0,
                    "total": (time.perf_counter() - t_frame_start) * 1000.0,
                },
            )
            annotated = self.visualizer.render(frame_bgr, assessment)
            lat_rend_ms = (time.perf_counter() - t_rend0) * 1000.0
            if assessment.stage_latencies_ms is not None:
                assessment.stage_latencies_ms["render"] = lat_rend_ms
                assessment.stage_latencies_ms["total"] = (time.perf_counter() - t_frame_start) * 1000.0

            return annotated, assessment

        # ACTIVE MONITORING (All 3 Subsystems in parallel)
        base = self.calibrator.baseline

        # 1. Head Pose & Pitch Droop
        t_pose0 = time.perf_counter()
        pose_out = self.pose_estimator.evaluate_pose(pts, base.rho, base.roll)
        lat_pose_ms = (time.perf_counter() - t_pose0) * 1000.0

        # 2. Gaze Distraction
        t_dist0 = time.perf_counter()
        distraction_out = self.gaze_engine.process_frame(frame_bgr)
        lat_dist_ms = (time.perf_counter() - t_dist0) * 1000.0

        # 3. EAR & MAR Dynamics
        t_ear0 = time.perf_counter()
        ear_mar_out = self.ear_mar.process(pts, is_active=True, current_time=t)
        lat_ear_ms = (time.perf_counter() - t_ear0) * 1000.0

        # 4. Multimodal Fusion Arbitration
        t_fuse0 = time.perf_counter()
        risk_level, banner, color = self.fusion.evaluate(pose_out, distraction_out, ear_mar_out)
        lat_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0

        t_rend0 = time.perf_counter()
        assessment = UnifiedDriverAssessment(
            risk_level=risk_level,
            status_banner=banner,
            primary_color=color,
            phase=CalibrationPhase.ACTIVE,
            guidance_prompt="Active Monitoring",
            pose=pose_out,
            distraction=distraction_out,
            ear_mar=ear_mar_out,
            fps=fps,
            landmarks=pts,
            face_bbox=face_bbox,
            stage_latencies_ms={
                "face_detect": lat_face_detect_ms,
                "landmarks": lat_landmarks_ms,
                "face_total": lat_face_detect_ms + lat_landmarks_ms,
                "distraction": lat_dist_ms,
                "pose": lat_pose_ms,
                "ear_mar": lat_ear_ms,
                "fusion": lat_fuse_ms,
                "render": 0.0,
                "total": (time.perf_counter() - t_frame_start) * 1000.0,
            },
        )
        annotated = self.visualizer.render(frame_bgr, assessment)
        lat_rend_ms = (time.perf_counter() - t_rend0) * 1000.0
        if assessment.stage_latencies_ms is not None:
            assessment.stage_latencies_ms["render"] = lat_rend_ms
            assessment.stage_latencies_ms["total"] = (time.perf_counter() - t_frame_start) * 1000.0

        return annotated, assessment
        annotated = self.visualizer.render(frame_bgr, assessment)
        lat_rend_ms = (time.perf_counter() - t_rend0) * 1000.0
        if assessment.stage_latencies_ms is not None:
            assessment.stage_latencies_ms["render"] = lat_rend_ms
            assessment.stage_latencies_ms["total"] = (time.perf_counter() - t_frame_start) * 1000.0

        return annotated, assessment

    def run(
        self,
        camera_idx: Union[int, str] = 0,
        enable_benchmark: bool = True,
        report_dir: str = "reports/sessions",
        save_csv: bool = True,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
        session_tag: Optional[str] = None,
    ) -> Optional[Tuple[str, Optional[str]]]:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/video device: {camera_idx}")

        print("=" * 65)
        print("[*] Unified ADAS Pipeline Active")
        print(f"[*] Distraction Backend: {self.gaze_engine.backend.value.upper()}")
        if enable_benchmark:
            print("[*] Real-Time Session Telemetry Benchmarking: ACTIVE")
            print(f"[*] Reports Directory: {os.path.abspath(report_dir)}")
            print("[*] Individual run results (Markdown + CSV) generated on exit.")
        print("[*] Controls: 'c' = Calibrate | 'r' = Reset | 'q' = Quit & Save Report")
        print("=" * 65)

        tracker: Optional[LiveSessionBenchmarkTracker] = None
        if enable_benchmark:
            tracker = LiveSessionBenchmarkTracker(
                backend=self.gaze_engine.backend,
                output_dir=report_dir,
                session_tag=session_tag,
            )

        prev_time = time.time()
        start_session_time = time.time()
        fps = 0.0
        frame_idx = 0
        report_paths: Optional[Tuple[str, Optional[str]]] = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only flip horizontal if webcam index (not pre-recorded video path)
                if isinstance(camera_idx, int):
                    frame = cv2.flip(frame, 1)

                curr_time = time.time()
                diff = curr_time - prev_time
                if diff > 0:
                    fps = 1.0 / diff
                prev_time = curr_time
                frame_idx += 1

                annotated, assessment = self.process_frame(frame, fps=fps, current_time=curr_time)

                if tracker is not None and assessment is not None:
                    tracker.record_frame(
                        frame_idx=frame_idx,
                        assessment=assessment,
                        stage_latencies=assessment.stage_latencies_ms or {},
                        fps=fps,
                    )

                cv2.imshow("Morphed ADAS System (All 3 Tasks Unified)", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.calibrator.start_calibration()
                elif key == ord("r"):
                    self.reset_calibration()

                if max_frames is not None and frame_idx >= max_frames:
                    print(f"[*] Reached target maximum frames ({max_frames}). Stopping.")
                    break
                if max_seconds is not None and (curr_time - start_session_time) >= max_seconds:
                    print(f"[*] Reached target duration ({max_seconds:.1f}s). Stopping.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if tracker is not None:
                report_paths = tracker.finalize(save_csv=save_csv)

        return report_paths
