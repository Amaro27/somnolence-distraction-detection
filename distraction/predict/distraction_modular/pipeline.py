"""
Pipeline orchestrator and runtime loop for modular driver distraction detection.

Coordinates camera capture, preprocessing, model inference (LightGazeCNN or C-LSTM),
temporal smoothing, and HUD visualization in a unified processing loop.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

import cv2
import numpy as np
import torch

try:
    from .configuration import (
        BackendType,
        BenchmarkStats,
        CLSTMConfig,
        DistractionAssessment,
        LightGazeConfig,
        RawInferenceResult,
    )
    from .models import GazeResNetLSTM, LightGazeCNN, load_clstm, load_light_cnn
    from .preprocessing import FramePreprocessor
    from .temporal import LSTMSequenceBuffer, ModeTemporalSmoother
    from .visualizer import DistractionVisualizer
except ImportError:
    from configuration import (
        BackendType,
        BenchmarkStats,
        CLSTMConfig,
        DistractionAssessment,
        LightGazeConfig,
        RawInferenceResult,
    )
    from models import GazeResNetLSTM, LightGazeCNN, load_clstm, load_light_cnn
    from preprocessing import FramePreprocessor
    from temporal import LSTMSequenceBuffer, ModeTemporalSmoother
    from visualizer import DistractionVisualizer


class DistractionDetector(Protocol):
    """Protocol contract implemented by both LightGazeCNN and C-LSTM backends."""
    def process_frame(self, frame_bgr: np.ndarray) -> DistractionAssessment:
        ...

    def reset(self) -> None:
        ...

    @property
    def model_name(self) -> str:
        ...


class LightCNNDetector:
    """Encapsulates LightGazeCNN forward inference and mode temporal smoothing."""

    def __init__(
        self,
        config: Optional[LightGazeConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or LightGazeConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FramePreprocessor.for_light_cnn(self.config, self.device)
        self.model: LightGazeCNN = load_light_cnn(self.config, self.device)
        self.smoother = ModeTemporalSmoother(window_size=self.config.window_size)

    @property
    def model_name(self) -> str:
        return "LightGazeCNN"

    def reset(self) -> None:
        self.smoother.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> DistractionAssessment:
        input_tensor = self.preprocessor.preprocess_bgr(frame_bgr)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf_tensor, pred_tensor = torch.max(probs, 1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        pred_idx = int(pred_tensor.item())
        confidence = float(conf_tensor.item() * 100.0)
        prob_tuple = tuple(float(p) for p in probs[0].cpu().numpy())
        label_name = "looking_road" if pred_idx == 1 else "not_looking_road"

        raw = RawInferenceResult(
            pred_idx=pred_idx,
            raw_label=label_name,
            confidence=confidence,
            probabilities=prob_tuple,
            latency_ms=latency_ms,
        )

        return self.smoother.update(raw)


class CLSTMDetector:
    """Encapsulates ResNet18 feature extraction and recurrent LSTM sequence evaluation."""

    def __init__(
        self,
        config: Optional[CLSTMConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or CLSTMConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FramePreprocessor.for_clstm(self.config, self.device)
        self.model: GazeResNetLSTM = load_clstm(self.config, self.device)
        self.sequence_buffer = LSTMSequenceBuffer(
            model=self.model,
            seq_len=self.config.seq_len,
            device=self.device,
        )

    @property
    def model_name(self) -> str:
        return "GazeResNetLSTM (C-LSTM)"

    def reset(self) -> None:
        self.sequence_buffer.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> DistractionAssessment:
        input_tensor = self.preprocessor.preprocess_bgr(frame_bgr)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Step 1: Spatial 512-d feature extraction
        frame_feat = self.model.extract_single_frame_feature(input_tensor)

        # Step 2: Temporal LSTM sequence evaluation
        assessment = self.sequence_buffer.update(frame_feat, 0.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return DistractionAssessment(
            status=assessment.status,
            smoothed_label=assessment.smoothed_label,
            raw_label=assessment.raw_label,
            confidence=assessment.confidence,
            is_distracted=assessment.is_distracted,
            buffer_length=assessment.buffer_length,
            buffer_capacity=assessment.buffer_capacity,
            inference_latency_ms=total_latency_ms,
        )


class DistractionPipeline:
    """Unified video processing pipeline for real-time driver distraction detection."""

    def __init__(
        self,
        detector: DistractionDetector,
        visualizer: Optional[DistractionVisualizer] = None,
    ):
        self.detector = detector
        self.visualizer = visualizer or DistractionVisualizer()
        self.latencies: list[float] = []

    @classmethod
    def create_light_cnn(
        cls,
        weights_path: Optional[str] = None,
        window_size: int = 15,
        device: Optional[torch.device] = None,
    ) -> DistractionPipeline:
        cfg = LightGazeConfig(
            weights_path=weights_path or LightGazeConfig.weights_path,
            window_size=window_size,
        )
        detector = LightCNNDetector(config=cfg, device=device)
        return cls(detector=detector)

    @classmethod
    def create_clstm(
        cls,
        weights_path: Optional[str] = None,
        seq_len: int = 20,
        device: Optional[torch.device] = None,
    ) -> DistractionPipeline:
        cfg = CLSTMConfig(
            weights_path=weights_path or CLSTMConfig.weights_path,
            seq_len=seq_len,
        )
        detector = CLSTMDetector(config=cfg, device=device)
        return cls(detector=detector)

    def process_frame(self, frame_bgr: np.ndarray, fps: float = 0.0) -> np.ndarray:
        """Processes single image frame and overlays visual HUD."""
        assessment = self.detector.process_frame(frame_bgr)
        self.latencies.append(assessment.inference_latency_ms)
        return self.visualizer.draw_hud(frame_bgr, assessment, fps, self.detector.model_name)

    def get_benchmark_report(self) -> BenchmarkStats:
        """Calculates descriptive latency and throughput stats over processed frames."""
        if not self.latencies:
            return BenchmarkStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        arr = np.array(self.latencies)
        avg_lat = float(np.mean(arr))
        fps_cap = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        return BenchmarkStats(
            sample_count=len(arr),
            avg_latency_ms=avg_lat,
            median_latency_ms=float(np.median(arr)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            min_latency_ms=float(np.min(arr)),
            max_latency_ms=float(np.max(arr)),
            std_latency_ms=float(np.std(arr)),
            fps_cap=fps_cap,
        )

    def run(self, camera_idx: int = 0) -> None:
        """Main webcam acquisition and presentation loop."""
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera stream at index {camera_idx}.")

        print("=" * 60)
        print(f"[*] Starting Distraction Detection Pipeline ({self.detector.model_name})")
        print("[*] Controls: Press 'q' to quit | 'r' to reset temporal buffer")
        print("=" * 60)

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Frame capture dropped.")
                    break

                curr_time = time.time()
                diff = curr_time - prev_time
                if diff > 0:
                    fps = 1.0 / diff
                prev_time = curr_time

                annotated = self.process_frame(frame, fps=fps)
                cv2.imshow(f"Distraction Detection ({self.detector.model_name})", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.detector.reset()
                    print("[INFO] Temporal buffer reset.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        stats = self.get_benchmark_report()
        if stats.sample_count > 0:
            print("\n" + "=" * 55)
            print(f"[*] INFERENCE BENCHMARK REPORT ({self.detector.model_name})")
            print(f"[*] Total Frames Processed: {stats.sample_count:,}")
            print(f"[*] Avg Latency:            {stats.avg_latency_ms:.2f} ms ({stats.fps_cap:.1f} FPS theoretical cap)")
            print(f"[*] Median Latency:         {stats.median_latency_ms:.2f} ms")
            print(f"[*] 95th Percentile (p95):  {stats.p95_latency_ms:.2f} ms")
            print(f"[*] Min / Max Latency:      {stats.min_latency_ms:.2f} ms / {stats.max_latency_ms:.2f} ms")
            print("=" * 55 + "\n")
