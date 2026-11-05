"""
Driver Distraction Hardware Latency & Throughput Benchmark Suite.

Profiles raw pipeline execution speed, preprocessing time, neural forward pass,
and temporal aggregation on CPU and CUDA without camera I/O overhead.
Generates comprehensive descriptive statistics and markdown benchmark reports.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure modular imports work regardless of execution location
current_dir = os.path.dirname(os.path.abspath(__file__))
distraction_root = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(distraction_root, ".."))

for path in [project_root, distraction_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from distraction.predict.distraction_modular.configuration import (
    BackendType,
    BenchmarkStats,
    CLSTMConfig,
    LightGazeConfig,
)
from distraction.predict.distraction_modular.models import (
    GazeResNetLSTM,
    LightGazeCNN,
    load_clstm,
    load_light_cnn,
)
from distraction.predict.distraction_modular.preprocessing import FramePreprocessor
from distraction.predict.distraction_modular.temporal import (
    LSTMSequenceBuffer,
    ModeTemporalSmoother,
)


@dataclass
class StageLatencyStats:
    """Breakdown of individual pipeline execution stages."""
    stage_name: str
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkResult:
    """Summary record for a hardware benchmark profile run."""
    model_name: str
    device: str
    total_frames: int
    warmup_frames: int
    preprocess: StageLatencyStats
    inference: StageLatencyStats
    temporal: StageLatencyStats
    total: StageLatencyStats
    theoretical_fps: float


class LatencyProfiler:
    """Measures precise multi-stage latencies for distraction detection models."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_stats(self, name: str, timings: List[float]) -> StageLatencyStats:
        arr = np.array(timings)
        return StageLatencyStats(
            stage_name=name,
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )

    def benchmark_light_cnn(
        self,
        config: Optional[LightGazeConfig] = None,
        iterations: int = 500,
        warmup: int = 50,
    ) -> BenchmarkResult:
        cfg = config or LightGazeConfig()
        model: LightGazeCNN = load_light_cnn(cfg, self.device)
        preprocessor = FramePreprocessor.for_light_cnn(cfg, self.device)
        smoother = ModeTemporalSmoother(window_size=cfg.window_size)

        # Synthetic camera frame (480x640 BGR)
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            tensor = preprocessor.preprocess_bgr(dummy_frame)
            with torch.no_grad():
                _ = model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        prep_times: List[float] = []
        infer_times: List[float] = []
        temp_times: List[float] = []
        total_times: List[float] = []

        for _ in range(iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1. Preprocessing
            tensor = preprocessor.preprocess_bgr(dummy_frame)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 2. Forward inference
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # 3. Temporal mode aggregation
            raw_pred = int(pred.item())
            raw_conf = float(conf.item() * 100.0)
            from distraction.predict.distraction_modular.configuration import RawInferenceResult
            raw = RawInferenceResult(
                pred_idx=raw_pred,
                raw_label="road",
                confidence=raw_conf,
                probabilities=(0.5, 0.5),
                latency_ms=(t2 - t1) * 1000.0,
            )
            _ = smoother.update(raw)
            t3 = time.perf_counter()

            prep_times.append((t1 - t0) * 1000.0)
            infer_times.append((t2 - t1) * 1000.0)
            temp_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

        total_stats = self._compute_stats("Total Pipeline", total_times)
        fps_cap = 1000.0 / total_stats.mean_ms if total_stats.mean_ms > 0 else 0.0

        return BenchmarkResult(
            model_name="LightGazeCNN",
            device=str(self.device),
            total_frames=iterations,
            warmup_frames=warmup,
            preprocess=self._compute_stats("Preprocessing (Resize+Norm)", prep_times),
            inference=self._compute_stats("CNN Forward Pass", infer_times),
            temporal=self._compute_stats("Mode Smoothing", temp_times),
            total=total_stats,
            theoretical_fps=fps_cap,
        )

    def benchmark_clstm(
        self,
        config: Optional[CLSTMConfig] = None,
        iterations: int = 500,
        warmup: int = 50,
    ) -> BenchmarkResult:
        cfg = config or CLSTMConfig()
        model: GazeResNetLSTM = load_clstm(cfg, self.device)
        preprocessor = FramePreprocessor.for_clstm(cfg, self.device)
        seq_buffer = LSTMSequenceBuffer(model=model, seq_len=cfg.seq_len, device=self.device)

        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            tensor = preprocessor.preprocess_bgr(dummy_frame)
            feat = model.extract_single_frame_feature(tensor)
            _ = seq_buffer.update(feat, 0.0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        prep_times: List[float] = []
        infer_times: List[float] = []
        temp_times: List[float] = []
        total_times: List[float] = []

        for _ in range(iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1. Preprocessing (Resize + CLAHE + Norm)
            tensor = preprocessor.preprocess_bgr(dummy_frame)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 2. ResNet18 Spatial Feature Extraction
            feat = model.extract_single_frame_feature(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # 3. Temporal LSTM Sequence Forward
            _ = seq_buffer.update(feat, 0.0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            prep_times.append((t1 - t0) * 1000.0)
            infer_times.append((t2 - t1) * 1000.0)
            temp_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

        total_stats = self._compute_stats("Total Pipeline", total_times)
        fps_cap = 1000.0 / total_stats.mean_ms if total_stats.mean_ms > 0 else 0.0

        return BenchmarkResult(
            model_name="GazeResNetLSTM (C-LSTM)",
            device=str(self.device),
            total_frames=iterations,
            warmup_frames=warmup,
            preprocess=self._compute_stats("Preprocessing (CLAHE+Norm)", prep_times),
            inference=self._compute_stats("ResNet18 Feature Extractor", infer_times),
            temporal=self._compute_stats("LSTM Sequence Forward", temp_times),
            total=total_stats,
            theoretical_fps=fps_cap,
        )


def export_markdown_report(result: BenchmarkResult, output_path: str) -> None:
    """Generates an executive markdown benchmark report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    md_content = f"""# Hardware Latency & Throughput Benchmark Report: {result.model_name}

> **Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Target Device**: `{result.device}`  
> **Iterations**: {result.total_frames:,} frames (Warmup: {result.warmup_frames})  
> **Theoretical Maximum Throughput**: **{result.theoretical_fps:.1f} FPS**  

---

## 1. Executive Summary & Latency Breakdown

| Pipeline Stage | Mean Latency (ms) | Std Dev (ms) | Median (ms) | 95th Percentile (ms) | Min / Max (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **{result.preprocess.stage_name}** | {result.preprocess.mean_ms:.2f} ms | ±{result.preprocess.std_ms:.2f} | {result.preprocess.median_ms:.2f} ms | {result.preprocess.p95_ms:.2f} ms | {result.preprocess.min_ms:.2f} / {result.preprocess.max_ms:.2f} ms |
| **{result.inference.stage_name}** | {result.inference.mean_ms:.2f} ms | ±{result.inference.std_ms:.2f} | {result.inference.median_ms:.2f} ms | {result.inference.p95_ms:.2f} ms | {result.inference.min_ms:.2f} / {result.inference.max_ms:.2f} ms |
| **{result.temporal.stage_name}** | {result.temporal.mean_ms:.2f} ms | ±{result.temporal.std_ms:.2f} | {result.temporal.median_ms:.2f} ms | {result.temporal.p95_ms:.2f} ms | {result.temporal.min_ms:.2f} / {result.temporal.max_ms:.2f} ms |
| **{result.total.stage_name}** | **{result.total.mean_ms:.2f} ms** | **±{result.total.std_ms:.2f}** | **{result.total.median_ms:.2f} ms** | **{result.total.p95_ms:.2f} ms** | **{result.total.min_ms:.2f} / {result.total.max_ms:.2f} ms** |

---

## 2. Real-Time Deployment Assessment

- **Real-Time Viability**: At **{result.theoretical_fps:.1f} FPS**, the {result.model_name} model operates {'well above standard 30 FPS video stream rates' if result.theoretical_fps >= 30 else 'below 30 FPS stream rates, requiring strided or threaded execution on this hardware'}.
- **Bottleneck Analysis**:
  - Preprocessing represents approximately {(result.preprocess.mean_ms / max(result.total.mean_ms, 1e-6)) * 100.0:.1f}% of execution time.
  - Neural inference represents approximately {(result.inference.mean_ms / max(result.total.mean_ms, 1e-6)) * 100.0:.1f}% of execution time.
  - Temporal state aggregation represents approximately {(result.temporal.mean_ms / max(result.total.mean_ms, 1e-6)) * 100.0:.1f}% of execution time.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved benchmark report to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile distraction model execution latency")
    parser.add_argument(
        "--backend",
        type=str,
        choices=[BackendType.LIGHT_CNN.value, BackendType.CLSTM.value, "all"],
        default="all",
        help="Model architecture to profile",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(distraction_root, "eval", "reports"),
        help="Directory to save output reports",
    )
    args = parser.parse_args()

    dev = torch.device(args.device)
    profiler = LatencyProfiler(device=dev)

    backends_to_run = (
        [BackendType.LIGHT_CNN.value, BackendType.CLSTM.value]
        if args.backend == "all"
        else [args.backend]
    )

    for backend in backends_to_run:
        print(f"\n[*] Benchmarking {backend.upper()} on {dev} ({args.iterations} frames)...")
        if backend == BackendType.LIGHT_CNN.value:
            res = profiler.benchmark_light_cnn(iterations=args.iterations, warmup=args.warmup)
        else:
            res = profiler.benchmark_clstm(iterations=args.iterations, warmup=args.warmup)

        out_file = os.path.join(args.output_dir, f"latency_benchmark_{backend}.md")
        export_markdown_report(res, out_file)


if __name__ == "__main__":
    main()
