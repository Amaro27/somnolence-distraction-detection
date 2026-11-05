"""
Real-Time ADAS Hardware & Operational Telemetry Benchmarking Suite.

Captures per-frame latencies and safety telemetry during live camera / video execution.
On session termination, generates an individual, uniquely timestamped benchmark report
(Markdown + CSV) detailing hardware throughput, latency percentiles, and driver alertness metrics.
"""

from __future__ import annotations

import csv
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .configuration import (
        BackendType,
        DriverRiskLevel,
        GazeStatus,
        UnifiedDriverAssessment,
    )
except ImportError:
    from configuration import (
        BackendType,
        DriverRiskLevel,
        GazeStatus,
        UnifiedDriverAssessment,
    )


@dataclass
class FrameTelemetryRecord:
    """Individual frame telemetry snapshot."""
    frame_idx: int
    relative_time_sec: float
    fps: float
    lat_face_detect_ms: float
    lat_landmarks_ms: float
    lat_distraction_ms: float
    lat_pose_ms: float
    lat_ear_mar_ms: float
    lat_fusion_ms: float
    lat_render_ms: float
    lat_total_ms: float
    face_detected: bool
    gaze_status: str
    gaze_confidence: float
    is_distracted: bool
    pitch_deg: float
    roll_deg: float
    is_drooping: bool
    ear_val: float
    mar_val: float
    blink_ms: float
    is_microsleep: bool
    perclos_pct: float
    is_yawning: bool
    risk_level: str


class LiveSessionBenchmarkTracker:
    """Tracks per-frame metrics during live webcam runs and produces individual reports."""

    def __init__(
        self,
        backend: BackendType = BackendType.LIGHT_CNN,
        output_dir: str = "reports/sessions",
        session_tag: Optional[str] = None,
    ):
        self.backend = backend
        self.output_dir = output_dir
        self.start_wall_time = time.time()
        self.start_perf_time = time.perf_counter()
        self.records: List[FrameTelemetryRecord] = []

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        tag_suffix = f"_{session_tag}" if session_tag else ""
        self.session_id = f"live_session_{timestamp_str}_{self.backend.value}{tag_suffix}"

    def record_frame(
        self,
        frame_idx: int,
        assessment: UnifiedDriverAssessment,
        stage_latencies: Dict[str, float],
        fps: float,
    ) -> None:
        """Pushes instantaneous frame telemetry into chronological record buffer."""
        curr_rel_time = time.perf_counter() - self.start_perf_time
        face_present = assessment.landmarks is not None or assessment.face_bbox is not None

        record = FrameTelemetryRecord(
            frame_idx=frame_idx,
            relative_time_sec=curr_rel_time,
            fps=fps,
            lat_face_detect_ms=stage_latencies.get("face_detect", 0.0),
            lat_landmarks_ms=stage_latencies.get("landmarks", 0.0),
            lat_distraction_ms=stage_latencies.get("distraction", 0.0),
            lat_pose_ms=stage_latencies.get("pose", 0.0),
            lat_ear_mar_ms=stage_latencies.get("ear_mar", 0.0),
            lat_fusion_ms=stage_latencies.get("fusion", 0.0),
            lat_render_ms=stage_latencies.get("render", 0.0),
            lat_total_ms=stage_latencies.get("total", 0.0),
            face_detected=face_present,
            gaze_status=assessment.distraction.status.value,
            gaze_confidence=assessment.distraction.confidence,
            is_distracted=assessment.distraction.is_distracted,
            pitch_deg=assessment.pose.pitch_deg,
            roll_deg=assessment.pose.roll_deg,
            is_drooping=assessment.pose.is_drooping,
            ear_val=assessment.ear_mar.ear_raw,
            mar_val=assessment.ear_mar.mar_raw,
            blink_ms=assessment.ear_mar.last_blink_ms,
            is_microsleep=assessment.ear_mar.is_microsleep,
            perclos_pct=assessment.ear_mar.perclos_pct,
            is_yawning=assessment.ear_mar.is_yawning,
            risk_level=assessment.risk_level.value,
        )
        self.records.append(record)

    def finalize(self, save_csv: bool = True) -> Tuple[str, Optional[str]]:
        """Saves individual Markdown and CSV reports for this specific run."""
        if not self.records:
            print("[WARNING] No frames recorded during live session. Skipping report generation.")
            return "", None

        os.makedirs(self.output_dir, exist_ok=True)
        md_path = os.path.join(self.output_dir, f"{self.session_id}.md")
        csv_path = os.path.join(self.output_dir, f"{self.session_id}.csv") if save_csv else None

        self._export_markdown(md_path)
        if csv_path:
            self._export_csv(csv_path)

        self._print_terminal_summary()
        return md_path, csv_path

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _export_csv(self, output_path: str) -> None:
        """Exports row-by-row telemetry for granular time-series analysis."""
        fieldnames = [
            "frame_idx",
            "relative_time_sec",
            "fps",
            "lat_face_detect_ms",
            "lat_landmarks_ms",
            "lat_distraction_ms",
            "lat_pose_ms",
            "lat_ear_mar_ms",
            "lat_fusion_ms",
            "lat_render_ms",
            "lat_total_ms",
            "face_detected",
            "gaze_status",
            "gaze_confidence",
            "is_distracted",
            "pitch_deg",
            "roll_deg",
            "is_drooping",
            "ear_val",
            "mar_val",
            "blink_ms",
            "is_microsleep",
            "perclos_pct",
            "is_yawning",
            "risk_level",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))
        print(f"[+] Saved Live Telemetry CSV to: {output_path}")

    def _export_markdown(self, output_path: str) -> None:
        """Generates publication-quality session summary report."""
        total_frames = len(self.records)
        duration_sec = self.records[-1].relative_time_sec if total_frames > 0 else 0.0

        # Latency statistics
        fps_stats = self._compute_stats([r.fps for r in self.records if r.fps > 0])
        total_lat = self._compute_stats([r.lat_total_ms for r in self.records])
        face_det_lat = self._compute_stats([r.lat_face_detect_ms for r in self.records])
        lmk_lat = self._compute_stats([r.lat_landmarks_ms for r in self.records if r.face_detected])
        face_total_lat = self._compute_stats([r.lat_face_detect_ms + r.lat_landmarks_ms for r in self.records])
        dist_lat = self._compute_stats([r.lat_distraction_ms for r in self.records])
        pose_lat = self._compute_stats([r.lat_pose_ms for r in self.records if r.face_detected])
        ear_lat = self._compute_stats([r.lat_ear_mar_ms for r in self.records if r.face_detected])
        fuse_lat = self._compute_stats([r.lat_fusion_ms for r in self.records])
        rend_lat = self._compute_stats([r.lat_render_ms for r in self.records])

        # Operational statistics
        face_detected_count = sum(1 for r in self.records if r.face_detected)
        face_detected_pct = (face_detected_count / total_frames * 100.0) if total_frames > 0 else 0.0

        distracted_count = sum(1 for r in self.records if r.is_distracted)
        distracted_pct = (distracted_count / total_frames * 100.0) if total_frames > 0 else 0.0

        drooping_count = sum(1 for r in self.records if r.is_drooping)
        drooping_pct = (drooping_count / total_frames * 100.0) if total_frames > 0 else 0.0

        microsleep_count = sum(1 for r in self.records if r.is_microsleep)
        yawn_count = sum(1 for r in self.records if r.is_yawning)
        max_perclos = max((r.perclos_pct for r in self.records), default=0.0)

        # Risk breakdown
        risk_counts = {
            DriverRiskLevel.SAFE.value: sum(1 for r in self.records if r.risk_level == DriverRiskLevel.SAFE.value),
            DriverRiskLevel.DISTRACTED.value: sum(1 for r in self.records if r.risk_level == DriverRiskLevel.DISTRACTED.value),
            DriverRiskLevel.DROWSY.value: sum(1 for r in self.records if r.risk_level == DriverRiskLevel.DROWSY.value),
            DriverRiskLevel.CRITICAL_RISK.value: sum(1 for r in self.records if r.risk_level == DriverRiskLevel.CRITICAL_RISK.value),
        }

        backend_title = "LightGazeCNN (112x112)" if self.backend == BackendType.LIGHT_CNN else "GazeResNetLSTM / C-LSTM (224x224)"

        md_content = f"""# Real-Time ADAS Live Session Benchmark Report

> **Session ID**: `{self.session_id}`  
> **Start Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_wall_time))}  
> **Active Distraction Backend**: **{backend_title}**  
> **Host Platform**: `{platform.system()} {platform.release()} ({platform.machine()})`  
> **Python Version**: `{platform.python_version()}`  

---

## 1. Session Overview & Hardware Throughput

| Session Metric | Value |
|:---|:---:|
| **Total Monitored Duration** | **{duration_sec:.2f} seconds** ({duration_sec / 60.0:.2f} min) |
| **Total Frames Processed** | **{total_frames:,} frames** |
| **Effective Mean FPS** | **{fps_stats['mean']:.1f} FPS** (Median: {fps_stats['median']:.1f} FPS, P95: {fps_stats['p95']:.1f} FPS) |
| **Mean End-to-End Latency** | **{total_lat['mean']:.2f} ms** (Std: ±{total_lat['std']:.2f} ms) |
| **95th Percentile Latency** | **{total_lat['p95']:.2f} ms** |
| **Min / Max Latency** | {total_lat['min']:.2f} ms / {total_lat['max']:.2f} ms |
| **Real-Time 30 FPS Compliance** | {'✅ Fully Compliant (Mean FPS >= 30)' if fps_stats['mean'] >= 30.0 else '⚠️ Below 30 FPS Stream Threshold'} |

---

## 2. Granular Multi-Stage Latency Breakdown (Face & Landmarks Isolated)

| Pipeline Stage | Mean Latency | Median | 95th Percentile | Min / Max | % of Frame Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Face Bounding Box Detection (dlib HOG)** | **{face_det_lat['mean']:.2f} ms** | {face_det_lat['median']:.2f} ms | {face_det_lat['p95']:.2f} ms | {face_det_lat['min']:.2f} / {face_det_lat['max']:.2f} ms | {(face_det_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **2. 68 Facial Landmark Regression (dlib Predictor)** | **{lmk_lat['mean']:.2f} ms** | {lmk_lat['median']:.2f} ms | {lmk_lat['p95']:.2f} ms | {lmk_lat['min']:.2f} / {lmk_lat['max']:.2f} ms | {(lmk_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **[Subtotal: Combined Face & Landmarks]** | **{face_total_lat['mean']:.2f} ms** | {face_total_lat['median']:.2f} ms | {face_total_lat['p95']:.2f} ms | {face_total_lat['min']:.2f} / {face_total_lat['max']:.2f} ms | {(face_total_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **3. Distraction Inference ({self.backend.value})** | **{dist_lat['mean']:.2f} ms** | {dist_lat['median']:.2f} ms | {dist_lat['p95']:.2f} ms | {dist_lat['min']:.2f} / {dist_lat['max']:.2f} ms | {(dist_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **4. Head Pose Estimation (rho & roll)** | **{pose_lat['mean']:.3f} ms** | {pose_lat['median']:.3f} ms | {pose_lat['p95']:.3f} ms | {pose_lat['min']:.3f} / {pose_lat['max']:.3f} ms | {(pose_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **5. EAR / MAR & Adaptive Filter** | **{ear_lat['mean']:.3f} ms** | {ear_lat['median']:.3f} ms | {ear_lat['p95']:.3f} ms | {ear_lat['min']:.3f} / {ear_lat['max']:.3f} ms | {(ear_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **6. Multimodal Alert Fusion** | **{fuse_lat['mean']:.3f} ms** | {fuse_lat['median']:.3f} ms | {fuse_lat['p95']:.3f} ms | {fuse_lat['min']:.3f} / {fuse_lat['max']:.3f} ms | {(fuse_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **7. HUD Visualization & Render** | **{rend_lat['mean']:.2f} ms** | {rend_lat['median']:.2f} ms | {rend_lat['p95']:.2f} ms | {rend_lat['min']:.2f} / {rend_lat['max']:.2f} ms | {(rend_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% |
| **Full Frame Cycle Total** | **{total_lat['mean']:.2f} ms** | **{total_lat['median']:.2f} ms** | **{total_lat['p95']:.2f} ms** | **{total_lat['min']:.2f} / {total_lat['max']:.2f} ms** | **100.0%** |

---

## 3. Operational ADAS Safety & Alert Telemetry

### Face Visibility & Landmark Quality
- **Frames with Face Detected**: **{face_detected_count:,} / {total_frames:,}** ({face_detected_pct:.1f}%)
- **Frames without Face**: **{total_frames - face_detected_count:,}** ({100.0 - face_detected_pct:.1f}%)
  > *Note: When no face is visible, the pipeline maintains active gaze distraction monitoring on the full frame.*

### Distraction Telemetry
- **Gaze Distraction Active**: **{distracted_count:,} frames** ({distracted_pct:.1f}% of total session)
- **Road-Focused Attention**: **{total_frames - distracted_count:,} frames** ({100.0 - distracted_pct:.1f}%)

### Fatigue, Somnolence & Drowsiness Telemetry
- **Head Pitch Droop Frames**: **{drooping_count:,}** ({drooping_pct:.1f}%)
- **Micro-Sleep Events (>= 500ms)**: **{microsleep_count}**
- **Yawn Alerts (>= 4s)**: **{yawn_count}**
- **Peak 30s PERCLOS Observed**: **{max_perclos:.1f}%** (Threshold: 15.0%)

### Multimodal Alert Distribution

| Alert Risk Level | Frame Count | Percentage of Driving Time |
|:---|:---:|:---:|
| **SAFE** | {risk_counts[DriverRiskLevel.SAFE.value]:,} | {(risk_counts[DriverRiskLevel.SAFE.value] / max(total_frames, 1)) * 100.0:.1f}% |
| **DISTRACTED** | {risk_counts[DriverRiskLevel.DISTRACTED.value]:,} | {(risk_counts[DriverRiskLevel.DISTRACTED.value] / max(total_frames, 1)) * 100.0:.1f}% |
| **DROWSY** | {risk_counts[DriverRiskLevel.DROWSY.value]:,} | {(risk_counts[DriverRiskLevel.DROWSY.value] / max(total_frames, 1)) * 100.0:.1f}% |
| **CRITICAL_RISK** | {risk_counts[DriverRiskLevel.CRITICAL_RISK.value]:,} | {(risk_counts[DriverRiskLevel.CRITICAL_RISK.value] / max(total_frames, 1)) * 100.0:.1f}% |

---

## 4. Engineering Takeaways & Bottleneck Analysis

1. **Face Detection vs. Landmark Prediction Breakdown**:
   - **HOG Face Bounding Box Detection**: Takes **{face_det_lat['mean']:.2f} ms** ({(face_det_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% of frame cycle).
   - **68-Point Landmark Regression**: Takes only **{lmk_lat['mean']:.2f} ms** ({(lmk_lat['mean'] / max(total_lat['mean'], 1e-6)) * 100.0:.1f}% of frame cycle) once the bounding box is localized.
   - **Observation**: Bounding box localization is significantly more compute-intensive than landmark regression. On resource-constrained edge devices (e.g. Raspberry Pi), face detection can be strided or tracked via correlation filter across adjacent frames.
2. **Subsystem Overhead**:
   - Geometric landmark processing (**Head Pose** and **EAR/MAR**) took **{pose_lat['mean'] + ear_lat['mean']:.3f} ms** combined, contributing virtually zero latency overhead.
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n[+] Saved Real-Time Benchmark Session Report to: {output_path}")

    def _print_terminal_summary(self) -> None:
        """Prints high-level session summary to standard output upon session completion."""
        total_frames = len(self.records)
        duration_sec = self.records[-1].relative_time_sec if total_frames > 0 else 0.0
        fps_vals = [r.fps for r in self.records if r.fps > 0]
        mean_fps = float(np.mean(fps_vals)) if fps_vals else 0.0
        mean_lat = float(np.mean([r.lat_total_ms for r in self.records])) if self.records else 0.0
        face_det_ms = float(np.mean([r.lat_face_detect_ms for r in self.records])) if self.records else 0.0
        lmk_ms = float(np.mean([r.lat_landmarks_ms for r in self.records if r.face_detected])) if self.records else 0.0
        dist_ms = float(np.mean([r.lat_distraction_ms for r in self.records])) if self.records else 0.0

        print("\n" + "=" * 70)
        print(f"  LIVE ADAS SESSION BENCHMARK SUMMARY [{self.session_id}]")
        print("=" * 70)
        print(f"  Duration: {duration_sec:.2f}s | Total Frames: {total_frames} | Mean FPS: {mean_fps:.1f} | Mean Latency: {mean_lat:.2f} ms")
        print(f"  Backend: {self.backend.value.upper()}")
        print(f"  Face Detect (HOG):   {face_det_ms:>6.2f} ms")
        print(f"  Landmarks (68 pts):  {lmk_ms:>6.2f} ms")
        print(f"  Distraction Engine:  {dist_ms:>6.2f} ms")
        print(f"  Face Visibility:     {sum(1 for r in self.records if r.face_detected) / max(total_frames, 1) * 100.0:.1f}%")
        print(f"  Distraction Time:    {sum(1 for r in self.records if r.is_distracted) / max(total_frames, 1) * 100.0:.1f}%")
        print(f"  Reports Saved to:    {self.output_dir}/")
        print("=" * 70 + "\n")
