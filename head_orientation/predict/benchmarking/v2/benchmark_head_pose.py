"""
Head Pose Simple — Dedicated Performance Benchmarking Harness (v2).

Executes a non-intrusive benchmark on the head pose estimation pipeline,
importing domain logic directly from `head_orientation.predict.head_pose_simple`
without modifying a single line of production code.

Features:
    - Zero modification to production `head_pose_simple.py`.
    - Auto-terminates after 1000 frames by default (configurable via --max-frames).
    - Event-Log / Tidy CSV export for variable-length records (ragged data).
    - Summary statistics CSV (count, mean, median, p95, min, max, % frame).
    - Supports webcam (default) or video file inputs for reproducible benchmarks.
    - Supports `--no-gui` for headless profiling.

Controls:
    q — Quit and export CSVs
    r — Re-calibrate baseline
"""

import argparse
import csv
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

# ── Dynamic Path Resolution ──────────────────────────────────────────────────
# Resolve paths so this runner can be called from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ../.. -> head_orientation/predict
PREDICT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
# ../../../.. -> repository root
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

for path in [PREDICT_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import dlib
import numpy as np

# Import pure production components from head_pose_simple
from head_orientation.predict.head_pose_simple import (
    PREDICTOR_CANDIDATES,
    Calibrator,
    FatigueClassifier,
    draw_calibration_overlay,
    draw_face_axes,
    draw_hud,
    draw_landmarks,
    estimate_head_pose,
)


# ── Benchmark Collector ──────────────────────────────────────────────────────

class BenchmarkCollector:
    """Collects high-resolution execution timestamps and exports Event-Log & Summary CSVs."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or SCRIPT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.records = []  # Tuples: (timestamp, frame_id, face_id, scope, depth, elapsed_ms, context)
        self.current_frame = 0

    @contextmanager
    def measure(self, scope_name, depth=1, face_id=-1, context=""):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            iso_ts = datetime.now().isoformat(timespec="milliseconds")
            self.records.append((
                iso_ts,
                self.current_frame,
                face_id,
                scope_name,
                depth,
                elapsed_ms,
                context,
            ))

    def export_csvs(self):
        if not self.records:
            print("[BENCHMARK] No measurements recorded.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        events_path = os.path.join(self.output_dir, f"benchmark_events_{timestamp_str}.csv")
        summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp_str}.csv")

        # 1. Export Raw Events Log CSV (Tidy format, handles ragged data cleanly)
        with open(events_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["timestamp", "frame_id", "face_id", "scope_name", "depth", "elapsed_ms", "context"])
            writer.writerows(self.records)

        # 2. Compute Summary Statistics by (scope_name, depth, context)
        grouped = {}
        total_frame_cycle_ms = 0.0

        for r in self.records:
            _, frame_id, face_id, scope_name, depth, elapsed_ms, context = r
            key = (scope_name, depth, context)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(elapsed_ms)
            if scope_name == "total_frame_cycle":
                total_frame_cycle_ms += elapsed_ms

        summary_rows = []
        for (scope_name, depth, context), vals in grouped.items():
            n = len(vals)
            mean_v = sum(vals) / n
            var_v = sum((x - mean_v) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_v = math.sqrt(var_v)
            sorted_v = sorted(vals)
            med_v = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
            p95_idx = min(int(n * 0.95), n - 1)
            p95_v = sorted_v[p95_idx]
            min_v = sorted_v[0]
            max_v = sorted_v[-1]
            tot_v = sum(vals)
            pct_frame = (tot_v / total_frame_cycle_ms * 100.0) if total_frame_cycle_ms > 0 else 0.0

            summary_rows.append({
                "scope_name": scope_name,
                "depth": depth,
                "context": context,
                "count": n,
                "mean_ms": round(mean_v, 3),
                "std_ms": round(std_v, 3),
                "median_ms": round(med_v, 3),
                "p95_ms": round(p95_v, 3),
                "min_ms": round(min_v, 3),
                "max_ms": round(max_v, 3),
                "total_ms": round(tot_v, 3),
                "pct_of_total_frame": round(pct_frame, 2),
            })

        # Sort summary rows: depth first, then total time descending
        summary_rows.sort(key=lambda x: (x["depth"], -x["total_ms"]))

        with open(summary_path, "w", newline="", encoding="utf-8") as fp:
            fields = [
                "scope_name", "depth", "context", "count", "mean_ms", "std_ms",
                "median_ms", "p95_ms", "min_ms", "max_ms", "total_ms", "pct_of_total_frame"
            ]
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n[BENCHMARK] Events log saved to: {events_path}")
        print(f"[BENCHMARK] Summary stats saved to: {summary_path}")
        self._print_terminal_summary(summary_rows)

    def _print_terminal_summary(self, summary_rows):
        print("\n" + "=" * 98)
        print("                            BENCHMARK PERFORMANCE SUMMARY")
        print("=" * 98)
        header = (
            f"{'Scope Name':<24} | {'Depth':<5} | {'Context':<13} | "
            f"{'Count':<6} | {'Mean(ms)':<8} | {'p95(ms)':<8} | {'Total(s)':<8} | {'% Frame':<7}"
        )
        print(header)
        print("-" * 98)
        for r in summary_rows:
            tot_s = r["total_ms"] / 1000.0
            pct_str = f"{r['pct_of_total_frame']:.1f}%" if r["scope_name"] != "total_frame_cycle" else "100.0%"
            print(
                f"{r['scope_name']:<24} | {r['depth']:<5} | {r['context']:<13} | "
                f"{r['count']:<6} | {r['mean_ms']:<8.2f} | {r['p95_ms']:<8.2f} | {tot_s:<8.2f} | {pct_str:<7}"
            )
        print("=" * 98 + "\n")


# ── Benchmark Runner ─────────────────────────────────────────────────────────

def run_benchmark(source="0", max_frames=1000, no_gui=False, output_dir=None):
    # Locate facial landmarks predictor
    candidate_paths = [
        os.path.join(PREDICT_DIR, "input", "shape_predictor_68_face_landmarks.dat"),
    ] + PREDICTOR_CANDIDATES

    predictor_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if predictor_path is None:
        raise FileNotFoundError(
            "Could not locate 'shape_predictor_68_face_landmarks.dat'. "
            "Please check the input directory."
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Determine video source (camera index or file path)
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
        source_label = f"Webcam ({source})"
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
        source_label = f"Video ({os.path.basename(source)})"

    calibrator = Calibrator()
    classifier = FatigueClassifier()
    bench = BenchmarkCollector(output_dir=output_dir)

    max_frames_label = f"{max_frames} (auto-quit)" if (max_frames and max_frames > 0) else "Unlimited"
    print(f"[INFO] Starting dedicated benchmark on: {source_label}")
    print(f"[INFO] GUI enabled: {not no_gui} | Max frames: {max_frames_label}")
    print("[INFO] Controls: Press 'q' in video window to exit early, 'r' to recalibrate.")

    try:
        while True:
            if max_frames and max_frames > 0 and bench.current_frame >= max_frames:
                print(f"\n[INFO] Auto-quitting: Reached target limit of {max_frames} frames.")
                break

            with bench.measure("total_frame_cycle", depth=1):
                # 1. Camera / Video Frame Read
                with bench.measure("camera_read", depth=1):
                    ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream reached.")
                    break

                # 2. Frame Preprocessing
                with bench.measure("frame_preprocess", depth=1):
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 3. Face Detection (Dlib HOG)
                with bench.measure("face_detector", depth=1):
                    rostros = detector(gray, 0)

                # 4. Face Processing Loop (Depth 2)
                for face_id, rostro in enumerate(rostros):
                    with bench.measure("shape_predictor", depth=2, face_id=face_id):
                        shape = predictor(gray, rostro)

                    with bench.measure("landmarks_array_init", depth=2, face_id=face_id):
                        puntos = np.zeros((68, 2), dtype="int")
                        for i in range(68):
                            puntos[i] = (shape.part(i).x, shape.part(i).y)

                    with bench.measure("draw_landmarks", depth=2, face_id=face_id):
                        draw_landmarks(frame, puntos)

                    if not calibrator.is_calibrated:
                        with bench.measure("estimate_head_pose", depth=2, face_id=face_id, context="uncalibrated"):
                            metrics = estimate_head_pose(puntos, baseline=None)

                        with bench.measure("calibrator_accumulate", depth=2, face_id=face_id):
                            calibrator.accumulate(metrics)

                        with bench.measure("draw_face_axes", depth=2, face_id=face_id, context="calibrating"):
                            draw_face_axes(frame, metrics, "NORMAL")

                        with bench.measure("draw_calibration_bar", depth=2, face_id=face_id):
                            draw_calibration_overlay(frame, calibrator.progress)
                    else:
                        with bench.measure("estimate_head_pose", depth=2, face_id=face_id, context="calibrated"):
                            metrics = estimate_head_pose(puntos, baseline=calibrator.baseline)

                        with bench.measure("fatigue_classify", depth=2, face_id=face_id):
                            pitch_lvl, roll_lvl = classifier.classify(
                                metrics["pitch_deg"], metrics["roll_deg"], calibrator.baseline
                            )

                        with bench.measure("draw_face_axes", depth=2, face_id=face_id, context="tracking"):
                            draw_face_axes(frame, metrics, pitch_lvl)

                        with bench.measure("draw_hud", depth=2, face_id=face_id):
                            draw_hud(frame, metrics, pitch_lvl, roll_lvl)

                # 5. Optional GUI Rendering
                key = -1
                if not no_gui:
                    with bench.measure("gui_imshow", depth=1):
                        cv2.imshow("Head Pose Benchmark (Pitch & Roll)", frame)

                    with bench.measure("gui_waitkey", depth=1):
                        key = cv2.waitKey(1) & 0xFF

            bench.current_frame += 1

            if key == ord('q'):
                break
            elif key == ord('r'):
                calibrator.reset()
                classifier.pitch_counter = 0
                classifier.roll_counter = 0

    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        bench.export_csvs()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dedicated non-intrusive benchmark harness for head_pose_simple.py"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to video file (.avi, .mov, .mp4). Default: '0'.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum number of frames to benchmark before auto-terminating (default: 1000). Set to 0 or negative for unlimited.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without cv2.imshow/waitKey to measure pure CV computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where output CSVs will be saved. Default: directory of this script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        source=args.source,
        max_frames=args.max_frames,
        no_gui=args.no_gui,
        output_dir=args.output_dir,
    )
