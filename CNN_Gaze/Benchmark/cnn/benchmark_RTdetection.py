"""
Real-Time Gaze Classification (LightGazeCNN) — Performance Benchmarking Harness.

Executes a non-intrusive benchmark on the CNN gaze classification pipeline,
measuring preprocessing, model inference, postprocessing, and frame cycle metrics.

Features:
    - Auto-terminates after 1000 frames by default (configurable via --max-frames).
    - Event-Log / Tidy CSV export for variable-length records (ragged data).
    - Summary statistics CSV (count, mean, median, p95, min, max, % frame).
    - Supports webcam (default) or video file inputs for reproducible benchmarks.
    - Supports `--no-gui` for headless profiling.
    - Synchronizes CUDA if GPU is used to measure true kernel execution latency.

Controls:
    q — Quit and export CSVs
"""

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ==========================================
# 1. LIGHTWEIGHT CNN ARCHITECTURE
# (Must match the trained model definition)
# ==========================================
class LightGazeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LightGazeCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Benchmark Collector ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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


# ==========================================
# 2. BENCHMARK RUNNER & INFERENCE SETUP
# ==========================================
def run_benchmark(
    source="0",
    max_frames=1000,
    no_gui=False,
    output_dir=None,
    model_path=None,
):
    # Candidate paths for model weights
    candidate_paths = [
        model_path,
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Modelos", "light_cnn_session_full_model.pth")),
        os.path.join("CNN_Gaze", "Modelos", "light_cnn_session_full_model.pth"),
        os.path.join("Modelos", "light_cnn_session_full_model.pth"),
        "light_cnn_session_full_model.pth",
    ]
    resolved_model_path = next((p for p in candidate_paths if p and os.path.isfile(p)), None)

    if resolved_model_path is None:
        target = model_path or "light_cnn_session_full_model.pth"
        raise FileNotFoundError(
            f"Could not locate model file: '{target}'. "
            f"Please verify the model path or specify via --model-path."
        )

    IMG_SIZE = 112  # Match training resolution
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on device: {device}")

    # Initialize model architecture and load weights
    model = LightGazeCNN(num_classes=2).to(device)
    
    try:
        checkpoint = torch.load(resolved_model_path, map_location=device)
        # Handle full checkpoint dict or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[+] Successfully loaded model weights from '{resolved_model_path}'")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    model.eval()

    # Pre-warm model to eliminate cold-start initialization latency
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    except Exception:
        pass

    # Preprocessing transforms (Must match validation/evaluation transforms)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ['not_looking_road', 'looking_road']

    # Determine video source (camera index or file path)
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
        source_label = f"Webcam ({source})"
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
        source_label = f"Video ({os.path.basename(source)})"

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {source_label}")
        return

    bench = BenchmarkCollector(output_dir=output_dir)

    max_frames_label = f"{max_frames} (auto-quit)" if (max_frames and max_frames > 0) else "Unlimited"
    print(f"[INFO] Starting dedicated benchmark on: {source_label}")
    print(f"[INFO] GUI enabled: {not no_gui} | Max frames: {max_frames_label}")
    print("[INFO] Controls: Press 'q' in video window to exit early and export CSVs.")

    # FPS Calculation variables
    prev_time = time.time()
    fps = 0.0

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
                    print("[INFO] End of video stream reached or failed to grab frame.")
                    break

                # Calculate display FPS
                curr_time = time.time()
                time_diff = curr_time - prev_time
                if time_diff > 0:
                    fps = 1.0 / time_diff
                prev_time = curr_time

                # 2. Preprocessing (Depth 1 with Depth 2 breakdown)
                with bench.measure("frame_preprocess", depth=1):
                    with bench.measure("color_and_pil_convert", depth=2):
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)

                    with bench.measure("transform_and_tensor", depth=2):
                        input_tensor = transform(pil_image).unsqueeze(0).to(device)

                # 3. Model Inference (Depth 1)
                with bench.measure("model_inference", depth=1):
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                # 4. Postprocessing (Softmax & Argmax)
                with bench.measure("postprocessing", depth=1):
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    label = class_names[predicted_idx.item()]
                    conf_score = confidence.item() * 100

                # 5. UI Rendering & Drawing (Optional if not --no-gui)
                key = -1
                if not no_gui:
                    with bench.measure("draw_overlay", depth=1):
                        color = (0, 255, 0) if label == 'looking_road' else (0, 0, 255)
                        display_text = f"{label} ({conf_score:.1f}%)"
                        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(frame, display_text, (20, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    with bench.measure("gui_imshow", depth=1):
                        cv2.imshow('Real-Time Gaze Classification', frame)

                    with bench.measure("gui_waitkey", depth=1):
                        key = cv2.waitKey(1) & 0xFF

            bench.current_frame += 1

            if key == ord('q'):
                break

    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        bench.export_csvs()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Dedicated performance benchmark harness for RTdetection.py (LightGazeCNN)"
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
        help="Run in headless mode without cv2.imshow/waitKey to measure pure computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where output CSVs will be saved. Default: directory of this script.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved PyTorch model checkpoint (.pth).",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    run_benchmark(
        source=parsed.source,
        max_frames=parsed.max_frames,
        no_gui=parsed.no_gui,
        output_dir=parsed.output_dir,
        model_path=parsed.model_path,
    )


if __name__ == '__main__':
    main()
