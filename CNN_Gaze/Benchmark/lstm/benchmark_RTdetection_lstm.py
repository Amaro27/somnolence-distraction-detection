"""
Real-Time Temporal CNN-LSTM Driver Gaze Classifier (GazeResNetLSTM) — Performance Benchmarking Harness.

Executes a non-intrusive benchmark on the CNN-LSTM gaze classification pipeline,
measuring frame read, preprocessing (including CLAHE), spatial CNN feature extraction,
temporal LSTM sequence inference, postprocessing, GUI rendering, and total frame cycle metrics.

Features:
    - Auto-terminates after 1000 frames by default (configurable via --max-frames).
    - Event-Log / Tidy CSV export for variable-length records (ragged data).
    - Summary statistics CSV (count, mean, std, median, p95, min, max, total, % frame).
    - Supports webcam (default) or video file inputs for reproducible benchmarks.
    - Supports `--no-gui` for headless profiling.
    - Synchronizes CUDA if GPU is used to measure true kernel execution latency.

Controls:
    q — Quit and export CSVs
"""

import os
import sys

user_site = r'C:\Users\genyd\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages'
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import argparse
from collections import deque
from contextlib import contextmanager
import csv
from datetime import datetime
import math
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ==========================================
# 1. CLAHE TRANSFORM
# ==========================================
class CLAHETransform:
    def __init__(self, clip_limit=2.5, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        np_img = np.array(img.convert('L'))
        enhanced = self.clahe.apply(np_img)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_enhanced)


# ==========================================
# 2. MODEL ARCHITECTURE
# (Must match the trained model definition in RTdetection_lstm.py)
# ==========================================
class GazeResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, num_classes=2):
        super(GazeResNetLSTM, self).__init__()

        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes)
        )

    def extract_single_frame_feature(self, img_tensor):
        # img_tensor: (1, C, H, W)
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor)  # (1, 512, 1, 1)
        return feat.view(1, self.feature_dim)  # (1, 512)

    def forward_sequence_features(self, feat_sequence):
        # feat_sequence: (1, Seq_Len, 512)
        lstm_out, _ = self.lstm(feat_sequence)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)


# ── Benchmark Collector ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_candidate_path(custom_path, default_filename):
    """Resolve model checkpoint path across standard project locations."""
    candidate_paths = [
        custom_path,
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Modelos", default_filename)),
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", default_filename)),
        os.path.join("CNN_Gaze", "Modelos", default_filename),
        os.path.join("Modelos", default_filename),
        default_filename,
    ]
    return next((p for p in candidate_paths if p and os.path.isfile(p)), None)


def load_checkpoint_robustly(model, model_path, device):
    resolved_model_path = _resolve_candidate_path(model_path, "resnet18_lstm_unified.pth")
    if resolved_model_path is None:
        target = model_path or "resnet18_lstm_unified.pth"
        raise FileNotFoundError(
            f"Could not locate model file: '{target}'. "
            f"Please verify the model path or specify via --model-path."
        )

    ckpt = torch.load(resolved_model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    # Check if this is a unified checkpoint containing feature_extractor
    if any(k.startswith('feature_extractor.') for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=True)
        print(f"[+] Loaded unified CNN-LSTM model weights from '{resolved_model_path}'", flush=True)
    else:
        # Checkpoint only has LSTM and Classifier weights
        # 1. Load LSTM & Classifier
        model.load_state_dict(state_dict, strict=False)
        print(f"[+] Loaded LSTM & Classifier weights from '{resolved_model_path}'", flush=True)

        # 2. Automatically load ResNet18 backbone weights from trained CLAHE model
        backbone_path = _resolve_candidate_path(None, "resnet18_preprocess_clahe_full_model.pth")
        if backbone_path and os.path.exists(backbone_path):
            bb_state = torch.load(backbone_path, map_location=device, weights_only=False)
            unified_bb = {}
            for k, v in bb_state.items():
                if not k.startswith('fc.'):
                    if k.startswith('conv1.'):
                        new_k = k.replace('conv1.', 'feature_extractor.0.')
                    elif k.startswith('bn1.'):
                        new_k = k.replace('bn1.', 'feature_extractor.1.')
                    elif k.startswith('layer1.'):
                        new_k = k.replace('layer1.', 'feature_extractor.4.')
                    elif k.startswith('layer2.'):
                        new_k = k.replace('layer2.', 'feature_extractor.5.')
                    elif k.startswith('layer3.'):
                        new_k = k.replace('layer3.', 'feature_extractor.6.')
                    elif k.startswith('layer4.'):
                        new_k = k.replace('layer4.', 'feature_extractor.7.')
                    else:
                        new_k = f'feature_extractor.{k}'
                    unified_bb[new_k] = v
            model.load_state_dict(unified_bb, strict=False)
            print(f"[+] Loaded pretrained ResNet18 backbone weights from '{backbone_path}'", flush=True)
        else:
            print(
                "[!] Warning: 'resnet18_preprocess_clahe_full_model.pth' not found; "
                "backbone will use default uninitialized weights.",
                flush=True,
            )


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
# 3. BENCHMARK RUNNER & INFERENCE SETUP
# ==========================================
def run_benchmark(
    source="0",
    max_frames=1000,
    no_gui=False,
    output_dir=None,
    model_path=None,
    seq_len=20,
):
    IMG_SIZE = 224  # Match ResNet18 training resolution

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==================================================", flush=True)
    print(f"[*] Real-Time Temporal CNN-LSTM Gaze Benchmark on {device}", flush=True)
    print(f"[*] Temporal Window: {seq_len} consecutive frames", flush=True)
    print("==================================================", flush=True)

    # Initialize model architecture and load weights
    model = GazeResNetLSTM(hidden_dim=128, num_layers=2, num_classes=2).to(device)
    try:
        load_checkpoint_robustly(model, model_path, device)
    except Exception as e:
        print(f"[!] Error loading model: {e}", flush=True)
        return

    model.eval()

    # Pre-warm model (both CNN backbone and LSTM) to eliminate cold-start initialization latency
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)
            dummy_feat = model.extract_single_frame_feature(dummy)
            dummy_seq = dummy_feat.unsqueeze(0).repeat(1, seq_len, 1)
            _ = model.forward_sequence_features(dummy_seq)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    except Exception:
        pass

    # Preprocessing transforms (Must match RTdetection_lstm.py)
    clahe = CLAHETransform(clip_limit=2.5, tile_grid_size=(8, 8))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        clahe,
        transforms.ToTensor(),
        norm
    ])

    class_names = ['not_looking_road', 'looking_road']

    # FIFO queue of pre-extracted 512-dim features
    feature_buffer = deque(maxlen=seq_len)

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
        print(f"[!] Error: Could not open video source: {source_label}", flush=True)
        return

    bench = BenchmarkCollector(output_dir=output_dir)

    max_frames_label = f"{max_frames} (auto-quit)" if (max_frames and max_frames > 0) else "Unlimited"
    print(f"[INFO] Starting dedicated benchmark on: {source_label}", flush=True)
    print(f"[INFO] GUI enabled: {not no_gui} | Max frames: {max_frames_label}", flush=True)
    print("[INFO] Controls: Press 'q' in video window to exit early and export CSVs.", flush=True)

    # FPS Calculation variables
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            if max_frames and max_frames > 0 and bench.current_frame >= max_frames:
                print(f"\n[INFO] Auto-quitting: Reached target limit of {max_frames} frames.", flush=True)
                break

            with bench.measure("total_frame_cycle", depth=1):
                # 1. Camera / Video Frame Read
                with bench.measure("camera_read", depth=1):
                    ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream reached or failed to grab frame.", flush=True)
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
                        pil_img = Image.fromarray(rgb_frame)

                    with bench.measure("transform_and_tensor", depth=2):
                        input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # 3. Model Inference: Spatial CNN Feature Extraction + Temporal LSTM (Depth 1 & Depth 2)
                ##################################################
                ##################################################
                ##################################################
                t_infer_start = time.perf_counter()
                with bench.measure("model_inference", depth=1):
                    with torch.no_grad():
                        with bench.measure("cnn_feature_extraction", depth=2):
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                            feat = model.extract_single_frame_feature(input_tensor)
                            feature_buffer.append(feat)
                            if device.type == 'cuda':
                                torch.cuda.synchronize()

                        with bench.measure("lstm_sequence_forward", depth=2):
                            stacked_features = torch.cat(list(feature_buffer), dim=0).unsqueeze(0)  # (1, curr_len, 512)
                            outputs = model.forward_sequence_features(stacked_features)
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                infer_time_ms = (time.perf_counter() - t_infer_start) * 1000.0
                ##################################################
                ##################################################
                ##################################################
                # 4. Postprocessing (Softmax & Argmax)
                with bench.measure("postprocessing", depth=1):
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    label = class_names[pred_idx.item()]
                    conf_score = confidence.item() * 100.0

                # 5. UI Rendering & Drawing (Optional if not --no-gui)
                key = -1
                if not no_gui:
                    with bench.measure("draw_overlay", depth=1):
                        color = (0, 255, 0) if label == 'looking_road' else (0, 0, 255)
                        fps_text = f"FPS: {fps:.1f} | Latency: {infer_time_ms:.1f} ms"
                        status_text = f"LSTM ({len(feature_buffer)}/{seq_len}f): {label} ({conf_score:.1f}%)"

                        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

                    with bench.measure("gui_imshow", depth=1):
                        cv2.imshow('Real-Time Temporal CNN-LSTM Driver Gaze Detection', frame)

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
        description="Dedicated performance benchmark harness for RTdetection_lstm.py (GazeResNetLSTM)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or path to video file (.avi, .mov, .mp4). Default: '0'.",
    )
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        dest="max_frames",
        type=int,
        default=1000,
        help="Maximum number of frames to benchmark before auto-terminating (default: 1000). Set to 0 or negative for unlimited.",
    )
    parser.add_argument(
        "--no-gui",
        "--no_gui",
        dest="no_gui",
        action="store_true",
        help="Run in headless mode without cv2.imshow/waitKey to measure pure computation.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory where output CSVs will be saved. Default: directory of this script.",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        default=None,
        help="Path to saved PyTorch LSTM model checkpoint (.pth). Default: auto-detected resnet18_lstm_unified.pth.",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        dest="seq_len",
        type=int,
        default=20,
        help="Temporal window length in frames (default: 20).",
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
        seq_len=parsed.seq_len,
    )


if __name__ == '__main__':
    main()
