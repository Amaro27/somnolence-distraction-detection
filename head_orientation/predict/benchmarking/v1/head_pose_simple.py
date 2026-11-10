"""
Head Pose Estimation — Pitch & Roll via 2D Facial Landmarks (dlib).

Estimates head pitch (droop / somnolence) and roll (lateral tilt) relative to
a calibrated neutral baseline using vertical facial landmark vectors.

Controls:
    q — Quit
    r — Re-calibrate baseline
"""

import csv
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime

import cv2
import dlib
import numpy as np

# ── Model & Camera Setup Candidates ────────────────────────────────────────

PREDICTOR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join(os.path.dirname(__file__), "..", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join(os.path.dirname(__file__), "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join("head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join("head_orientation", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join("predict", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join("input", "shape_predictor_68_face_landmarks.dat"),
]

# Key landmarks: right eye outer (36), left eye outer (45), nose tip (33), chin (8)
KEY_LANDMARKS = [36, 45, 33, 8]

# ── Parameters & Thresholds ────────────────────────────────────────────────

KP_PITCH = 0.18  # Anthropometric constant for pitch sensitivity

# Pitch thresholds (degrees): (-) is looking DOWN / drooping
PITCH_THRESH_MILD = -8.0
PITCH_THRESH_WARNING = -15.0

# Roll thresholds (degrees): absolute deviation from baseline
ROLL_THRESH_MILD = 8.0
ROLL_THRESH_WARNING = 18.0

SUSTAINED_FRAMES = 20    # Frames a condition must persist before alerting
CALIBRATION_FRAMES = 60  # Frames needed to establish neutral baseline (~2s)

LEVEL_COLORS = {
    "NORMAL":           (0, 255, 0),    # green
    "MILD_DROOP":       (0, 200, 255),  # yellow
    "FATIGUE_WARNING":  (0, 0, 255),    # red
    "MILD_TILT":        (0, 200, 255),  # yellow
    "SIGNIFICANT_TILT": (0, 0, 255),    # red
}


# ── Geometric Pose Estimation ──────────────────────────────────────────────

def build_face_frame(puntos):
    """Constructs 2D face coordinates and calculates roll from eye corners."""
    p36 = puntos[36].astype(np.float64)  # right eye outer corner
    p45 = puntos[45].astype(np.float64)  # left eye outer corner

    e_mid = (p36 + p45) / 2.0
    v_eyes = p45 - p36
    w_eyes = float(np.linalg.norm(v_eyes))

    if w_eyes < 1e-6:
        u_x = np.array([1.0, 0.0])
        u_y = np.array([0.0, 1.0])
        roll_deg = 0.0
    else:
        u_x = v_eyes / w_eyes
        u_y = np.array([-u_x[1], u_x[0]])  # downward perpendicular
        roll_deg = float(-np.degrees(np.arctan2(v_eyes[1], v_eyes[0])))

    return e_mid, u_x, u_y, w_eyes, roll_deg


def estimate_head_pose(puntos, baseline=None):
    """Computes pitch and roll angles relative to facial axes."""
    e_mid, u_x, u_y, w_eyes, roll_deg = build_face_frame(puntos)

    p33 = puntos[33].astype(np.float64)  # nose tip
    p8  = puntos[8].astype(np.float64)   # chin

    # Project vertical facial segments onto midline vector u_y
    d_top = float(np.dot(p33 - e_mid, u_y))
    d_bottom = float(np.dot(p8 - p33, u_y))
    h_face = max(d_top + d_bottom, 1e-6)

    rho = d_top / h_face

    if baseline is not None:
        delta_rho = rho - baseline["rho"]
        pitch_deg = float(np.degrees(np.arctan2(-delta_rho, KP_PITCH)))
        delta_roll = roll_deg - baseline["roll"]
    else:
        pitch_deg = 0.0
        delta_roll = 0.0

    return {
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "delta_roll": delta_roll,
        "rho": rho,
        "e_mid": e_mid,
        "u_x": u_x,
        "u_y": u_y,
        "w_eyes": w_eyes,
    }


# ── Calibration & Classification ───────────────────────────────────────────

class Calibrator:
    """Accumulates neutral pose measurements over N frames to set baseline."""

    def __init__(self, n_frames=CALIBRATION_FRAMES):
        self.n_frames = n_frames
        self.reset()

    def reset(self):
        self.rho_buf = []
        self.roll_buf = []
        self.baseline = None

    @property
    def is_calibrated(self):
        return self.baseline is not None

    @property
    def progress(self):
        return min(len(self.rho_buf) / self.n_frames, 1.0)

    def accumulate(self, metrics):
        self.rho_buf.append(metrics["rho"])
        self.roll_buf.append(metrics["roll_deg"])

        if len(self.rho_buf) >= self.n_frames:
            self.baseline = {
                "rho":  float(np.median(self.rho_buf)),
                "roll": float(np.median(self.roll_buf)),
            }
            return self.baseline
        return None


class FatigueClassifier:
    """Classifies pitch droop (somnolence) and roll tilt."""

    def __init__(self, sustained=SUSTAINED_FRAMES):
        self.sustained = sustained
        self.pitch_counter = 0
        self.roll_counter = 0

    def classify(self, pitch_deg, roll_deg, baseline):
        delta_roll = roll_deg - baseline["roll"]

        # Pitch droop classification
        if pitch_deg > PITCH_THRESH_MILD:
            self.pitch_counter = 0
            pitch_level = "NORMAL"
        elif pitch_deg > PITCH_THRESH_WARNING:
            self.pitch_counter += 1
            pitch_level = "MILD_DROOP" if self.pitch_counter > self.sustained else "NORMAL"
        else:
            self.pitch_counter += 1
            pitch_level = "FATIGUE_WARNING" if self.pitch_counter > self.sustained else "MILD_DROOP"

        # Roll tilt classification
        abs_droll = abs(delta_roll)
        if abs_droll < ROLL_THRESH_MILD:
            self.roll_counter = 0
            roll_level = "NORMAL"
        elif abs_droll < ROLL_THRESH_WARNING:
            self.roll_counter += 1
            roll_level = "MILD_TILT" if self.roll_counter > self.sustained else "NORMAL"
        else:
            self.roll_counter += 1
            roll_level = "SIGNIFICANT_TILT" if self.roll_counter > self.sustained else "MILD_TILT"

        return pitch_level, roll_level


# ── UI & Debug Overlay ─────────────────────────────────────────────────────

def draw_landmarks(frame, puntos):
    """Draw key landmarks (eye corners, nose tip, chin)."""
    for idx in KEY_LANDMARKS:
        cv2.circle(frame, tuple(puntos[idx]), 3, (0, 255, 255), -1)


def draw_face_axes(frame, metrics, pitch_level):
    """Draw face coordinate axes (eye horizontal and midline vertical)."""
    e_mid = metrics["e_mid"].astype(int)
    u_x = metrics["u_x"]
    u_y = metrics["u_y"]
    w = int(metrics["w_eyes"] * 0.5)

    # Eye line axis (cyan)
    pt_x = (int(e_mid[0] + u_x[0] * w), int(e_mid[1] + u_x[1] * w))
    cv2.arrowedLine(frame, tuple(e_mid), pt_x, (255, 255, 0), 2, tipLength=0.2)

    # Vertical midline axis (color by pitch status)
    color = LEVEL_COLORS.get(pitch_level, (0, 255, 0))
    pt_y = (int(e_mid[0] + u_y[0] * w * 1.5), int(e_mid[1] + u_y[1] * w * 1.5))
    cv2.arrowedLine(frame, tuple(e_mid), pt_y, color, 2, tipLength=0.2)


def draw_calibration_overlay(frame, progress):
    """Draw compact calibration progress bar."""
    bar_x, bar_y, bar_w, bar_h = 20, 25, 180, 14
    cv2.putText(frame, f"Calibrating: {int(progress * 100)}%",
                (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    fill_w = int(bar_w * progress)
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), (0, 230, 0), -1)


def draw_hud(frame, metrics, pitch_level, roll_level):
    """Draw simplified debug overlay with live pitch and roll values."""
    h, w = frame.shape[:2]
    pitch_deg = metrics["pitch_deg"]
    roll_deg = metrics["roll_deg"]

    pitch_color = LEVEL_COLORS.get(pitch_level, (255, 255, 255))
    roll_color = LEVEL_COLORS.get(roll_level, (255, 255, 255))

    cv2.putText(frame, f"Pitch: {pitch_deg:+5.1f} deg  [{pitch_level}]",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pitch_color, 2)
    cv2.putText(frame, f"Roll : {roll_deg:+5.1f} deg  [{roll_level}]",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, roll_color, 2)

    if pitch_level == "FATIGUE_WARNING":
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 255), -1)
        cv2.putText(frame, "FATIGUE DETECTED: HEAD DROOPING!",
                    (w // 2 - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


# ── Performance Benchmarking Collector ─────────────────────────────────────

class BenchmarkCollector:
    """Collects high-resolution execution timestamps and exports Event-Log & Summary CSVs."""

    def __init__(self, script_dir=None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
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
        events_path = os.path.join(self.script_dir, f"benchmark_events_{timestamp_str}.csv")
        summary_path = os.path.join(self.script_dir, f"benchmark_summary_{timestamp_str}.csv")

        # 1. Export Raw Events Log CSV
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

        # Sort summary rows by depth, then total_ms descending
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


def main():
    predictor_path = next((p for p in PREDICTOR_CANDIDATES if os.path.isfile(p)), None)
    if predictor_path is None:
        raise FileNotFoundError(
            "Could not locate 'shape_predictor_68_face_landmarks.dat'. "
            "Please check the input directory."
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(0)

    calibrator = Calibrator()
    classifier = FatigueClassifier()
    bench = BenchmarkCollector()

    print("[INFO] Starting head pose estimation. Press 'q' to quit, 'r' to recalibrate.")

    try:
        while True:
            with bench.measure("total_frame_cycle", depth=1):
                with bench.measure("camera_read", depth=1):
                    ret, frame = cap.read()
                if not ret:
                    break

                with bench.measure("frame_preprocess", depth=1):
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                with bench.measure("face_detector", depth=1):
                    rostros = detector(gray, 0)

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

                with bench.measure("gui_imshow", depth=1):
                    cv2.imshow("Head Pose Estimation (Pitch & Roll)", frame)

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
        cv2.destroyAllWindows()
        bench.export_csvs()


if __name__ == "__main__":
    main()
