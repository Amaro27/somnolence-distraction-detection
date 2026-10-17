"""
Head Pose Estimation — Pitch & Roll via 2D Facial Landmarks (dlib).

Estimates head pitch (droop / somnolence) and roll (lateral tilt) relative to
a calibrated neutral baseline using vertical facial landmark vectors.

Controls:
    q — Quit
    r — Re-calibrate baseline
"""

import os
import cv2
import dlib
import numpy as np

# ── Model & Camera Setup Candidates ────────────────────────────────────────

PREDICTOR_CANDIDATES = [
    os.path.join("head_orientation", "predict", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join("head_orientation", "input", "shape_predictor_68_face_landmarks.dat"),
    os.path.join(os.path.dirname(__file__), "input", "shape_predictor_68_face_landmarks.dat"),
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

    print("[INFO] Starting head pose estimation. Press 'q' to quit, 'r' to recalibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detector(gray, 0)

        for rostro in rostros:
            shape = predictor(gray, rostro)
            puntos = np.zeros((68, 2), dtype="int")
            for i in range(68):
                puntos[i] = (shape.part(i).x, shape.part(i).y)

            draw_landmarks(frame, puntos)

            if not calibrator.is_calibrated:
                metrics = estimate_head_pose(puntos, baseline=None)
                calibrator.accumulate(metrics)
                draw_face_axes(frame, metrics, "NORMAL")
                draw_calibration_overlay(frame, calibrator.progress)
            else:
                metrics = estimate_head_pose(puntos, baseline=calibrator.baseline)
                pitch_lvl, roll_lvl = classifier.classify(
                    metrics["pitch_deg"], metrics["roll_deg"], calibrator.baseline
                )
                draw_face_axes(frame, metrics, pitch_lvl)
                draw_hud(frame, metrics, pitch_lvl, roll_lvl)

        cv2.imshow("Head Pose Estimation (Pitch & Roll)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            calibrator.reset()
            classifier.pitch_counter = 0
            classifier.roll_counter = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
