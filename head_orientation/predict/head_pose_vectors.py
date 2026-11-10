"""
Head Pose Estimation — Pitch, Roll & Yaw via 2D Facial Landmarks (dlib)
Corrected Algorithm (v4 — Yaw-Decoupled)

Reference:
    head_orientation/input/head_orientation_reference.png

Sign Conventions (from reference):
    PITCH:  (+) Looking UP,           (-) Looking DOWN / Drooping
    ROLL:   (+) Tilting Image-Left,   (-) Tilting Image-Right
    YAW:    (+) Turning Image-Left,   (-) Turning Image-Right

Why Yaw Was Mixed Up with Pitch in v3:
    In v3, vertical distance d_top was normalized by horizontal eye width W_eyes (eta = d_top / W_eyes).
    When rotating in yaw (angle psi), horizontal eye width foreshortens by cos(psi).
    Because W_eyes was in the denominator, eta artificially blew up by 1/cos(psi),
    falsely registering a ~15 degree head droop when the user simply turned their head sideways.

The Solution (Pure Vertical Ratio):
    Yaw rotates around the vertical axis u_y. By the rotation invariance theorem,
    the projection of ANY facial vector onto u_y is completely invariant to yaw!
    We normalize vertical eye-to-nose distance (d_top) by total vertical face height (H_face = d_top + d_bottom):
        rho = d_top / H_face = d_top / (d_top + d_bottom)
    Because BOTH numerator and denominator are vertical:
        1. Scale-invariant (scales with camera distance)
        2. Roll-invariant (projected along rotated u_y)
        3. 100% YAW-INVARIANT (neither contracts with yaw!)

Controls:
    q — quit
    r — re-calibrate (look straight for ~2 seconds)
"""

import cv2
import dlib
import numpy as np
import os

# ── dlib setup ──────────────────────────────────────────────────────────────

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    os.path.join("head_orientation", "input", "shape_predictor_68_face_landmarks.dat")
)

cap = cv2.VideoCapture(0)

# ── Landmark indices ────────────────────────────────────────────────────────

PUNTOS_ORIENTACION_DLIB = {
    "ojo_derecho":       36,  # right eye outer corner (image-left)
    "ojo_izquierdo":     45,  # left eye outer corner (image-right)
    "punta_nariz":       33,  # nose tip
    "barbilla":           8,  # chin
    "labio_derecho":     48,  # right mouth corner
    "labio_izquierdo":   54,  # left mouth corner
    "puente_nariz_sup":  27,  # nose bridge top
}

INDICES_PUNTOS_DLIB = list(PUNTOS_ORIENTACION_DLIB.values())

# ── Configurable parameters ─────────────────────────────────────────────────

# Anthropometric constant: nose protrusion depth relative to total face height
KP_PITCH = 0.18

# Yaw sensitivity constant for eye-nose asymmetry
KY_YAW = 0.60

# Pitch thresholds (degrees)
# (-) is looking DOWN / drooping, (+) is looking UP
PITCH_THRESH_MILD = -8.0         # droop > 8 deg down
PITCH_THRESH_WARNING = -15.0     # droop > 15 deg down

# Roll thresholds (degrees deviation from baseline)
ROLL_THRESH_MILD = 8.0
ROLL_THRESH_WARNING = 18.0

# Yaw threshold for detecting when driver is looking away (distraction)
YAW_THRESH_LOOKAWAY = 22.0

# Temporal filtering
SUSTAINED_FRAMES = 20    # frames a condition must persist before alert
CALIBRATION_FRAMES = 60  # ~2 seconds at 30 fps


# ── Geometric Face Frame & Pose Estimation ──────────────────────────────────

def build_face_frame(puntos):
    """
    Constructs an intrinsic orthonormal 2D coordinate system on the face
    using the outer eye corners.

    Returns
    -------
    E_mid : np.ndarray, shape (2,)
        Midpoint between outer eye corners.
    u_x : np.ndarray, shape (2,)
        Unit vector pointing along the eye line (image-left to image-right).
    u_y : np.ndarray, shape (2,)
        Unit vector pointing downward along the facial midline.
    W_eyes : float
        Inter-ocular distance in pixels.
    roll_deg : float
        Head roll angle in degrees.
        (+) Tilting image-left (subject's right ear down).
        (-) Tilting image-right (subject's left ear down).
    """
    P36 = puntos[36].astype(np.float64)  # right eye outer (image-left)
    P45 = puntos[45].astype(np.float64)  # left eye outer (image-right)

    E_mid = (P36 + P45) / 2.0
    v_eyes = P45 - P36
    W_eyes = float(np.linalg.norm(v_eyes))

    if W_eyes < 1e-6:
        u_x = np.array([1.0, 0.0])
        u_y = np.array([0.0, 1.0])
        roll_deg = 0.0
    else:
        u_x = v_eyes / W_eyes
        # Perpendicular vector pointing downwards in image coords (+y is down): [x, y] -> [-y, x]
        u_y = np.array([-u_x[1], u_x[0]])
        # Negative atan2 ensures:
        # Tilt image-left (P45 higher than P36, so P45.y < P36.y, dy < 0) -> positive (+)
        # Tilt image-right (P45 lower than P36, so P45.y > P36.y, dy > 0) -> negative (-)
        roll_deg = float(-np.degrees(np.arctan2(v_eyes[1], v_eyes[0])))

    return E_mid, u_x, u_y, W_eyes, roll_deg


def estimate_head_pose(puntos, baseline=None):
    """
    Computes decoupled pitch, roll, and yaw angles from 2D landmarks.

    Parameters
    ----------
    puntos : np.ndarray, shape (68, 2)
        68 facial landmarks.
    baseline : dict or None
        Calibrated baseline dictionary with keys 'rho', 'roll', and 'asym'.

    Returns
    -------
    metrics : dict
        Contains 'pitch_deg', 'roll_deg', 'yaw_deg', 'rho', 'H_face', etc.
    """
    E_mid, u_x, u_y, W_eyes, roll_deg = build_face_frame(puntos)

    P36 = puntos[36].astype(np.float64)  # right eye outer (image-left)
    P45 = puntos[45].astype(np.float64)  # left eye outer (image-right)
    P33 = puntos[33].astype(np.float64)  # nose tip
    P8  = puntos[8].astype(np.float64)   # chin

    # ── PITCH via Pure Vertical Ratio (Yaw-Invariant) ──
    # Project eye-to-nose and nose-to-chin onto the face vertical axis u_y
    d_top = float(np.dot(P33 - E_mid, u_y))       # vertical distance: eyes to nose
    d_bottom = float(np.dot(P8 - P33, u_y))      # vertical distance: nose to chin
    H_face = d_top + d_bottom                    # total face height along u_y

    # Pure vertical ratio (completely independent of horizontal eye contraction during yaw)
    rho = d_top / max(H_face, 1e-6)

    # ── YAW via Bilateral Eye-Nose Asymmetry ──
    dist_R = float(np.linalg.norm(P33 - P36))     # nose to right eye corner (image-left)
    dist_L = float(np.linalg.norm(P33 - P45))     # nose to left eye corner (image-right)
    asym_yaw = (dist_L - dist_R) / max(dist_L + dist_R, 1e-6)
    # When turning image-left: nose moves left -> dist_L > dist_R -> asym_yaw > 0 (+)
    # When turning image-right: nose moves right -> dist_R > dist_L -> asym_yaw < 0 (-)

    if baseline is not None:
        # Pitch: delta_rho = rho - rho_baseline
        # Looking DOWN -> nose drops -> d_top increases -> delta_rho > 0 -> pitch_deg < 0 (-)
        # Looking UP   -> nose rises -> d_top shrinks   -> delta_rho < 0 -> pitch_deg > 0 (+)
        delta_rho = rho - baseline["rho"]
        pitch_deg = float(np.degrees(np.arctan2(-delta_rho, KP_PITCH)))

        # Yaw: delta_asym = asym - asym_baseline
        delta_asym = asym_yaw - baseline["asym"]
        yaw_deg = float(np.degrees(np.arcsin(np.clip(delta_asym / KY_YAW, -1.0, 1.0))))

        delta_roll = roll_deg - baseline["roll"]
    else:
        pitch_deg = 0.0
        yaw_deg = 0.0
        delta_roll = 0.0

    return {
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "yaw_deg": yaw_deg,
        "delta_roll": delta_roll,
        "rho": rho,
        "asym_yaw": asym_yaw,
        "d_top": d_top,
        "d_bottom": d_bottom,
        "H_face": H_face,
        "W_eyes": W_eyes,
        "E_mid": E_mid,
        "u_x": u_x,
        "u_y": u_y,
    }


# ── Calibration ─────────────────────────────────────────────────────────────

class Calibrator:
    """Accumulates neutral pose measurements over N frames to set a personal baseline."""

    def __init__(self, n_frames=CALIBRATION_FRAMES):
        self.n_frames = n_frames
        self.reset()

    def reset(self):
        self.rho_buf = []
        self.roll_buf = []
        self.asym_buf = []
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
        self.asym_buf.append(metrics["asym_yaw"])

        if len(self.rho_buf) >= self.n_frames:
            self.baseline = {
                "rho":  float(np.median(self.rho_buf)),
                "roll": float(np.median(self.roll_buf)),
                "asym": float(np.median(self.asym_buf)),
            }
            return self.baseline
        return None


# ── Classification ──────────────────────────────────────────────────────────

class FatigueClassifier:
    """
    Classifies pitch (looking down = droop), roll tilt, and yaw (looking away).
    Distinguishes lateral head turn (distraction) from head drooping (somnolence).
    """

    def __init__(self, sustained=SUSTAINED_FRAMES):
        self.sustained = sustained
        self.pitch_counter = 0
        self.roll_counter = 0
        self.yaw_counter = 0

    def classify(self, pitch_deg, roll_deg, yaw_deg, baseline):
        delta_roll = roll_deg - baseline["roll"]

        # ── Yaw classification (looking away / distraction) ──
        abs_yaw = abs(yaw_deg)
        if abs_yaw > YAW_THRESH_LOOKAWAY:
            self.yaw_counter += 1
            yaw_level = "LOOKING_AWAY" if self.yaw_counter > self.sustained else "TURNING"
        else:
            self.yaw_counter = 0
            yaw_level = "FORWARD"

        # ── Pitch classification (focus on downward droop) ──
        # If user is actively turning head in yaw (> 22 deg), do not false-trigger somnolence
        if abs_yaw > YAW_THRESH_LOOKAWAY:
            self.pitch_counter = 0
            pitch_level = "NORMAL"
        else:
            if pitch_deg > PITCH_THRESH_MILD:  # > -8 deg (upright / normal)
                self.pitch_counter = 0
                pitch_level = "NORMAL"
            elif pitch_deg > PITCH_THRESH_WARNING:  # between -8 deg and -15 deg
                self.pitch_counter += 1
                pitch_level = "MILD_DROOP" if self.pitch_counter > self.sustained else "NORMAL"
            else:  # <= -15 deg (pronounced drooping)
                self.pitch_counter += 1
                pitch_level = "FATIGUE_WARNING" if self.pitch_counter > self.sustained else "MILD_DROOP"

        # ── Roll classification (lateral tilt) ──
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

        return pitch_level, roll_level, yaw_level, delta_roll


# ── Visualization Helpers ───────────────────────────────────────────────────

LEVEL_COLORS = {
    "NORMAL":           (0, 255, 0),    # green
    "FORWARD":          (0, 255, 0),
    "TURNING":          (0, 200, 255),
    "LOOKING_AWAY":     (0, 165, 255),  # orange
    "MILD_DROOP":       (0, 200, 255),  # yellow
    "FATIGUE_WARNING":  (0, 0, 255),    # red
    "MILD_TILT":        (0, 200, 255),
    "SIGNIFICANT_TILT": (0, 0, 255),
}


def draw_face_axes(frame, metrics, pitch_level):
    """Draw the face-aligned intrinsic coordinate frame and nose bridge."""
    E_mid = metrics["E_mid"].astype(int)
    u_x = metrics["u_x"]
    u_y = metrics["u_y"]
    W = int(metrics["W_eyes"] * 0.5)

    # Eye line axis (u_x) in cyan
    pt_x = (int(E_mid[0] + u_x[0] * W), int(E_mid[1] + u_x[1] * W))
    cv2.arrowedLine(frame, tuple(E_mid), pt_x, (255, 255, 0), 2, tipLength=0.2)

    # Facial vertical spine (u_y) in white/color based on pitch status
    color = LEVEL_COLORS.get(pitch_level, (0, 255, 0))
    pt_y = (int(E_mid[0] + u_y[0] * W * 1.5), int(E_mid[1] + u_y[1] * W * 1.5))
    cv2.arrowedLine(frame, tuple(E_mid), pt_y, color, 2, tipLength=0.2)


def draw_landmarks(frame, puntos):
    """Draw target landmarks."""
    for idx in INDICES_PUNTOS_DLIB:
        pt = tuple(puntos[idx])
        cv2.circle(frame, pt, 4, (0, 255, 255), -1)


def draw_calibration_overlay(frame, progress):
    """Draw calibration progress bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "CALIBRATING — Look straight at camera",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    bar_x, bar_y, bar_w, bar_h = 20, 42, w - 80, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    fill_w = int(bar_w * progress)
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), (0, 230, 0), -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w + 10, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hud(frame, metrics, pitch_level, roll_level, yaw_level, delta_roll):
    """Draw HUD with live pitch, roll, and yaw values matching reference conventions."""
    h, w = frame.shape[:2]
    pitch_deg = metrics["pitch_deg"]
    roll_deg = metrics["roll_deg"]
    yaw_deg = metrics["yaw_deg"]

    # Status lines matching head_orientation_reference.png
    lines = [
        f"PITCH: {pitch_deg:+6.1f} deg  [{pitch_level}]  (+ up, - droop)",
        f"ROLL : {roll_deg:+6.1f} deg  [{roll_level}]  (+ tilt left, - tilt right)",
        f"YAW  : {yaw_deg:+6.1f} deg  [{yaw_level}]  (+ turn left, - turn right)",
        f"rho: {metrics['rho']:.3f} | H_face: {metrics['H_face']:.0f}px",
    ]

    y0 = h - 25 * len(lines) - 15
    for i, line in enumerate(lines):
        y = y0 + i * 25
        # Background shadow for clarity
        cv2.putText(frame, line, (17, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

        if i == 0:
            color = LEVEL_COLORS.get(pitch_level, (255, 255, 255))
        elif i == 1:
            color = LEVEL_COLORS.get(roll_level, (255, 255, 255))
        elif i == 2:
            color = LEVEL_COLORS.get(yaw_level, (255, 255, 255))
        else:
            color = (200, 200, 200)

        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    # Flashing alert banner on severe fatigue
    if pitch_level == "FATIGUE_WARNING":
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, "FATIGUE DETECTED: HEAD DROOPING!",
                    (w // 2 - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    elif yaw_level == "LOOKING_AWAY":
        cv2.rectangle(frame, (0, 0), (w, 35), (0, 140, 255), -1)
        cv2.putText(frame, "DISTRACTION: HEAD TURNED AWAY",
                    (w // 2 - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


# ── Main loop ───────────────────────────────────────────────────────────────

calibrator = Calibrator()
classifier = FatigueClassifier()

print("[INFO] Starting head pose estimation loop. Press 'q' to quit, 'r' to recalibrate.")

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
            # Calibration phase: compute raw metrics without baseline
            metrics = estimate_head_pose(puntos, baseline=None)
            calibrator.accumulate(metrics)
            draw_face_axes(frame, metrics, "NORMAL")
            draw_calibration_overlay(frame, calibrator.progress)
        else:
            # Active tracking phase
            metrics = estimate_head_pose(puntos, baseline=calibrator.baseline)
            pitch_lvl, roll_lvl, yaw_lvl, d_roll = classifier.classify(
                metrics["pitch_deg"], metrics["roll_deg"], metrics["yaw_deg"], calibrator.baseline
            )
            draw_face_axes(frame, metrics, pitch_lvl)
            draw_hud(frame, metrics, pitch_lvl, roll_lvl, yaw_lvl, d_roll)

    cv2.imshow("Head Pose Estimation (Pitch, Roll, Yaw)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        calibrator.reset()
        classifier.pitch_counter = 0
        classifier.roll_counter = 0
        classifier.yaw_counter = 0

cap.release()
cv2.destroyAllWindows()
