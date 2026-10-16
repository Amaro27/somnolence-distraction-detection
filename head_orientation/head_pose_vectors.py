"""
Head Pose Estimation — Pitch & Roll via 2D Facial Landmarks (dlib)
Corrected Algorithm (v3)

Why the previous version failed:
    In the previous version, the 2D angle between the nose bridge (P27→P33)
    and chin line (P27→P8) was used. In 2D, these points all lie on the
    vertical midline; nodding up/down (pitch) keeps them collinear (~0°),
    while turning left/right (yaw) causes the nose tip to swing sideways.
    Thus, that formula was computing YAW, not pitch!

Corrected Geometric Approach:
    1. FACE-ALIGNED COORDINATE SYSTEM:
       We build an intrinsic 2D coordinate frame {u_x, u_y} from the eye line (P36→P45).
       - u_x: unit vector along eye line (measures ROLL via atan2).
       - u_y: perpendicular unit vector pointing straight down the facial midline.
       Because all measurements are projected onto u_y, pitch is 100% IMMUNE TO ROLL!

    2. SAGITTAL PITCH ESTIMATION:
       In 3D, the nose tip (P33) protrudes forward in depth (Z > 0).
       When the head rotates in pitch around the transverse axis:
       - Tilting DOWN (droop): the nose tip drops along u_y (distance from eye to nose increases).
       - Tilting UP: the nose tip rises along u_y (distance from eye to nose shrinks).
       Projecting onto u_y:
           d_top = (P33 - E_midpoint) · u_y
           eta   = d_top / W_eyes  (normalized by inter-ocular distance)
       The pitch angle is computed via the sagittal right triangle:
           theta_pitch = -arctan((eta - eta_baseline) / k_p) * (180 / pi)
       Sign convention matches reference:
           (+) Looking UP
           (-) Looking DOWN / drooping

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
    "ojo_derecho":       36,  # right eye outer corner
    "ojo_izquierdo":     45,  # left eye outer corner
    "punta_nariz":       33,  # nose tip
    "barbilla":           8,  # chin
    "labio_derecho":     48,  # right mouth corner
    "labio_izquierdo":   54,  # left mouth corner
    "puente_nariz_sup":  27,  # nose bridge top
}

INDICES_PUNTOS_DLIB = list(PUNTOS_ORIENTACION_DLIB.values())

# ── Configurable parameters ─────────────────────────────────────────────────

# Anthropometric constant: nose depth protrusion relative to inter-ocular width
KP_PITCH = 0.35

# Pitch thresholds (degrees)
# (-) is looking DOWN / drooping, (+) is looking UP
PITCH_THRESH_MILD = -8.0         # droop > 8 deg down
PITCH_THRESH_WARNING = -15.0     # droop > 15 deg down

# Roll thresholds (degrees deviation from baseline)
ROLL_THRESH_MILD = 8.0
ROLL_THRESH_WARNING = 18.0

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
        Unit vector pointing along the eye line (left to right from subject's POV).
    u_y : np.ndarray, shape (2,)
        Unit vector pointing downward along the facial midline.
    W_eyes : float
        Inter-ocular distance in pixels (pitch-invariant scale factor).
    roll_deg : float
        Head roll angle in degrees relative to camera horizon.
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
        # Perpendicular vector pointing downwards (rotates u_x by 90 deg clockwise)
        # in image coords where +y is down: [x, y] -> [-y, x]
        u_y = np.array([-u_x[1], u_x[0]])
        roll_deg = float(np.degrees(np.arctan2(v_eyes[1], v_eyes[0])))

    return E_mid, u_x, u_y, W_eyes, roll_deg


def estimate_head_pose(puntos, baseline=None):
    """
    Computes pitch and roll angles along with intermediate geometric metrics.

    Parameters
    ----------
    puntos : np.ndarray, shape (68, 2)
        68 facial landmarks.
    baseline : dict or None
        Calibrated baseline dictionary with keys 'eta' and 'roll'.

    Returns
    -------
    metrics : dict
        Contains:
          - 'pitch_deg': estimated pitch angle in degrees (+ up, - down)
          - 'roll_deg': current roll angle in degrees
          - 'eta': current normalized eye-to-nose vertical distance
          - 'd_top': raw pixel eye-to-nose distance along u_y
          - 'W_eyes': eye width
          - 'E_mid', 'u_x', 'u_y': face coordinate frame components
    """
    E_mid, u_x, u_y, W_eyes, roll_deg = build_face_frame(puntos)

    P33 = puntos[33].astype(np.float64)  # nose tip
    P8  = puntos[8].astype(np.float64)   # chin

    # Project nose tip and chin onto the face vertical axis u_y
    v_nose = P33 - E_mid
    d_top = float(np.dot(v_nose, u_y))          # vertical distance: eyes to nose
    d_bottom = float(np.dot(P8 - P33, u_y))     # vertical distance: nose to chin

    # Scale-invariant vertical position of nose tip
    eta = d_top / max(W_eyes, 1e-6)

    # Optional: horizontal nose offset (pure YAW indicator, for verification)
    d_lateral = float(np.dot(v_nose, u_x))
    yaw_proxy = d_lateral / max(W_eyes, 1e-6)

    # Pitch calculation from sagittal triangle
    if baseline is not None:
        delta_eta = eta - baseline["eta"]
        # Sagittal triangle: theta = -arctan(delta_eta / kp)
        # Negative sign ensures:
        # Looking DOWN -> d_top increases -> delta_eta > 0 -> theta_pitch < 0 (-)
        # Looking UP   -> d_top shrinks   -> delta_eta < 0 -> theta_pitch > 0 (+)
        pitch_deg = float(-np.degrees(np.arctan2(delta_eta, KP_PITCH)))
    else:
        pitch_deg = 0.0

    return {
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "eta": eta,
        "d_top": d_top,
        "d_bottom": d_bottom,
        "yaw_proxy": yaw_proxy,
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
        self.eta_buf = []
        self.roll_buf = []
        self.baseline = None

    @property
    def is_calibrated(self):
        return self.baseline is not None

    @property
    def progress(self):
        return min(len(self.eta_buf) / self.n_frames, 1.0)

    def accumulate(self, metrics):
        self.eta_buf.append(metrics["eta"])
        self.roll_buf.append(metrics["roll_deg"])

        if len(self.eta_buf) >= self.n_frames:
            self.baseline = {
                "eta":  float(np.median(self.eta_buf)),
                "roll": float(np.median(self.roll_buf)),
            }
            return self.baseline
        return None


# ── Classification ──────────────────────────────────────────────────────────

class FatigueClassifier:
    """
    Classifies pitch (looking down = droop) and roll deviations.
    Sustained negative pitch indicates fatigue/microsleep.
    """

    def __init__(self, sustained=SUSTAINED_FRAMES):
        self.sustained = sustained
        self.pitch_counter = 0
        self.roll_counter = 0

    def classify(self, pitch_deg, roll_deg, baseline):
        delta_roll = roll_deg - baseline["roll"]

        # ── Pitch classification (focus on downward droop) ──
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

        return pitch_level, roll_level, delta_roll


# ── Visualization Helpers ───────────────────────────────────────────────────

LEVEL_COLORS = {
    "NORMAL":           (0, 255, 0),    # green
    "MILD_DROOP":       (0, 200, 255),  # yellow
    "FATIGUE_WARNING":  (0, 0, 255),    # red
    "MILD_TILT":        (0, 200, 255),
    "SIGNIFICANT_TILT": (0, 0, 255),
}


def draw_face_axes(frame, metrics, pitch_level):
    """Draw the face-aligned intrinsic coordinate frame and nose projection."""
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


def draw_hud(frame, metrics, pitch_level, roll_level, delta_roll):
    """Draw HUD with live pitch/roll values and alerts."""
    h, w = frame.shape[:2]
    pitch_deg = metrics["pitch_deg"]
    roll_deg = metrics["roll_deg"]

    # Status lines
    lines = [
        f"PITCH: {pitch_deg:+6.1f} deg  [{pitch_level}]  (nodding up +, droop -)",
        f"ROLL : {roll_deg:+6.1f} deg  (delta: {delta_roll:+5.1f} deg)  [{roll_level}]",
        f"eta: {metrics['eta']:.3f} | scale: {metrics['W_eyes']:.0f}px",
    ]

    y0 = h - 25 * len(lines) - 15
    for i, line in enumerate(lines):
        y = y0 + i * 25
        # Background shadow for clarity
        cv2.putText(frame, line, (17, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        text_color = LEVEL_COLORS.get(pitch_level if i == 0 else (roll_level if i == 1 else "NORMAL"))
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1)

    # Flashing alert banner on severe fatigue
    if pitch_level == "FATIGUE_WARNING":
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, "FATIGUE DETECTED: HEAD DROOPING!",
                    (w // 2 - 190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


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
            pitch_lvl, roll_lvl, d_roll = classifier.classify(
                metrics["pitch_deg"], metrics["roll_deg"], calibrator.baseline
            )
            draw_face_axes(frame, metrics, pitch_lvl)
            draw_hud(frame, metrics, pitch_lvl, roll_lvl, d_roll)

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
