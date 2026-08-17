"""
MediaPipe Face Landmarker & Iris Tracking
=========================================
Real-time facial landmark and iris detection using MediaPipe Tasks Vision API.
Provides 468 standard facial landmarks + 10 refined iris landmarks (478 total)
for face detection, eye contour extraction, and eye tracking without needing
separate Face ROI cropping.

Features:
  - 478 3D facial landmarks with sub-millimeter precision.
  - Configurable Iris tracking toggle (enable/disable iris detection and rendering).
  - Toggles for face tessellation mesh and feature contours (eyes, lips, eyebrows).
  - Real-time interactive controls via keyboard shortcuts.
  - Modular `FaceLandmarker` class reusable in downstream drowsiness and distraction pipelines.

How to Run:
  1. Install dependencies (using existing environment):
       C:\\Users\\joabl\\anaconda3\\envs\\pef\\python.exe -m pip install -r requirements.txt
  2. Run the script:
       C:\\Users\\joabl\\anaconda3\\envs\\pef\\python.exe face_landmarker.py

Interactive Keyboard Controls:
  • 'i' / 'I' : Toggle Iris detection / visualization ON/OFF (Eye Tracking toggle)
  • 't' / 'T' : Toggle Full Face Mesh Tessellation ON/OFF
  • 'c' / 'C' : Toggle Face & Eye Contours ON/OFF
  • 'h' / 'H' : Toggle On-Screen HUD / Help Overlay
  • 'q' / ESC : Quit application
"""

import argparse
import time
import os
import urllib.request
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarker:
    """Wrapper around MediaPipe FaceLandmarker for facial landmark and iris detection.

    Attributes:
        max_num_faces (int): Maximum number of faces to detect.
        draw_tesselation (bool): Whether full face mesh tessellation is rendered.
        draw_contours (bool): Whether face, eye, and lip contours are rendered.
        draw_iris (bool): Whether iris landmarks and connections are rendered.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_iris: bool = True,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.draw_tesselation = draw_tesselation
        self.draw_contours = draw_contours
        self.draw_iris = draw_iris

        self._mp_drawing = mp.tasks.vision.drawing_utils
        self._mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.refine_landmarks = True  # Always true in the new API

        # Initialize the landmarker using the downloaded task model
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            print(f"Model not found. Downloading face_landmarker.task to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, image_rgb: np.ndarray, timestamp_ms: int):
        """Processes an RGB image and returns facial landmark results.

        Args:
            image_rgb: RGB image as a NumPy ndarray.
            timestamp_ms: Current timestamp of the frame in milliseconds.

        Returns:
            FaceLandmarkerResult.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self._face_landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks_on_image(
        self,
        annotated_image: np.ndarray,
        detection_result,
    ) -> np.ndarray:
        """Draws configured face mesh elements onto the provided image in-place.

        Args:
            annotated_image: BGR or RGB image (NumPy ndarray) to draw on.
            detection_result: Result object from FaceLandmarker.detect_for_video().

        Returns:
            Annotated image with landmarks drawn.
        """
        if not detection_result.face_landmarks:
            return annotated_image

        for face_landmarks in detection_result.face_landmarks:
            if self.draw_tesselation:
                self._mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

            if self.draw_contours:
                self._mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            if self.draw_iris:
                self._mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
                self._mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        return annotated_image

    def close(self) -> None:
        """Releases MediaPipe resources."""
        self._face_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_faces_detected: int,
    landmarker: FaceLandmarker,
    show_help: bool = True,
) -> None:
    """Renders status HUD and keyboard shortcuts overlay onto frame."""
    overlay = frame.copy()
    box_height = 140 if show_help else 70
    cv2.rectangle(overlay, (10, 10), (380, box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title & FPS
    cv2.putText(
        frame,
        f"MediaPipe Face Landmarker | FPS: {fps:.1f}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Detection status
    iris_active = landmarker.draw_iris and landmarker.refine_landmarks
    iris_status = "ON (Tracking)" if iris_active else "OFF"
    iris_color = (0, 255, 0) if iris_active else (0, 0, 255)

    cv2.putText(
        frame,
        f"Faces: {num_faces_detected}  |  Iris Tracking: {iris_status}",
        (20, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        iris_color,
        1,
        cv2.LINE_AA,
    )

    if show_help:
        tess_status = "ON" if landmarker.draw_tesselation else "OFF"
        cont_status = "ON" if landmarker.draw_contours else "OFF"

        help_text_1 = f"[I] Toggle Iris ({iris_status}) | [T] Mesh ({tess_status}) | [C] Contours ({cont_status})"
        help_text_2 = "[H] Toggle Help Overlay | [Q/ESC] Quit"

        cv2.putText(
            frame,
            help_text_1,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            help_text_2,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Landmarker & Iris Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Maximum number of faces to detect",
    )
    parser.add_argument(
        "--no-iris",
        action="store_true",
        help="Disable iris tracking / visualization by default",
    )
    parser.add_argument(
        "--no-tesselation",
        action="store_true",
        help="Disable full face mesh tessellation rendering",
    )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable face and eye contour rendering",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence value ([0.0, 1.0]) for face detection",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence value ([0.0, 1.0]) for landmark tracking",
    )
    return parser.parse_args()


def run_landmarker(
    camera_id: int = 0,
    max_num_faces: int = 1,
    enable_iris: bool = True,
    draw_tesselation: bool = True,
    draw_contours: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """Captures webcam stream and runs MediaPipe Face Mesh landmarker."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {camera_id}.")

    landmarker = FaceLandmarker(
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        draw_tesselation=draw_tesselation,
        draw_contours=draw_contours,
        draw_iris=enable_iris,
    )

    show_hud_help = True
    start_time = time.perf_counter()
    prev_time = start_time
    fps = 0.0

    window_name = "MediaPipe Face Landmarker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Mirror frame horizontally for natural UX
            frame = cv2.flip(frame, 1)

            # Performance calculation
            curr_time = time.perf_counter()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Timestamp calculation
            timestamp_ms = int((curr_time - start_time) * 1000)

            results = landmarker.process(rgb_frame, timestamp_ms)

            # Draw landmarks
            landmarker.draw_landmarks_on_image(frame, results)

            # Draw HUD
            num_faces = len(results.face_landmarks) if results.face_landmarks else 0
            draw_hud(frame, fps, num_faces, landmarker, show_help=show_hud_help)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("i"), ord("I")):
                landmarker.draw_iris = not landmarker.draw_iris
            elif key in (ord("t"), ord("T")):
                landmarker.draw_tesselation = not landmarker.draw_tesselation
            elif key in (ord("c"), ord("C")):
                landmarker.draw_contours = not landmarker.draw_contours
            elif key in (ord("h"), ord("H")):
                show_hud_help = not show_hud_help
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entrypoint function."""
    args = parse_args()
    run_landmarker(
        camera_id=args.camera,
        max_num_faces=args.max_faces,
        enable_iris=not args.no_iris,
        draw_tesselation=not args.no_tesselation,
        draw_contours=not args.no_contours,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


if __name__ == "__main__":
    main()
